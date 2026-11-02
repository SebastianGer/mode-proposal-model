import os
from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from ProbabilisticUnetPytorch.probabilistic_unet import ProbabilisticUnet
from ProbabilisticUnetPytorch.utils import l2_regularisation
from torch.optim import Adam
from tqdm import tqdm

from firespreadscenarios.utils.GED import compute_ged
from firespreadscenarios.utils.hm_iou import compute_hm_iou

from .utils import to_unit_interval


class BaseProbabilisticUNetLightning(pl.LightningModule, ABC):
    def __init__(
        self,
        input_channels: int,
        num_classes: int = 1,
        latent_dim: int = 6,
        beta: float = 1.0,
        dataset_evaluator_arg: Any = None,
        conditioning: Literal[
            "unconditional", "debug_ground_truth", "image", "segmentation_features"
        ] = "unconditional",
        aggregate_n_samples: int = 1,
        ignore_index: Optional[int] = None,  # dataset class index for class to ignore
        write_full_val_samples: bool = False,
        use_original_scheduler: bool = False,
        lr: float = 1e-4,
        base_channels: int = 32,
        num_updown_blocks: int = 4,
        probability_eps: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        num_filters = [
            base_channels,
            2 * base_channels,
            4 * base_channels,
            6 * base_channels,
            6 * base_channels,
            6 * base_channels,
            6 * base_channels,
        ]
        num_filters = num_filters[
            : (num_updown_blocks + 1)
        ]  # potentially truncate to desired number of blocks
        self.model = ProbabilisticUnet(
            input_channels=input_channels,
            num_classes=num_classes,
            latent_dim=latent_dim,
            beta=beta,
            num_filters=num_filters,
            probability_eps=probability_eps,
        )
        self.dataset_evaluator = self.get_dataset_evaluator(
            self.hparams.dataset_evaluator_arg
        )

        # Switches the validation step between a simple forward pass and sampling a larger batch.
        # Should be switched between training end and full validation.
        self.training_ended = False

        self.val_distribution_eval_partial_dfs = []

    @abstractmethod
    def get_dataset_evaluator(self):
        pass

    @abstractmethod
    def unbatch(self, batch):
        # Should return a dictionary with:
        # - image: B x C x H x W
        # - target: B x H x W, randomly chosen if multiple exist
        # - target_summary: B x H x W, contains some kind of summary of the target, e.g. the mean over all modes; used for visualization
        # - all_targets: B x N x H x W, all targets
        # Optionally:
        # - logging_target: B x H x W, the target that is visualized in logging, e.g. a filtered class only containing the flipping classes, for Cityscapes
        pass

    @abstractmethod
    def compute_dataset_specific_metrics(self, metrics_agg, batch_dict, i):
        pass

    def forward(self, batch, batch_idx=-1, step_type="??", *args, **kwargs):
        """Forward pass used during training of the diffusion model, including computation of loss."""

        batch_dict = self.batch_to_cond_and_mask(batch)
        self.model.forward(batch_dict["cond"], batch_dict["target"], training=True)
        elbo = self.model.elbo(batch_dict["target"])
        reg_loss = (
            l2_regularisation(self.model.posterior)
            + l2_regularisation(self.model.prior)
            + l2_regularisation(self.model.fcomb.layers)
        )
        loss = -elbo + 1e-5 * reg_loss

        x_pred = self.model.reconstruction

        # Debug logging
        if batch_idx == 0 and not self.training_ended and self.trainer.is_global_zero:
            b, c, h, w = x_pred.shape
            separating_line = torch.ones((b, c, h, 4), device=self.device) * 0.75
            merged_images = torch.cat(
                (
                    batch_dict["target_summary"][:, None],
                    separating_line,
                    x_pred.clamp(0, 1),
                ),
                dim=3,
            )

            wandb.log(
                {
                    f"In {step_type} forward step: Target summary | sample (clipped)": [
                        wandb.Image(image) for image in merged_images[:16]
                    ]
                }
            )
            print(
                f"In PUNet forward, logged reconstruction has min: {x_pred.min().item()} max: {x_pred.max().item()}"
            )

        return {"loss": loss, "x_pred": x_pred}

    def batch_to_cond_and_mask(self, batch, move_to_device=False):
        """
        Takes a batch of conditioning information, desired output, and cluster ids and returns the conditioning information and the segmentation map.
        """
        # rgb_image, seg_map_randomly_binarized, seg_map_deterministicly_binarized, full_seg_map
        batch_dict = self.unbatch(batch)

        # Make sure we have a channel dimension, not just B x H x W
        if len(batch_dict["target"].shape) == 3:
            batch_dict["target"] = batch_dict["target"].unsqueeze(1)
        if len(batch_dict["image"].shape) == 3:
            batch_dict["image"] = batch_dict["image"].unsqueeze(1)

        # Transform from long to float
        batch_dict["target"] = batch_dict["target"].float()

        # Compute segmentation features
        cond = None
        match self.hparams.conditioning:
            case "debug_ground_truth":
                cond = batch_dict["target"]
            case "unconditional":
                cond = None
            case "image":
                cond, _ = self.replace_ignored_idx(batch_dict["image"])

        batch_dict["cond"] = cond.float()

        return batch_dict

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, step_type="train", batch_idx=batch_idx)["loss"]

        self.log(
            "train_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self.batch_to_cond_and_mask(batch)

        loss = self.forward(batch, step_type="train", batch_idx=batch_idx)["loss"]
        self.log(
            "val_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        distribution_eval_df = None
        if self.training_ended:

            evaluation_results = self.compute_tvd(
                batch_dict, segment="val", batch_idx=batch_idx
            )
            distribution_eval_df = evaluation_results["distribution_eval_df"]

            self.val_distribution_eval_partial_dfs.append(distribution_eval_df)

            if batch_idx == 0 and self.trainer.is_global_zero:
                generated_segmaps = {
                    "seg_map_preds": evaluation_results["quality_inspection_x_preds"],
                    "target_summary": evaluation_results["quality_inspection_seg_map"],
                }
                self.log_val_quality_inspection(generated_segmaps)

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        batch_dict = self.batch_to_cond_and_mask(batch)
        self.compute_tvd(batch_dict, segment="val", batch_idx=batch_idx)

        return float("NaN")

    def compute_tvd(self, batch_dict, segment: Literal["val", "test"], batch_idx=99999):
        """
        Computes the total variation distance sbetween the true mode distribution and the mode distributio in the generated samples. Also generates these samples.
        """

        batch_size = self.hparams.aggregate_n_samples
        n_reps = 1  # could be added as an argument if needed; easier to leave the structure in for now than to recreate it completely if it's needed

        # Debug: Condition on GT instead of feature map
        if self.hparams.conditioning == "debug_ground_truth":
            raise NotImplementedError(
                "This is not implemented. It does not make much sense for this model."
            )
        elif self.hparams.conditioning == "unconditional":
            cond = None
            self.hparams.batch_size = batch_dict["image"].shape[
                0
            ]  # Set batch size, since the sampling function can't get it from anywhere else
        else:
            cond = batch_dict["cond"]

        dataframe_agg = []
        quality_inspection_x_preds = None
        quality_inspection_full_seg_map = None

        # Replicate each item in the batch batch_size times and run the diffusion model on the batch,
        # this way we can get a good estimation of the distribution of the generated samples by having a lot of samples per conditioning.
        for i in tqdm(
            range(len(cond)), desc=f"Sampling individual items for Batch {batch_idx}"
        ):
            hm_ious = []
            geds = []
            deduped_ious = []
            dedupued_individual_ious = []
            metrics_agg = []

            # Replicate the i-th item batch_size times, then run the evaluation on the batch
            replicated_cond = cond[i].repeat(batch_size, 1, 1, 1)

            dataset_eval_input = batch_dict["dataset_eval_input"][i]
            if len(dataset_eval_input.shape) == 3:
                replicated_full_seg_map = dataset_eval_input[None].repeat(
                    batch_size, 1, 1, 1
                )
            else:
                replicated_full_seg_map = dataset_eval_input.repeat(batch_size, 1, 1)

            # Internally compute UNet features and latent distribution
            self.model.forward(replicated_cond, None, training=False)

            for j in range(n_reps):

                x_pred = self.model.sample(testing=True)
                binarized_predictions = (x_pred > 0.5).int()

                if self.dataset_evaluator is not None:
                    metrics = self.dataset_evaluator.compute_metrics(
                        binarized_predictions, replicated_full_seg_map
                    )
                    metrics_agg.append(metrics)

                possible_modes = batch_dict["all_targets"][i, :, None]  # N x 1 x H x W

                ## NOTE: HM IoU and GED are only computed on one batch of data. Any repetitions are evaluated separately.
                # This could be altered, but needs to consider possible memory issues.
                # The dataset_evaluator instead aggregates the metrics over all samples, because the intermediate results to store are much smaller.
                ged = compute_ged(possible_modes, binarized_predictions)["ged"]
                hm_iou = compute_hm_iou(possible_modes, binarized_predictions)
                deduped_iou, dedupued_individual_iou_vals = compute_hm_iou(
                    possible_modes.unique(dim=0),
                    binarized_predictions,
                    return_ious=True,
                )
                geds.append(ged)
                hm_ious.append(hm_iou)
                deduped_ious.append(deduped_iou)
                dedupued_individual_ious.append(dedupued_individual_iou_vals)

                if quality_inspection_x_preds is None:
                    quality_inspection_x_preds = binarized_predictions
                    quality_inspection_full_seg_map = batch_dict["target_summary"][
                        i
                    ].repeat(batch_size, 1, 1, 1)

                if self.hparams.write_full_val_samples:
                    torch.save(
                        binarized_predictions,
                        os.path.join(wandb.run.dir, f"val_preds_batch{batch_idx}.pt"),
                    )

            dataset_specific_metrics_dict = self.compute_dataset_specific_metrics(
                metrics_agg, batch_dict, i
            )
            # Generated clusters is useless if we compute it based on [B,1,H,W], becase it is computed across the 1 sample dimension
            df_dict = {
                "GED": torch.stack(geds).mean().cpu().item(),
                "HM IoU": torch.stack(hm_ious).mean().cpu().item(),
                "HM IoU (deduped)": torch.stack(deduped_ious).mean().cpu().item(),
            }
            individual_ious = torch.stack(dedupued_individual_ious).mean(dim=0).cpu()
            df_individual_ious = {
                "HM IoU class " + str(i): individual_ious[i].item()
                for i in range(len(individual_ious))
            }
            df_dict.update(df_individual_ious)

            self.log_dict(df_dict)
            self.log_dict(dataset_specific_metrics_dict)

            distribution_eval_df = pd.DataFrame(
                df_dict | dataset_specific_metrics_dict, index=[i]
            )
            dataframe_agg.append(distribution_eval_df)

        distribution_eval_df = pd.concat(dataframe_agg, ignore_index=True)
        self.log_dict(distribution_eval_df.mean().to_dict())
        return {
            "distribution_eval_df": distribution_eval_df,
            "quality_inspection_x_preds": quality_inspection_x_preds,
            "quality_inspection_seg_map": quality_inspection_full_seg_map,
        }

    def log_val_quality_inspection(self, generated_segmaps):
        # Merge the ground truth and the prediction into one image, separated by a white line for easier visual inspection
        if not self.trainer.is_global_zero:
            return
        seg_map_preds = generated_segmaps["seg_map_preds"]
        target_summary = generated_segmaps["target_summary"]

        b, c, h, w = seg_map_preds.shape
        separating_line = torch.ones((b, c, h, 4), device=self.device) * 0.75

        full_seg_map_ = to_unit_interval(target_summary)
        merged_images = torch.cat(
            (full_seg_map_, separating_line, seg_map_preds), dim=3
        )

        wandb.log(
            {"sampled_images": [wandb.Image(image) for image in merged_images[:32]]}
        )

    def on_validation_epoch_end(self) -> None:
        """_summary_ Log the val PR curve after predicting all val samples."""

        # Aggregate partial results of the distribution evaluation into one dataframe and save it to disk
        if (
            self.hparams.aggregate_n_samples > 1
            and len(self.val_distribution_eval_partial_dfs) > 0
        ):
            all_dfs = self.all_gather(self.val_distribution_eval_partial_dfs)

            if self.trainer.is_global_zero:
                df = pd.concat(all_dfs, ignore_index=True)
                df.to_csv(os.path.join(wandb.run.dir, "val_distribution_eval.csv"))
                self.log_dict(df.mean(skipna=True).to_dict())

        self.val_distribution_eval_partial_dfs = []

    def replace_ignored_idx(self, x):
        # Replace ignore idx with 0.
        if self.hparams.ignore_index is not None:
            valid_index_map = x != self.hparams.ignore_index
            return (
                torch.where(valid_index_map, x, torch.full_like(x, 0)),
                valid_index_map,
            )
        return x, torch.ones_like(x)

    def configure_optimizers(self):
        optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.lr,
        )
        if self.hparams.use_original_scheduler:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=[  # Taken from Prob. UNet supplementary, milestones measured in epochs on LIDC, not steps
                    170,
                    340,
                    510,
                ],
                gamma=0.5,
            )
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.trainer.max_steps,
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

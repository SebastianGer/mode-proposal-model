import os
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from edm.generate import timestep_schedule
from edm.loss import EDMLoss2
from edm.networks import EDMPrecondImageCond
from matplotlib import pyplot as plt

from .sampling.generate_diverse import edm_sampler2
from .utils import (
    from_pm1_to_unit_interval,
    from_unit_interval_to_pm1,
    to_unit_interval,
)


class BaseDiffusionTraining(pl.LightningModule, ABC):
    """
    Base model class for diffusion model training for segmentation tasks. Needs to be subclassed with dataset-specific implementations.
    """

    def __init__(
        self,
        image_shape: int | Tuple[int, int] = (64, 128),
        timesteps: int = 10,
        asymmetric_time_difference: float = 0.0,
        channels: int = 3,
        aggregate_n_samples: int = 1,
        conditioning: Literal[
            "unconditional", "debug_ground_truth", "image", "segmentation_features"
        ] = "unconditional",
        use_random_unconditional_conditioning: bool = False,
        step_unrolled_denoising_start_epoch: int = 99999999,  # Index of epoch after which to start unrolling the denoising steps in training. Starts with 0.
        log_all_val_denoising_steps: bool = False,
        ignore_index: Optional[int] = None,  # dataset class index for class to ignore
        write_full_val_samples: bool = False,
        sigma_data: float = 0.0463,
        train_p_log_mean: float = -1.2,
        rho: float = 7,
        sigma_max: float = 80,
        compute_loss_per_noise_level: bool = False,
        s_churn: float = 0.0,
        s_max: float = float("inf"),
        s_min: float = 0.0,
        use_second_order_correction: bool = False,
        conditioning_noise_factor: float = 0.0,
        conditioning_noise_prob: float = 0.0,
        model_channels: int = 128,  # base number of features in model,
        dataset_evaluator_arg: Any = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_function = EDMLoss2(
            sigma_data=self.hparams.sigma_data,
            P_mean=self.hparams.train_p_log_mean,
            conditioning_noise_factor=self.hparams.conditioning_noise_factor,
            conditioning_noise_prob=self.hparams.conditioning_noise_prob,
        )

        self.backbone = self.load_backbone()
        self.diffusion_model = self.build_diffusion_model()

        self.val_distribution_eval_partial_dfs = []
        self.val_loss_per_noise_level_dfs = []

        # Needed in evaluation; Needs to be implemented in subclasses
        self.dataset_evaluator = self.get_dataset_evaluator(
            self.hparams.dataset_evaluator_arg
        )

    def build_diffusion_model(self):
        # Builds a freshly initialized diffusion network from the model hyperparameters.
        # Must only be called after load_backbone(), which adjusts hparams.channels for
        # some conditioning modes. Also used by BaseModePredictionNetwork to rebuild the
        # architecture with fresh weights for freeze_layers="retrain_from_scratch".
        channel_mult = [2, 2, 2]
        # Attention layer should always be at lowest resolution
        attention_at_dim = min(self.hparams.image_shape) / 2 ** (len(channel_mult) - 1)
        cond_dropout = (
            0.1 if self.hparams.use_random_unconditional_conditioning else 0.0
        )
        return EDMPrecondImageCond(
            img_resolution=self.hparams.image_shape,  # Image resolution.
            img_channels=1,  # Number of color channels.
            cond_dim=self.hparams.channels,  # Number of class labels, 0 = unconditional.
            use_fp16=False,  # Execute the underlying model at FP16 precision?
            sigma_min=0,  # Minimum supported noise level.
            sigma_max=80,  # Maximum supported noise level. Taken from edm_sampler, not from the model.
            sigma_data=self.hparams.sigma_data,  # Expected standard deviation of the training data.
            model_type="SongUNetImageCond",  # Class name of the underlying model.)
            cond_dropout=cond_dropout,
            encoder_type="residual",  # NCSN++
            decoder_type="standard",  # NCSN++
            resample_filter=[
                1,
                3,
                3,
                1,
            ],  # NCSN++, bilinear resampling, according to the EDM appendix
            embedding_type="fourier",
            channel_mult_noise=2,
            channel_mult=channel_mult,
            attn_resolutions=[attention_at_dim],
            model_channels=self.hparams.model_channels,
            custom_model_conditioning=True,
        )

    @abstractmethod
    def get_dataset_evaluator(self):
        pass

    def load_backbone(self):
        # Load segmentation model and set number of channels of the conditioning information that is passed to the diffusion model.

        match self.hparams.conditioning:
            case "image":
                return None
            case "debug_ground_truth":
                self.hparams.channels = 1
                return None
            case "unconditional":
                self.hparams.channels = 0
                return None
            case "segmentation_features":
                backbone, channels = self.load_concrete_backbone_model()
                self.hparams.channels = channels

                # Freeze the backbone
                for param in backbone.parameters():
                    param.requires_grad = False

                return backbone

    @abstractmethod
    def load_concrete_backbone_model(self):
        pass

    def forward(
        self, seg_map_gt_, cond=None, step_type="", fixed_sigma=None, *args, **kwargs
    ):
        """Forward pass used during training of the diffusion model, including computation of loss."""

        batch, c, h, w = (*seg_map_gt_.shape,)
        seg_map_gt_binary, _ = self.replace_ignored_idx(seg_map_gt_)

        # Transform input segmentation map to [-1,1]
        seg_map_gt_pm1 = from_unit_interval_to_pm1(seg_map_gt_binary)

        # Decide whether to use the ground truth as target or the prediction from the previous step
        use_step_unrolling = False
        if (
            step_type == "train"
            and self.current_epoch >= self.hparams.step_unrolled_denoising_start_epoch
        ):
            use_step_unrolling = True

        loss, x_pred, noisy_input = self.loss_function(
            self.diffusion_model,
            seg_map_gt_pm1,
            conditioning=cond,
            fixed_sigma=fixed_sigma,
            use_step_unrolling=use_step_unrolling,
        )

        # Debug logging
        if kwargs.get("batch_idx", -1) == 0:
            print(
                f"\nIn forward {step_type} - Min/max/median: \nseg_map_gt: {seg_map_gt_pm1.min()} {seg_map_gt_pm1.max()} {seg_map_gt_pm1.median()}  \nnoise: {noisy_input.min()} {noisy_input.max()} {noisy_input.median()} \nx_pred: {x_pred.min()} {x_pred.max()} {x_pred.median()}"
            )
            b, c, h, w = x_pred.shape
            separating_line = torch.ones((b, c, h, 4), device=self.device) * 0.75
            noisy_input = to_unit_interval(noisy_input)
            pred_ = from_pm1_to_unit_interval(x_pred)
            merged_images = torch.cat(
                (
                    seg_map_gt_binary,
                    separating_line,
                    noisy_input,
                    separating_line,
                    pred_,
                ),
                dim=3,
            )

            wandb.log(
                {
                    f"In {step_type} forward step: GT|noisy input|pred_x0": [
                        wandb.Image(image) for image in merged_images[:16]
                    ]
                }
            )

        return {"loss": loss, "x_pred": x_pred, "noisy_input": noisy_input}

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
                cond = from_unit_interval_to_pm1(batch_dict["target"])
            case "unconditional":
                cond = None
            case "image":
                cond = batch_dict["image"]
            case "segmentation_features":
                with torch.no_grad():
                    # Only works for detectron backbone for now, might need to be factored out if others are used.
                    backbone_features = self.backbone(batch_dict["image"])
                    backbone_features = [
                        backbone_features[stage] for stage in ["res2", "res3", "res5"]
                    ]

                    target_size = backbone_features[0].shape[2:]

                    upsampled_features = [
                        F.interpolate(
                            f, size=target_size, mode="bilinear", align_corners=False
                        )
                        for f in backbone_features
                    ]

                    # Concatenate along the channel dimension
                    cond = torch.cat(upsampled_features, dim=1)

        batch_dict["cond"] = cond

        return batch_dict

    def training_step(self, batch, batch_idx):
        batch_dict = self.batch_to_cond_and_mask(batch)

        loss = self.forward(
            batch_dict["target"],
            batch_dict["cond"],
            batch_idx=batch_idx,
            step_type="train",
        )["loss"]
        self.log(
            "train_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self.batch_to_cond_and_mask(batch)

        # Only done after training, not during regular validation
        if self.hparams.compute_loss_per_noise_level:
            loss_per_noise_level_batch = self.compute_loss_per_noise_level(
                batch_dict["target"], batch_dict["cond"]
            )
            self.val_loss_per_noise_level_dfs.append(loss_per_noise_level_batch)

        # Regular loss computation during training
        loss = self.forward(
            batch_dict["target"],
            batch_dict["cond"],
            batch_idx=batch_idx,
            step_type="val",
        )["loss"]
        self.log(
            "val_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )

        # Only log the first batch's images for quality inspection.
        if batch_idx == 0:

            # Run diffusion model on the segmentation features and sample a segmentation map
            generated_segmaps = self.generate_segmap_and_evaluate(
                batch_dict["cond"],
                batch_dict["target"],
                segment="val",
                batch_idx=batch_idx,
            )

            generated_segmaps["target_summary"] = batch_dict[
                "target_summary"
            ].unsqueeze(1)
            self.log_val_quality_inspection(generated_segmaps)

        return loss

    def log_val_quality_inspection(self, generated_segmaps):
        # Merge the ground truth and the prediction into one image, separated by a white line for easier visual inspection

        seg_map_preds = generated_segmaps["seg_map_preds"]
        seg_map_preds_steps = generated_segmaps["seg_map_preds_stepwise"]
        target_summary = generated_segmaps["target_summary"]

        if (
            seg_map_preds is None
            or seg_map_preds_steps is None
            or target_summary is None
        ):
            return

        b, c, h, w = seg_map_preds.shape
        separating_line = torch.ones((b, c, h, 4), device=self.device) * 0.75

        full_seg_map_ = to_unit_interval(target_summary)
        merged_images = torch.cat(
            (full_seg_map_, separating_line, seg_map_preds), dim=3
        )

        wandb.log(
            {"sampled_images": [wandb.Image(image) for image in merged_images[:32]]}
        )

        # Log the evolution of a predicted segmentation map over the steps of the DDIM denoising process
        # Separate wandb calls for separate "step" parameters in wandb UI, to make it easier to compare images
        if self.hparams.log_all_val_denoising_steps and seg_map_preds_steps is not None:
            for i in range(min(seg_map_preds_steps.shape[1], 32)):
                wandb.log(
                    {
                        "sampled_images_stepwise": [wandb.Image(full_seg_map_[i])]
                        + [wandb.Image(image) for image in seg_map_preds_steps[:, i]]
                    }
                )

    def compute_loss_per_noise_level(self, seg_map_gt, seg_features):
        # Manually chosen as a good balance between high and low noise levels
        res = {}
        time_steps = self.get_eval_noise_level_schedule().to(self.device)
        for sigma in time_steps:

            with torch.no_grad():
                sigma_ = torch.full(
                    (seg_map_gt.shape[0], 1, 1, 1), sigma.item(), device=self.device
                )
                loss = self.forward(
                    seg_map_gt,
                    seg_features,
                    batch_idx=-1,
                    step_type="None",
                    fixed_sigma=sigma_,
                )["loss"]
            res[sigma.item()] = loss.item()
        res_df = pd.DataFrame(res, index=[0])
        return res_df

    def generate_segmap_and_evaluate(
        self, cond_, seg_map_gt_, segment: Literal["val", "test"], batch_idx=99999
    ):
        """
        Generate a segmentation map from the given segmentation features and evaluate it against the ground truth.
        Return the generated batch of segmentation predictions.
        """
        # Debug: Condition on GT instead of feature map
        if self.hparams.conditioning == "debug_ground_truth":
            cond = from_unit_interval_to_pm1(seg_map_gt_)
        elif self.hparams.conditioning == "unconditional":
            cond = None
            self.hparams.batch_size = seg_map_gt_.shape[
                0
            ]  # Set batch size, since the sampling function can't get it from anywhere else
        else:
            cond = cond_

        # Sample a segmentation map from the diffusion model
        seg_map_preds_agg = []
        seg_map_preds_stepwise = None
        for i in range(self.hparams.aggregate_n_samples):

            latents = torch.randn(
                (seg_map_gt_.shape[0], 1, *self.hparams.image_shape), device=self.device
            )
            sampling_result = edm_sampler2(
                self.diffusion_model,
                latents,
                conditioning=cond,
                rho=self.hparams.rho,
                num_steps=self.hparams.timesteps,
                sigma_max=self.hparams.sigma_max,
                S_churn=self.hparams.s_churn,
                S_max=self.hparams.s_max,
                S_min=self.hparams.s_min,
                use_second_order_correction=self.hparams.use_second_order_correction,
                asymmetric_time_difference=self.hparams.asymmetric_time_difference,
                conditioning_noise_factor=self.hparams.conditioning_noise_factor,
            )
            seg_map_preds_agg.append(sampling_result["x_next"])

            # Only log the first set of generated images for quality inspection,
            # otherwise we'd have to take the mean of all generated images, which would likely not lead to
            # more reasonable results, or we'd have to log all generated images, which would be too much.
            if i == 0 and self.hparams.log_all_val_denoising_steps:
                seg_map_preds_stepwise = torch.stack(sampling_result["x_steps"])

        seg_map_preds_agg = torch.stack(seg_map_preds_agg, dim=1)  # B x N x C x H x W
        seg_map_preds_agg = from_pm1_to_unit_interval(seg_map_preds_agg)
        seg_map_preds = seg_map_preds_agg.mean(dim=1)

        if self.hparams.log_all_val_denoising_steps:
            seg_map_preds_stepwise = from_pm1_to_unit_interval(seg_map_preds_stepwise)

        if self.hparams.write_full_val_samples:
            torch.save(
                seg_map_preds_agg,
                os.path.join(wandb.run.dir, f"val_preds_batch{batch_idx}.pt"),
            )

        return {
            "seg_map_preds": seg_map_preds,
            "seg_map_preds_stepwise": seg_map_preds_stepwise,
        }

    def test_step(self, batch, batch_idx):
        batch_dict = self.batch_to_cond_and_mask(batch)

        # Run diffusion model on the segmentation features and sample a segmentation map
        self.generate_segmap_and_evaluate(
            batch_dict["cond"], batch_dict["target"], segment="test"
        )

    def on_validation_epoch_end(self) -> None:
        """_summary_ Log the val PR curve after predicting all val samples."""

        # Aggregate partial results of the distribution evaluation into one dataframe and save it to disk
        if (
            self.hparams.aggregate_n_samples > 1
            and len(self.val_distribution_eval_partial_dfs) > 0
        ):
            df = pd.concat(self.val_distribution_eval_partial_dfs, ignore_index=True)
            df.to_csv(os.path.join(wandb.run.dir, "val_distribution_eval.csv"))
            self.log_dict(df.drop("id", axis=1).mean(skipna=True).to_dict())

        self.val_distribution_eval_partial_dfs = []

        if self.hparams.compute_loss_per_noise_level:
            df = pd.concat(self.val_loss_per_noise_level_dfs, ignore_index=True)
            df_path = os.path.join(wandb.run.dir, "val_loss_per_noise_level.csv")
            df.to_csv(df_path)

            # Plot, save to wandb
            df = pd.read_csv(df_path)
            mean_values = df.mean()[1:].round(4)

            ts = self.get_eval_noise_level_schedule().cpu().numpy().tolist()
            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(ts, mean_values, marker="o")
            plt.xlabel("Noise std")
            plt.ylabel("Loss")
            plt.title("Mean loss per noise level")
            plt.grid(True)
            print("Loss per noise level: ", mean_values)
            wandb.log(
                {
                    "loss_per_noise_level.png": plt,
                    "mean_loss_per_noise_level": df.mean()[1:].mean(),
                }
            )

        self.val_loss_per_noise_level_dfs = []

    def get_eval_noise_level_schedule(self):
        return timestep_schedule(sigma_max=80, sigma_min=0.002, num_steps=20, rho=3)

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-4,
            betas=(0.9, 0.99),
        )

    def replace_ignored_idx(self, x):
        # Replace ignore idx with 0.
        if self.hparams.ignore_index is not None:
            valid_index_map = x != self.hparams.ignore_index
            return (
                torch.where(valid_index_map, x, torch.full_like(x, 0)),
                valid_index_map,
            )
        return x, torch.ones_like(x)

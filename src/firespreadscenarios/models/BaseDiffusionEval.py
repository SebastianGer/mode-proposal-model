import os
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils
import wandb
from tqdm.auto import tqdm

from firespreadscenarios.utils.GED import compute_ged
from firespreadscenarios.utils.hm_iou import compute_hm_iou

from .BaseDiffusionTraining import BaseDiffusionTraining
from .sampling.sampling_strategies.SamplingStrategy import SamplingStrategy
from .utils import from_pm1_to_unit_interval, to_unit_interval


class BaseDiffusionEval(BaseDiffusionTraining, ABC):
    """
    Base model class for diffusion model evaluation on segmentation tasks. Needs to be subclassed with dataset-specific implementations.

    Separating training and evaluation into two classes seemed simpler, since many sampling-related parameters aren't needed during training,
    and the what is required during the validation_step is quite different between training and validation phase. There are likely more elegant ways to do this.
    """

    def __init__(
        self,
        # Basic model and training parameters
        model_channels=128,  # base number of features in model
        image_shape: int | Tuple[int, int] = (64, 128),
        channels: int = 3,
        conditioning: Literal[
            "unconditional", "debug_ground_truth", "image", "segmentation_features"
        ] = "unconditional",
        use_random_unconditional_conditioning: bool = False,
        conditioning_noise_factor: float = 0.0,
        train_p_log_mean: float = -1.2,
        use_uniform_noise_level: bool = False,  # set maximum noise via sigma_max, only applied in training
        sigma_data: float = 0.0463,
        custom_model_conditioning: bool = False,  # switch between official c-parameters and simplified version for our setup
        ignore_index: Optional[int] = None,  # dataset class index for class to ignore
        compute_loss_per_noise_level: bool = False,
        # Sampling parameters
        timesteps: int = 10,
        rho: float = 7,
        use_second_order_correction: bool = False,
        asymmetric_time_difference: float = 0.0,
        sigma_max: float = 80,
        s_churn: float = 0.0,
        s_max: float = float("inf"),
        s_min: float = 0.0,
        aggregate_n_samples: int = 1,
        aggregate_samples_repetitions: int = 1,
        sampling_strategy: (
            SamplingStrategy | None
        ) = None,  # Sampling strategy to use, e.g. Particle Guidance
        log_all_val_denoising_steps: bool = False,
        write_full_val_samples: Literal["x_steps", "x_next"] | None = None,
        write_samples_dir: Optional[str] = None,
        dataset_evaluator_arg: Any = None,
        # Checkpoints saved by BaseDiffusionTraining subclasses store hyperparameters that
        # only the training class knows (e.g. step_unrolled_denoising_start_epoch). Accept
        # and ignore them, so such checkpoints can be loaded with the eval classes via
        # load_from_checkpoint, whose subclasses forward all checkpoint hyperparameters here.
        **training_only_hparams,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.test_distribution_eval_partial_dfs = []

        if training_only_hparams:
            print(
                "BaseDiffusionEval: ignoring training-only hyperparameters: "
                f"{sorted(training_only_hparams)}"
            )

        self.sampling_strategy = sampling_strategy

    def validation_step(self, batch, batch_idx):
        batch_dict = self.batch_to_cond_and_mask(batch)

        evaluation_results = self.compute_tvd(
            batch_dict, segment="val", batch_idx=batch_idx
        )
        distribution_eval_df = evaluation_results["distribution_eval_df"]

        self.val_distribution_eval_partial_dfs.append(distribution_eval_df)

        # Only log the first batch's images for quality inspection.
        if batch_idx == 0:
            generated_segmaps = {
                "seg_map_preds_stepwise": None,
                "seg_map_preds": evaluation_results["quality_inspection_x_preds"],
                "target_summary": evaluation_results["quality_inspection_seg_map"],
            }
            self.log_val_quality_inspection(generated_segmaps)
        return float("NaN")

    def on_validation_epoch_end(self) -> None:
        """_summary_ Log the val PR curve after predicting all val samples."""
        # Aggregate partial results of the distribution evaluation into one dataframe and save it to disk
        if (
            self.hparams.aggregate_n_samples > 1
            and len(self.val_distribution_eval_partial_dfs) > 0
        ):
            df = pd.concat(self.val_distribution_eval_partial_dfs, ignore_index=True)
            df.to_csv(os.path.join(wandb.run.dir, "val_distribution_eval.csv"))

        self.val_distribution_eval_partial_dfs = []

    def test_step(self, batch, batch_idx):
        batch_dict = self.batch_to_cond_and_mask(batch)

        evaluation_results = self.compute_tvd(
            batch_dict, segment="test", batch_idx=batch_idx
        )
        distribution_eval_df = evaluation_results["distribution_eval_df"]

        self.test_distribution_eval_partial_dfs.append(distribution_eval_df)

        return float("NaN")

    def on_test_epoch_end(self) -> None:
        if (
            self.hparams.aggregate_n_samples > 1
            and len(self.test_distribution_eval_partial_dfs) > 0
        ):
            df = pd.concat(self.test_distribution_eval_partial_dfs, ignore_index=True)
            df.to_csv(os.path.join(wandb.run.dir, "test_distribution_eval.csv"))

        self.test_distribution_eval_partial_dfs = []

    def compute_tvd(self, batch_dict, segment: Literal["val", "test"], batch_idx=99999):
        """
        Computes the total variation distance sbetween the true mode distribution and the mode distributio in the generated samples. Also generates these samples.
        """

        batch_size = self.hparams.aggregate_n_samples
        n_reps = self.hparams.aggregate_samples_repetitions

        # Val metrics keep their historical unprefixed names (wandb sweeps and old runs
        # reference them); test metrics get a prefix so they don't collide with val.
        prefix = "test_" if segment == "test" else ""

        if self.hparams.write_samples_dir:
            os.makedirs(self.hparams.write_samples_dir, exist_ok=True)

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

        dataframe_agg, additional_metrics_agg = [], []
        quality_inspection_x_preds, quality_inspection_full_seg_map = None, None

        if self.hparams.write_full_val_samples:
            # Make sure we can match the saved data with the original data by saving the dataset indices.
            torch.save(
                batch_dict["idx"],
                os.path.join(wandb.run.dir, f"batch_idx_{batch_idx}_idc.pt"),
            )

        # Replicate each item in the batch n_reps times and run the diffusion model on the batch.
        # This way we can get a good estimation of the distribution of the generated samples by generating multiple samples per conditioning.
        for i in tqdm(
            range(len(cond)), desc=f"Sampling individual items for Batch {batch_idx}"
        ):
            binarized_predictions_list = []
            replicated_cond = cond[i].repeat(batch_size, 1, 1, 1)
            dataset_eval_input = batch_dict["dataset_eval_input"][i]
            possible_modes = batch_dict["all_targets"][i]

            if len(dataset_eval_input.shape) == 3:
                replicated_full_seg_map = dataset_eval_input[None].repeat(
                    batch_size, 1, 1, 1
                )
            else:
                replicated_full_seg_map = dataset_eval_input.repeat(batch_size, 1, 1)

            for j in range(n_reps):

                if j > 0:
                    binarized_predictions = torch.concat(
                        binarized_predictions_list, dim=0
                    ).double()  # Size [batch_size*n_reps, H, W]
                else:
                    binarized_predictions = None

                latents = torch.randn(
                    (batch_size, 1, *self.hparams.image_shape), device=self.device
                )
                sampling_results = self.sampling_strategy.sample(
                    self.diffusion_model,
                    latents,
                    replicated_cond,
                    binarized_predictions,
                    self.hparams,
                )

                additional_metrics = self.sampling_strategy.additional_metrics(
                    sampling_results, batch_dict, i
                )
                additional_metrics_agg.append(additional_metrics)

                x_pred = from_pm1_to_unit_interval(
                    sampling_results["x_next"]
                )  # Size B x 1 x H x W
                binarized_predictions = (x_pred > 0.5).int().squeeze(1)
                binarized_predictions_list.append(binarized_predictions)

                if self.hparams.write_samples_dir:
                    # Add run id to be able to generate with multiple jobs in parallel without overwriting
                    fname = f"{batch_idx:05d}_{i:04d}_rep{j:04d}_{wandb.run.id}.npz"
                    np.savez_compressed(
                        os.path.join(self.hparams.write_samples_dir, fname),
                        binarized_predictions.cpu().numpy(),
                    )

                if quality_inspection_x_preds is None:
                    quality_inspection_x_preds = x_pred
                    quality_inspection_full_seg_map = batch_dict["target_summary"][
                        i
                    ].repeat(batch_size, 1, 1, 1)

                if self.hparams.write_full_val_samples:
                    all_targets = batch_dict["all_targets"][i].unsqueeze(0)
                    ious = binarized_predictions.unsqueeze(1) * all_targets
                    
                    ious = ious.sum(dim=(2, 3)) / all_targets.sum(dim=(2, 3))
                    chosen_modes = ious.argmax(dim=1)

                    data_to_save = sampling_results[
                        self.hparams.write_full_val_samples
                    ][-1]
                    torch.save(
                        data_to_save,
                        os.path.join(
                            wandb.run.dir,
                            f"val_preds_batch_{batch_idx}_img_{i}_rep_{j}.pt",
                        ),
                    )
                    torch.save(
                        chosen_modes,
                        os.path.join(
                            wandb.run.dir,
                            f"val_predicted_modes_batch_{batch_idx}_img_{i}_rep_{j}.pt",
                        ),
                    )

                if (
                    batch_idx == 0 and i < 4 and j == 0
                ):  # and not self.hparams.prune_clusters:
                    # if prune_clusters, the size of tensors varies between subsequent steps, meaning that we can't just concat them.
                    self.log_sampling_data(
                        sampling_results, batch_dict["logging_target"][i], batch_size, i
                    )

            # Compute metrics individually for each repetition, to determine how many repetitions are useful.
            df_dict = {}
            binarized_predictions = None
            for pred_idx in range(n_reps):
                binarized_predictions = torch.concat(
                    binarized_predictions_list[: (pred_idx + 1)], dim=0
                )  # Size [batch_size*n_reps, H, W]

                # Compute metrics for each binarized prediction
                ged = (
                    compute_ged(possible_modes, binarized_predictions)["ged"]
                    .cpu()
                    .item()
                )
                hm_iou = (
                    compute_hm_iou(possible_modes, binarized_predictions).cpu().item()
                )
                deduped_iou, dedupued_individual_iou_vals = compute_hm_iou(
                    possible_modes.unique(dim=0),
                    binarized_predictions,
                    return_ious=True,
                )
                deduped_iou = deduped_iou.cpu().item()

                df_dict.update(
                    {
                        f"GED_rep_{pred_idx}": ged,
                        f"HM IoU_rep_{pred_idx}": hm_iou,
                        f"HM IoU (deduped)_rep_{pred_idx}": deduped_iou,
                    }
                )
                df_dict.update(
                    {
                        f"HM IoU class {k}_rep_{pred_idx}": dedupued_individual_iou_vals[
                            k
                        ].item()
                        for k in range(len(dedupued_individual_iou_vals))
                    }
                )

                # Dataset-specific metrics related to mode distribution; we only care about how the number of distinct modes develops with the number of repetitions,
                # and ignore the other metrics here.
                dataset_specific_metrics_dict = {}
                if self.dataset_evaluator is not None:
                    if len(dataset_eval_input.shape) == 3:
                        replicated_full_seg_map = dataset_eval_input[None].repeat(
                            binarized_predictions.shape[0], 1, 1, 1
                        )
                    else:
                        replicated_full_seg_map = dataset_eval_input.repeat(
                            binarized_predictions.shape[0], 1, 1
                        )
                    metrics = self.dataset_evaluator.compute_metrics(
                        binarized_predictions.unsqueeze(1), replicated_full_seg_map
                    )
                    dataset_specific_metrics_dict = (
                        self.compute_dataset_specific_metrics([metrics], batch_dict, i)
                    )

                    # Not true for LIDC
                    if "Distinct modes" in dataset_specific_metrics_dict:
                        df_dict.update(
                            {
                                f"Distinct modes_rep_{pred_idx}": dataset_specific_metrics_dict[
                                    "Distinct modes"
                                ]
                            }
                        )

                # Consistent name for the metrics on the full batch, independent of number of repetitions.
                if pred_idx == n_reps - 1:
                    df_dict.update(
                        {
                            "GED": ged,
                            "HM IoU": hm_iou,
                            "HM IoU (deduped)": deduped_iou,
                        }
                    )
                    # Metrics related to the distribution of modes. Here, we also log the metrics that only really make sense with large batches,
                    # since the computations here are based on all generated samples.
                    df_dict.update(dataset_specific_metrics_dict)

            additional_metrics_agg_dict = (
                self.sampling_strategy.aggregate_additional_metrics(
                    additional_metrics_agg
                )
            )
            df_dict.update(additional_metrics_agg_dict)

            self.log_dict({f"{prefix}{k}": v for k, v in df_dict.items()})

            distribution_eval_df = pd.DataFrame(df_dict, index=[i])
            dataframe_agg.append(distribution_eval_df)

        distribution_eval_df = pd.concat(dataframe_agg, ignore_index=True)
        self.log_dict(
            {
                f"{prefix}{k}": v
                for k, v in distribution_eval_df.mean().to_dict().items()
            }
        )
        return {
            "distribution_eval_df": distribution_eval_df,
            "quality_inspection_x_preds": quality_inspection_x_preds,
            "quality_inspection_seg_map": quality_inspection_full_seg_map,
        }

    def log_sampling_data(self, sampling_results, logging_target, batch_size, i):
        # Ensure that all steps have the same shape. This isn't true when pruning via clustering,
        # where the second step is much smaller than the first one, making logging less straight-forward.
        for i in range(1, len(sampling_results["x0_preds"])):
            if (
                sampling_results["x0_preds"][i - 1].shape
                != sampling_results["x0_preds"][i].shape
            ):
                return None

        x0_preds = [
            from_pm1_to_unit_interval(step) for step in sampling_results["x0_preds"]
        ]
        x_steps = [to_unit_interval(step) for step in sampling_results["x_steps"]]
        pg_grads = [to_unit_interval(step) for step in sampling_results["pg_grads"]]

        append_img = to_unit_interval(logging_target.repeat(batch_size, 1, 1))
        x0_preds_grid = self.visualize_batch_sampling_progression(
            x0_preds, append_img=append_img
        )
        x_steps_grid = self.visualize_batch_sampling_progression(
            x_steps, append_img=append_img
        )

        wandb.log(
            {
                "Sampling progression: x0 predictions": wandb.Image(
                    x0_preds_grid.unsqueeze(0)
                ),
                "Sampling progression: x_steps": wandb.Image(x_steps_grid.unsqueeze(0)),
            }
        )

        if len(pg_grads) > 0:
            pg_grads_grid = self.visualize_batch_sampling_progression(pg_grads)
            wandb.log(
                {
                    "Sampling progression: PG gradients": wandb.Image(
                        pg_grads_grid.unsqueeze(0)
                    )
                }
            )

        # Also works without i==0, but then I don't know what I'm actually seeing in the GUI
        if i == 0:
            for step_id, step in enumerate(sampling_results["pg_grads"]):
                log_dict = {
                    "pg_grads_mean": step.mean(),
                    "pg_grads_max": step.max(),
                    "pg_grads_min": step.min(),
                    "pg_grads_median": step.median(),
                }
                wandb.log(log_dict)
                print(f"PG gradient stats in step {i}: {log_dict}")

    def visualize_batch_sampling_progression(self, tensor_list, append_img=None):

        N = len(tensor_list)  # Number of tensors in the list
        B, C, H, W = tensor_list[0].size()  # Assuming all tensors have the same size

        # Stack tensors along a new dimension (batch axis) to make transposing easier
        stacked_images = torch.stack(tensor_list)  # Shape: [N, B, C, H, W]

        # Transpose to swap the first two dimensions (N and B -> B and N)
        transposed_images = stacked_images.permute(
            1, 0, 2, 3, 4
        )  # Shape: [B, N, C, H, W]

        if append_img is not None:
            # Append the image to the list
            append_img = append_img[:, None, None, :, :]  # Shape: [B, 1, 1, H, W]
            transposed_images = torch.cat([transposed_images, append_img], dim=1)
            N += 1

        # Reshape back to a large batch for make_grid, i.e., [B * N, C, H, W]
        all_images = transposed_images.reshape(B * N, C, H, W)

        # Create a grid of BxN images (B rows, N images per row)
        grid = vutils.make_grid(
            all_images, nrow=N, padding=2, normalize=True, pad_value=0.25
        )[0]

        return grid

    @abstractmethod
    def compute_dataset_specific_metrics(self, metrics_agg, batch_dict, i):
        pass

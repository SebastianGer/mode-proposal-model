import copy
import os
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from pytorch_lightning.utilities import rank_zero_warn
from edm.networks import Conv2d
from monai.losses import DiceFocalLoss, DiceLoss
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from torchmetrics.regression import KLDivergence
from torchmetrics.utilities import dim_zero_cat

from firespreadscenarios.utils.hm_iou import compute_hm_iou_batch, compute_iou_matrix
from firespreadscenarios.utils.utils import download_if_path_is_wandb_artifact

from .velocity_decomposition import (
    estimate_cityscapes_priors,
    estimate_mode_probabilities_from_velocity,
)


class MedianMetric(torchmetrics.Metric):
    """Custom metric to compute median of accumulated values."""

    def __init__(self):
        super().__init__()
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, value: torch.Tensor) -> None:
        """Update the metric with a new value."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.values.append(value.detach())

    def compute(self) -> torch.Tensor:
        """Compute the median of accumulated values."""
        if not self.values:
            return torch.tensor(0.0)
        values_tensor = dim_zero_cat(self.values)
        return torch.median(values_tensor)


class ReliabilityDiagramAccumulator(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("probs", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        self.probs.append(probs.detach().float())
        self.targets.append(targets.detach().float())

    def compute(self):
        if not self.probs:
            # No updates happened, e.g. when mode probability estimation is disabled.
            return {}
        all_probs = dim_zero_cat(self.probs)
        all_targets = dim_zero_cat(self.targets)
        unique_tgts = all_targets.unique().sort().values
        groups = {}
        for tgt_val in unique_tgts:
            mask = all_targets == tgt_val
            groups[tgt_val.item()] = all_probs[mask].cpu()
        return groups


class GlobalBrierSkillScore(torchmetrics.Metric):
    # Keeps track of all squared errors across the entire validation/test set,
    # and computes the Brier Skill Score against a reference of a uniform distribution.
    # Importantly, this implicitly handles the case that different batch items have different numbers of modes.
    def __init__(self):
        super().__init__()
        self.add_state("sum_bs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_bs_ref", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds: [K], target: [K] (one-hot or soft)
        bs = torch.sum((preds - target) ** 2)

        # Reference: Uniform distribution
        k = preds.shape[0]
        ref_preds = torch.full_like(preds, 1.0 / k)
        bs_ref = torch.sum((ref_preds - target) ** 2)

        self.sum_bs += bs
        self.sum_bs_ref += bs_ref

    def compute(self):
        return 1.0 - (self.sum_bs / (self.sum_bs_ref + 1e-8))


class BaseModePredictionNetwork(pl.LightningModule, ABC):
    """
    Predicts a fixed set of mode proposals and their selection scores in one forward pass.
    The network is initialized from a trained diffusion model, which can also be used to
    estimate mode probabilities via velocity decomposition.
    """

    diff_model_class = None  # define in subclasses
    gt_prior_distribution = None  # define in subclasses

    def __init__(
        self,
        image_shape: Tuple[int, int] = (64, 128),
        ignore_index: Optional[int] = None,  # Not used by this class
        num_proposals: int = 8,  # Number of modes to sample from the proposal network
        ckpt_path: str = "",  # path to diffusion model checkpoint
        prob_loss_weight: float = 1e-4,  # weight of BCE loss; image-based MSE loss is treated as 1.0
        use_all_targets: bool = True,  # Whether to use all targets for training, or only one random target
        write_val_outputs: bool = False,
        loss_func_name: Literal["MSE", "Dice", "DiceFocal"] = "MSE",
        use_pu_loss: bool = False,  # Whether to use positive-unlabeled loss for the probability predictions
        pi_pos: Optional[
            float
        ] = None,  # Prior for positive samples in PU loss; if None, defaults to 1/2
        lr: float = 1e-4,
        combo_loss_second_weight: float = 1.0,  # Only relevant if loss_func_name is DiceFocal. Weight of the focal part of the loss (lambda_focal).
        estimate_mode_probabilities: Literal[
            "", "NNLS", "NNLS_FAST", "NNLS_SUM_TO_ONE", "SMOOTH_L1_SUM_TO_ONE"
        ] = "",  # Whether and how to estimate mode probabilities from velocity field during validation/test
        estimate_mode_probabilities_n_samples: int = 64,  # Number of random samples to use for estimating mode probabilities from velocity field
        estimate_mode_probabilities_arg: Any = None,  # Additional argument for mode probability estimation, if needed
        estimate_mode_probabilities_n_subsample: int = -1,  # Number of subsamples to use when estimating mode probabilities with SMOOTH_L1_SUM_TO_ONE
        freeze_layers: Literal[
            "encoder", "all_but_last_conv", "none", "retrain_from_scratch"
        ] = "encoder",  # Which layers to unfreeze in the proposal network; retrain_from_scratch resets the entire pretrained network
        estimate_mode_probabilities_hm_only: bool = True,  # If True, evaluate only on Hungarian-matched proposals
        estimate_mode_probabilities_iou_threshold: float = 1.0,  # IoU threshold to filter duplicate proposals. Not only applied when estimating mode probabilities.
        estimate_mode_probabilities_filter_selection_score: bool = False,  # Whether to filter proposals based on selection scores. Is applied before duplicate filtering. Not only applied when estimating mode probabilities.
        estimate_mode_probabilities_from_gt: bool = False,  # Whether to estimate mode probabilities from GT masks instead of proposals, as a baseline
        estimate_cityscapes_priors: bool = False,  # Whether to use the Cityscapes-specific prior estimation procedure, which leverages the known binary structure of the modes. Only relevant if using the Cityscapes dataset.
        train_with_unique_gts: bool = True,  # Whether to train with all GT modes, or only the unique GT modes (i.e. merging duplicates).
    ):
        super().__init__()

        self.save_hyperparameters()

        self.diff_model = self.load_diff_model()
        self.model, self.new_conv, self.n_features = self.build_proposal_network(
            self.diff_model
        )

        if not self.hparams.estimate_mode_probabilities:
            # No need to keep the diffusion model in memory if we're not using it to estimate probabilities.
            del self.diff_model
        else:
            self.diff_model = self.diff_model.diffusion_model
        self.probability_network = self.build_probability_network()

        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = self.get_loss_function()

        # Proposal network needs latent input, since it still has the diffusion model architecture.
        # We keep the latent constant, so there's no uncontrolled noise influence during training.
        # Without this, training oscillates between different proposals.
        # Batch size=1, so latents are duplicated for all samples via broadcasting.
        const_latents = (
            torch.randn((1, 1, image_shape[0], image_shape[1])).to(torch.float64) * 80.0
        )
        self.register_buffer("const_latents", const_latents)

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "IoU": torchmetrics.classification.BinaryJaccardIndex(
                    zero_division=1.0
                ),
                "F1": torchmetrics.classification.BinaryF1Score(threshold=0.5),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        # Separate metrics: regression-style confidence calibration (MSE) and
        # classification metrics (F1, threshold 0.5). These require different
        # target dtypes, so they must be kept separate.
        self.train_prob_network_confidence_mse = torchmetrics.MeanSquaredError()
        self.val_prob_network_confidence_mse = (
            self.train_prob_network_confidence_mse.clone()
        )
        self.test_prob_network_confidence_mse = (
            self.train_prob_network_confidence_mse.clone()
        )

        self.train_prob_metrics = torchmetrics.MetricCollection(
            {
                "F1": BinaryF1Score(threshold=0.5),
            },
            prefix="train_prob_",
        )
        self.val_prob_metrics = self.train_prob_metrics.clone(prefix="val_prob_")
        self.test_prob_metrics = self.train_prob_metrics.clone(prefix="test_prob_")

        # Median brier skill score tracking
        self.val_median_brier_skill_score = MedianMetric()
        self.test_median_brier_skill_score = MedianMetric()

        # Global brier skill score tracking (accumulates across entire validation/test set)
        self.val_global_brier_skill_score = GlobalBrierSkillScore()
        self.test_global_brier_skill_score = GlobalBrierSkillScore()

        # KL divergence tracking for estimated mode probabilities
        self.val_kld = KLDivergence(log_prob=False)
        self.test_kld = KLDivergence(log_prob=False)

        # L1 distance tracking for estimated mode probabilities
        self.val_mode_prob_l1 = torchmetrics.MeanAbsoluteError()
        self.test_mode_prob_l1 = torchmetrics.MeanAbsoluteError()


        self.val_reliability_diagram = ReliabilityDiagramAccumulator()

    def load_state_dict(self, state_dict, strict: bool = True):
        # Handle case where estimate_mode_probabilities is True but checkpoint was saved with False
        # In that case, state_dict won't have diff_model keys. These will instead be loaded in __init__
        # from self.hparams.ckpt_path, which is a different one than the ckpt_path passed to the
        # lightning trainer.
        if self.hparams.estimate_mode_probabilities and not any(
            k.startswith("diff_model.") for k in state_dict.keys()
        ):
            return super().load_state_dict(state_dict, strict=False)
        else:
            return super().load_state_dict(state_dict, strict=strict)

    # Hyperparameters that are fixed at training time: loss_func_name and
    # combo_loss_second_weight define the loss, and ckpt_path is
    # the diffusion checkpoint that the networks are built from.
    TRAIN_TIME_HPARAMS = ["loss_func_name", "combo_loss_second_weight", "ckpt_path"]

    def on_load_checkpoint(self, checkpoint):
        # trainer.validate/test(ckpt_path=...) builds the model from the current config
        # and only restores the weights. Use the training-time hyperparameters from the
        # checkpoint, so that evaluation matches training.
        ckpt_hparams = checkpoint.get("hyper_parameters") or {}
        mismatched = {
            k: (self.hparams[k], ckpt_hparams[k])
            for k in self.TRAIN_TIME_HPARAMS
            if k in ckpt_hparams and k in self.hparams and self.hparams[k] != ckpt_hparams[k]
        }
        if not mismatched:
            return

        if self._trainer is not None and self.trainer.state.fn != "fit":
            for k, (config_value, ckpt_value) in mismatched.items():
                rank_zero_warn(
                    f"Config sets {k}={config_value!r}, but the checkpoint was trained with "
                    f"{k}={ckpt_value!r}. Using the checkpoint's value for evaluation."
                )
                self.hparams[k] = ckpt_value
            self.loss_fn = self.get_loss_function()
            if "ckpt_path" in mismatched:
                # Rebuild the networks from the training-time diffusion checkpoint: the
                # architecture depends on its model_channels, and with
                # estimate_mode_probabilities the diffusion model is used at eval time.
                diff_model = self.load_diff_model()
                self.model, self.new_conv, self.n_features = (
                    self.build_proposal_network(diff_model)
                )
                if self.hparams.estimate_mode_probabilities:
                    self.diff_model = diff_model.diffusion_model
                self.probability_network = self.build_probability_network()
        else:
            # When (re)training, respect the config, but a mismatch is almost certainly
            # unintended, so complain loudly.
            rank_zero_warn(
                f"Training-time hyperparameters differ from the loaded checkpoint "
                f"(config vs. checkpoint): {mismatched}. Keeping the config's values "
                f"since a training run is in progress."
            )

    def get_loss_function(self):
        if self.hparams.loss_func_name == "MSE":
            loss_fn = torch.nn.MSELoss()
        elif self.hparams.loss_func_name == "Dice":
            loss_fn = DiceLoss(sigmoid=True)
        elif self.hparams.loss_func_name == "DiceFocal":
            loss_fn = DiceFocalLoss(
                sigmoid=True, lambda_focal=self.hparams.combo_loss_second_weight
            )
        else:
            raise ValueError(
                f"Unknown loss function name: {self.hparams.loss_func_name}"
            )
        return loss_fn

    def proposal_decision_threshold(self):
        return 0.5

    def load_diff_model(self):
        ckpt_path = download_if_path_is_wandb_artifact(
            self.hparams.ckpt_path, inside_a_run=False
        )
        diff_model = self.diff_model_class.load_from_checkpoint(ckpt_path)
        return diff_model

    def build_proposal_network(self, diff_model):
        n_features = diff_model.hparams.model_channels * 2

        if self.hparams.freeze_layers == "retrain_from_scratch":
            # Rebuild the network with freshly initialized weights instead of copying the
            # pretrained ones. The edm modules create their parameters directly in __init__
            # and have no reset_parameters(), so resetting via model.apply() would silently
            # keep the pretrained weights. Rebuilding from the checkpoint's hyperparameters
            # keeps the architecture identical, and also makes new_conv below start from
            # freshly initialized weights rather than the pretrained final conv.
            model = diff_model.build_diffusion_model()
        else:
            model = copy.deepcopy(diff_model.diffusion_model)

        # Duplicate the 1-channel output to N-channel output
        assert len(self.hparams.image_shape) == 2
        image_shape_min = min(self.hparams.image_shape)
        final_conv_name = f"{image_shape_min}x{image_shape_min}_aux_conv"
        old_conv = model.model.dec[final_conv_name]
        old_shape = model.model.dec[final_conv_name].weight.shape
        new_conv = Conv2d(
            in_channels=old_shape[1], out_channels=self.hparams.num_proposals, kernel=3
        )
        new_weight = old_conv.weight.repeat(self.hparams.num_proposals, 1, 1, 1)
        # Add noise in [-2.5%,+2.5%] to the repeated weights to break symmetry in proposal outputs, so Hungarian matching makes sense.
        new_weight = new_weight * (
            torch.rand(new_weight.shape, device=old_conv.weight.device) * 0.05 + 0.975
        )
        new_conv.weight = torch.nn.Parameter(new_weight)
        new_bias = old_conv.bias.repeat(self.hparams.num_proposals)
        new_bias = new_bias * (
            torch.rand(new_bias.shape, device=old_conv.weight.device) * 0.05 + 0.975
        )
        new_conv.bias = torch.nn.Parameter(new_bias)

        # We want access to the last feature layer to compute probabilities from it, thus we need to set the last conv as the identity.
        model.model.dec[final_conv_name] = torch.nn.Identity()

        # Potentially freeze parts of the model. The trainable head is self.new_conv; the
        # model's own final conv was replaced with an Identity above, so it holds no
        # parameters that would need to be exempted here.
        if self.hparams.freeze_layers not in ["none", "retrain_from_scratch"]:
            for name, param in model.named_parameters():
                # Freeze encoder, but not decoder
                if self.hparams.freeze_layers == "encoder" and "dec" in name:
                    continue

                param.requires_grad = False

        return model, new_conv, n_features

    def build_probability_network(self):
        # Builds a CNN that takes in the feature maps from the diffusion model and outputs proposal probabilities.
        # Source: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

        image_shape = self.hparams.image_shape
        image_shape_min = min(image_shape)
        image_shape_max = max(image_shape)
        # Cityscapes has 64x128 images, so we end up with 4x8 feature maps, instead of the regular 4x4.
        cityscapes_multiplier = image_shape_max // image_shape_min
        ndf = 64
        probability_layers = [
            # input is ``(n_features) x image_shape_min x image_shape_min``
            nn.Conv2d(self.n_features, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if image_shape_min == 128:
            # Add extra downsampling layer for case of 128x128 inputs
            probability_layers.extend(
                [
                    # state size. ``(ndf*8) x 4 x 4``
                    nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 16),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            final_channels = ndf * 16
        else:
            final_channels = ndf * 8

        probability_layers.extend(
            [
                # state size. ``(final_channels) x 4 x 4``
                # Outputs a probability per proposal for each item in the batch. The input size is 4x4 for square datasets,
                # 4x8 for Cityscapes.
                nn.Conv2d(
                    final_channels,
                    self.hparams.num_proposals,
                    (4, 4 * cityscapes_multiplier),
                    1,
                    0,
                    bias=False,
                ),
            ]
        )

        return nn.Sequential(*probability_layers)

    def compute_macro_iou(self, unique_targets_list, matched_proposals_list):
        # Expects hungarian matching to have been performed already, filtering the gen_imgs to the correct number of modes.

        # all_targets: [B,K,H,W]
        # gen_imgs: [B,K,H,W]

        iou = BinaryJaccardIndex(zero_division=1.0).to(matched_proposals_list[0].device)

        batch_ious = []
        for preds, targets in zip(matched_proposals_list, unique_targets_list):
            per_condition_ious = []
            for k in range(targets.shape[0]):
                per_condition_ious.append(iou(preds[[k]], targets[[k]]).nan_to_num(1.0))
            batch_ious.append(torch.stack(per_condition_ious).mean())

        macro_iou = torch.stack(batch_ious).mean().item()
        return macro_iou

    def compute_macro_iou_duplicate_proposals(
        self, unique_targets_list, matched_proposals_list, unique_groups_list
    ):
        iou = BinaryJaccardIndex(zero_division=1.0).to(matched_proposals_list[0].device)

        batch_ious = []
        for preds, targets, groups in zip(
            matched_proposals_list, unique_targets_list, unique_groups_list
        ):
            per_condition_ious = []
            for k in range(targets.shape[0]):
                # Duplicate IoUs for all proposals that belong to the same unique target group, so that they are weighted equally in the final average.
                group = groups[k]
                num_proposals_in_group = len(group["original_indices"])
                for _ in range(num_proposals_in_group):
                    per_condition_ious.append(
                        iou(preds[[k]], targets[[k]]).nan_to_num(1.0)
                    )
            batch_ious.append(torch.stack(per_condition_ious).mean())

        macro_iou = torch.stack(batch_ious).mean().item()
        return macro_iou

    def compute_max_dice(self, unique_targets_list, proposals_list):
        """
        For each ground-truth mode, pick the proposal with the highest Dice score.
        Does not perform Hungarian matching - a single proposal may be re-used
        for multiple targets.  Returns the mean over targets and then over the
        batch. If the set of ground truths contains duplicates, this metric will
        accordingly have a higher weight on them.
        """

        batch_dices = []
        for preds, targets in zip(proposals_list, unique_targets_list):
            cost_matrix = self.pairwise_dice_with_empty(preds, targets)
            max_dice = cost_matrix.max(dim=0).values.mean()
            batch_dices.append(max_dice)

        mean_dice = torch.stack(batch_dices).mean().item()
        return mean_dice

    def pairwise_dice_with_empty(self, preds, targets, eps=1e-8):
        """
        Computes pairwise dice scores. Returns 1.0 if both masks are empty.

        Args:
            preds: Tensor of shape (N, H, W) - binary (0 or 1)
            targets: Tensor of shape (M, H, W) - binary (0 or 1)
        """
        # 1. Flatten and cast
        preds = preds.view(preds.shape[0], -1).float()
        targets = targets.view(targets.shape[0], -1).float()

        # 2. Calculate Numerator (2 * Intersection)
        # Shape: (N, M)
        intersection = torch.mm(preds, targets.t())
        numerator = 2.0 * intersection

        # 3. Calculate Denominator (|A| + |B|)
        # Shape: (N, 1) + (1, M) -> (N, M) via broadcasting
        sum_preds = preds.sum(dim=1).view(-1, 1)
        sum_targets = targets.sum(dim=1).view(1, -1)
        denominator = sum_preds + sum_targets

        # 4. Compute Dice
        # We use eps only to prevent actual division by zero errors
        # before we apply our specific logic.
        dice = numerator / (denominator + eps)

        # 5. Handle the "Both Zero" case
        # If the denominator is 0, both masks were empty.
        # We replace those specific indices with 1.0.
        dice = torch.where(
            denominator == 0, torch.tensor(1.0, device=dice.device), dice
        )

        return dice

    def forward(self, conditioning=None, step_type="", *args, **kwargs):
        """
        conditioning: [B, C, H, W], conditioning for the diffusion model.
        """

        t = torch.tensor([80.0] * conditioning.shape[0], device=conditioning.device)
        features = self.model(self.const_latents, t, conditioning)
        x0 = self.new_conv(features)

        probability_logits = self.probability_network(features)

        return x0, probability_logits.squeeze(3).squeeze(2)

    @abstractmethod
    def unbatch(self, batch):
        # Should return a dictionary with:
        # - image: B x C x H x W
        # - target: B x H x W, randomly chosen if multiple exist
        # - target_summary: B x H x W, contains some kind of summary of the target, e.g. the mean over all modes; used for visualization
        # - all_targets: B x N x H x W, all targets
        # Optionally:
        # - logging_target: B x H x W, the target that is visualized in logging, e.g. a filtered class only containing the flipping classes, for Cityscapes
        raise NotImplementedError()

    def batch_to_cond_and_mask(self, batch, move_to_device=False):
        """
        Preprocess batch
        """
        # rgb_image, seg_map_randomly_binarized, seg_map_deterministicly_binarized, full_seg_map
        batch_dict = self.unbatch(batch)

        # Make sure we have a channel dimension, not just B x H x W
        if len(batch_dict["image"].shape) == 3:
            batch_dict["image"] = batch_dict["image"].unsqueeze(1)

        batch_dict["all_targets"] = batch_dict["all_targets"].float()

        return batch_dict

    def compute_hungarian_matching_cost_matrix(
        self,
        unique_targets,
        current_gen,
        Mi,
    ):

        H, W = current_gen.shape[-2:]
        Ki = unique_targets.shape[0]

        # During fit (training and the validation steps that select the checkpoint), the
        # cost is the training loss, so the cost matrix is also the loss. Standalone
        # validation and test use 1 - IoU, so that all models are evaluated in the same way.
        if self.trainer.state.fn == "fit":

            # Compute cost matrix [Ki, Mi]
            if self.hparams.loss_func_name == "MSE":
                targets_flat = unique_targets.view(Ki, -1).float()
                gen_flat = current_gen.view(Mi, -1).float()
                cost_matrix = torch.cdist(targets_flat, gen_flat) ** 2 / (H * W)
            else:
                # Vectorized computation for non-MSE losses

                # Expand targets and preds to [K, N, H, W] then reshape to [K*N, 1, H, W]
                targets_expanded = (
                    unique_targets.unsqueeze(1)
                    .expand(-1, Mi, -1, -1)
                    .reshape(Ki * Mi, 1, H, W)
                )
                preds_expanded = (
                    current_gen.unsqueeze(0)
                    .expand(Ki, -1, -1, -1)
                    .reshape(Ki * Mi, 1, H, W)
                )

                if self.hparams.loss_func_name == "Dice":
                    temp_loss_fn = DiceLoss(sigmoid=True, reduction="none")
                    cost_matrix = temp_loss_fn(preds_expanded, targets_expanded).view(
                        Ki, Mi
                    )
                elif self.hparams.loss_func_name == "DiceFocal":
                    loss_fn = DiceFocalLoss(
                        sigmoid=True,
                        lambda_focal=self.hparams.combo_loss_second_weight,
                        reduction="none",
                    )
                    cost_matrix = (
                        loss_fn(preds_expanded, targets_expanded)
                        .mean(dim=[1, 2, 3])
                        .view(Ki, Mi)
                    )
                else:
                    raise ValueError(
                        f"Unsupported loss function for vectorized computation: {self.hparams.loss_func_name}"
                    )

        else:
            # Use IoU as cost during evaluation, to have a standardized evaluation protocol between differently trained models.
            iou_matrix = compute_iou_matrix(unique_targets, current_gen)
            cost_matrix = 1.0 - iou_matrix

        return cost_matrix

    def compute_hm_mse_unique_targets(self, all_targets, gen_imgs, probability_logits):
        """
        Computes Hungarian matching loss based on pairwise loss between gen_imgs and all_targets.
        The matching is then used to compute the correct detection targets, which indicate the Ki proposals
        that are the closest to each of the Ki unique targets.
        These targets are then used to compute the loss for the predicted detection probabilities.

        The key difficulty in this function is that different batch items can have different numbers of unique targets,
        and thus need to be handled individually and aggregated as lists, instead of tensors.

        Args:
            all_targets: Tensor [B, K, H, W]; values in {0,1}
            gen_imgs: Tensor [B, N, H, W] OR List of B tensors [M_i, H, W]; values in {0,1}
            probability_logits: Tensor [B, N] OR List of B tensors [M_i]
        """
        B = len(gen_imgs)
        H, W = gen_imgs[0].shape[-2:]

        batch_losses, unique_targets_list, matched_proposals_list = [], [], []

        # We'll store results in lists since M_i varies
        prob_targets_list, prior_prob_targets_list, unique_groups_list = [], [], []

        # Phase 1: compute all cost matrices on GPU without any host synchronization.
        cost_matrices, inverse_indices_list, num_unique_list = [], [], []
        for b in range(B):
            current_gen = gen_imgs[b]  # [M_i, H, W]
            Mi = current_gen.shape[0]

            unique_targets, inverse_indices = torch.unique(
                all_targets[b], dim=0, return_inverse=True
            )
            if self.hparams.train_with_unique_gts:
                targets = unique_targets
            else:
                targets = all_targets[b]
            unique_targets_list.append(targets)
            inverse_indices_list.append(inverse_indices)
            num_unique_list.append(unique_targets.shape[0])

            cost_matrices.append(
                self.compute_hungarian_matching_cost_matrix(targets, current_gen, Mi)
            )

        # Phase 2: transfer everything the CPU-side matching needs in one go, so the
        # GPU pipeline is drained once per batch instead of once per batch item.
        costs_cpu = torch.cat([cm.reshape(-1) for cm in cost_matrices]).detach().cpu()
        inverse_indices_cpu = torch.cat(inverse_indices_list).cpu()
        gt_prior_cpu = self.gt_prior_distribution.cpu()

        # Phase 3: Hungarian matching on CPU, loss/target assembly on GPU.
        cost_offset, inv_offset = 0, 0
        for b in range(B):
            current_gen = gen_imgs[b]
            Mi = current_gen.shape[0]
            cost_matrix = cost_matrices[b]
            Ki = cost_matrix.shape[0]

            cm_cpu = costs_cpu[cost_offset : cost_offset + Ki * Mi].view(Ki, Mi)
            cost_offset += Ki * Mi
            row_ind, col_ind = linear_sum_assignment(cm_cpu.numpy())

            matched_proposals_list.append(current_gen[col_ind].detach().clip(0, 1))
            batch_losses.append(cost_matrix[row_ind, col_ind].mean())

            p_target = torch.zeros(Mi, device=current_gen.device)
            p_target[col_ind] = 1
            prob_targets_list.append(p_target)

            # Compute prior probability per mode and mapping from unique to all targets
            n_targets = inverse_indices_list[b].numel()
            item_inverse_cpu = inverse_indices_cpu[inv_offset : inv_offset + n_targets]
            inv_offset += n_targets
            aggregated_probs, unique_groups_dict = self.merge_duplicate_mode_probs(
                num_unique_list[b], item_inverse_cpu, gt_prior_cpu
            )
            unique_groups_list.append(unique_groups_dict)

            prior_p_target = torch.full((Mi,), -1.0)
            if self.hparams.train_with_unique_gts:
                # aggregated_probs has one value per unique GT, and col_ind has one index per
                # matched target. The lengths are equal only if the targets are the unique GTs.
                # Otherwise, all prior targets stay -1 (unknown).
                prior_p_target[col_ind] = aggregated_probs

            prior_prob_targets_list.append(prior_p_target.to(current_gen.device))

        image_loss = torch.stack(batch_losses).mean()

        if isinstance(probability_logits, list):
            all_logits = torch.cat(probability_logits)
        else:
            all_logits = probability_logits.flatten()
        all_p_targets = torch.cat(prob_targets_list)

        if self.hparams.use_pu_loss:
            # Ignore the -100 padding applied when filtering is performed.
            # Should not occur during training, but better to be safe.
            valid_logit_mask = all_logits > -50

            probability_loss = self.compute_pu_loss(
                all_logits[valid_logit_mask], all_p_targets[valid_logit_mask].float()
            )
        else:
            probability_loss = self.bce_loss(all_logits, all_p_targets)
        loss = image_loss + self.hparams.prob_loss_weight * probability_loss

        return (
            loss,
            unique_targets_list,
            matched_proposals_list,
            prob_targets_list,
            prior_prob_targets_list,
            unique_groups_list,
        )

    def compute_pu_loss(self, probability_logits, probability_targets):
        # Computes positive-unlabeled loss for probability predictions.

        if self.hparams.pi_pos is None:
            # Prior for positive samples, we usually propose twice as many samples as we know to be present.
            pi_pos = 1 / 2
        else:
            pi_pos = float(self.hparams.pi_pos)

        pos_mask = probability_targets == 1
        unlabeled_mask = probability_targets == 0

        pos_loss = (
            self.bce_loss(probability_logits[pos_mask], probability_targets[pos_mask])
            if pos_mask.any()
            else torch.tensor(0.0, device=probability_logits.device)
        )
        unlabeled_loss = (
            self.bce_loss(
                probability_logits[unlabeled_mask], probability_targets[unlabeled_mask]
            )
            if unlabeled_mask.any()
            else torch.tensor(0.0, device=probability_logits.device)
        )

        pos_inv_loss = (
            self.bce_loss(
                probability_logits[pos_mask], 1 - probability_targets[pos_mask]
            )
            if pos_mask.any()
            else torch.tensor(0.0, device=probability_logits.device)
        )
        probability_loss = pi_pos * pos_loss + max(
            unlabeled_loss - pi_pos * pos_inv_loss, 0
        )

        return probability_loss

    def merge_duplicate_mode_probs(self, num_unique, inverse_indices_cpu, gt_prior_cpu):
        """
        Merges probabilities of duplicate GT masks.

        Operates on CPU tensors so it doesn't force GPU synchronizations; the caller
        transfers the inverse indices and prior distribution once per batch.

        The initial draft of this method was written with Gemini.
        """

        # Sum probabilities for each unique mask group
        # We use scatter_add_ to aggregate probs based on the inverse mapping
        aggregated_probs = torch.zeros(num_unique).scatter_add_(
            0, inverse_indices_cpu, gt_prior_cpu
        )

        # Map which original indices belong to which unique group
        inverse_list = inverse_indices_cpu.tolist()
        groups = {}
        for i in range(num_unique):
            # Find original indices that were mapped to this unique mask
            original_ids = [j for j, inv in enumerate(inverse_list) if inv == i]
            groups[i] = {
                "original_indices": original_ids,
                "merged_probability": aggregated_probs[i].item(),
            }

        return aggregated_probs, groups

    def filter_duplicate_proposals(self, proposals):
        """
        Remove near-duplicate proposals based on pairwise IoU.

        Args:
            proposals: Tensor of shape [K, H, W] or [K, 1, H, W]
            iou_threshold: float, IoU above which a proposal is considered duplicate

        Returns:
            filtered_proposals: Tensor of kept proposals [M, H, W]
            kept_indices: LongTensor of kept original indices
        """
        iou_threshold = self.hparams.estimate_mode_probabilities_iou_threshold
        if iou_threshold >= 1.0:
            # No filtering needed, return all proposals and their indices
            K = proposals.shape[0]
            return proposals, torch.arange(K, device=proposals.device)

        if proposals.dim() == 4:
            proposals = proposals.squeeze(1)

        K = proposals.shape[0]
        if K <= 1:
            return proposals, torch.arange(K, device=proposals.device)

        # Binarize proposals (safe if already 0/1)
        bins = (proposals >= 0.5).float()

        # Compute pairwise IoU matrix
        iou = compute_iou_matrix(bins, bins)

        removed = torch.zeros(K, dtype=torch.bool, device=proposals.device)
        keep = []
        for i in range(K):
            if removed[i]:
                continue
            keep.append(i)
            # mark later indices with high IoU as removed
            j_idx = (iou[i] > iou_threshold).nonzero(as_tuple=True)[0]
            for j in j_idx.tolist():
                if j > i:
                    removed[j] = True

        kept_indices = torch.tensor(keep, device=proposals.device, dtype=torch.long)
        filtered = proposals[kept_indices]
        return filtered, kept_indices

    def _estimate_and_log_priors_for_proposals(
        self,
        proposals_01,
        prior_targets,
        cond,
        step_type: Literal["val", "test"],
        allow_reliability_diagram: bool = False,
    ):
        """
        Helper that estimates prior mode probabilities for a given set of proposals and logs metrics.

        Expectations:
        - `proposals_01` must be in the [0, 1] range and have shape [M, H, W] or [M, 1, H, W].
        - `prior_targets` is a 1-D tensor of length M with values in [0,1] or -1 for unknown.

        The helper converts `proposals_01` to the [-1, 1] range before calling
        `estimate_mode_probabilities_from_velocity` (which expects values roughly in [-1,1]).
        """

        # Ensure shape [M, H, W]
        if proposals_01.dim() == 4:
            proposals_01 = proposals_01.squeeze(1)

        M = proposals_01.shape[0]
        if M <= 1:
            return

        # Convert to [-1,1] for estimator
        proposals_minus1_1 = 2 * proposals_01 - 1

        # Mask out proposals without ground-truth prior
        valid_mask = prior_targets != -1
        if valid_mask.sum() <= 1:
            return

        if self.hparams.estimate_cityscapes_priors:
            estimated_prior_probs = estimate_cityscapes_priors(
                proposals_01,
                self.diff_model,
                cond,
                self.hparams.estimate_mode_probabilities_n_samples,
                self.hparams.estimate_mode_probabilities,
                self.hparams.estimate_mode_probabilities_arg,
                self.hparams.estimate_mode_probabilities_n_subsample,
                device=self.device,
            )
        else:
            estimated_prior_probs = estimate_mode_probabilities_from_velocity(
                self.diff_model,
                proposals_minus1_1[valid_mask],
                cond,
                self.hparams.estimate_mode_probabilities_n_samples,
                self.hparams.estimate_mode_probabilities,
                self.hparams.estimate_mode_probabilities_arg,
                self.hparams.estimate_mode_probabilities_n_subsample,
                device=self.device,
            )

        est = estimated_prior_probs
        tgt = prior_targets[valid_mask]

        prior_estimation_mse = torch.nn.functional.mse_loss(est, tgt)

        uniform_dist = torch.full_like(est, 1.0 / valid_mask.sum().item())
        reference_mse = torch.nn.functional.mse_loss(uniform_dist, tgt)
        brier_skill_score = 1 - (prior_estimation_mse / (reference_mse + 1e-8))

        # Compute KL divergence
        eps = 1e-8
        est_safe = torch.clamp(est, min=eps)
        tgt_safe = torch.clamp(tgt, min=eps)

        if step_type == "val":
            if allow_reliability_diagram:
                self.val_reliability_diagram.update(est, tgt)
            self.val_median_brier_skill_score.update(brier_skill_score)
            self.val_global_brier_skill_score.update(est, tgt)
            self.val_mode_prob_l1.update(est, tgt)
            self.val_kld.update(est_safe[None], tgt_safe[None])
            self.log_dict(
                {
                    "val_median_brier_skill_score": self.val_median_brier_skill_score,
                    "val_global_brier_skill_score": self.val_global_brier_skill_score,
                    "val_mode_prob_l1": self.val_mode_prob_l1,
                    "val_global_kld": self.val_kld,
                },
                on_step=False,
                on_epoch=True,
            )
        elif step_type == "test":
            self.test_median_brier_skill_score.update(brier_skill_score)
            self.test_global_brier_skill_score.update(est, tgt)
            self.test_mode_prob_l1.update(est, tgt)
            self.test_kld.update(est_safe[None], tgt_safe[None])
            self.log_dict(
                {
                    "test_median_brier_skill_score": self.test_median_brier_skill_score,
                    "test_global_brier_skill_score": self.test_global_brier_skill_score,
                    "test_mode_prob_l1": self.test_mode_prob_l1,
                    "test_global_kld": self.test_kld,
                },
                on_step=False,
                on_epoch=True,
            )

        self.log_dict(
            {
                f"{step_type}_brier_score_mean": prior_estimation_mse,
                f"{step_type}_brier_skill_score_mean": brier_skill_score,
            },
            on_step=False,
            on_epoch=True,
        )

    def cityscapes_gt_class_distribution(self, targets):
        # Indices 8, 4, 2, 1 are the modes that contain only one of the four flipping
        # classes (see estimate_cityscapes_priors). Classes that are not in the image get
        # probability 0, the same as in the estimate.
        valid_classes_mask = targets[[8, 4, 2, 1]].bool().any(dim=(1, 2))
        gt_distribution = self.gt_class_distribution.clone()
        gt_distribution[~valid_classes_mask] = 0.0
        return gt_distribution

    def prepare_filtered_proposals_list(self, x0, logits, all_targets):
        """
        Filters duplicates per batch item and ensures M_i >= K_i.
        Returns lists of tensors to avoid batch-wide superfluous padding.
        """
        B, _, H, W = x0.shape
        filtered_x0_list = []
        filtered_logits_list = []

        for b in range(B):
            # 1. Identify how many unique ground truths we need to cover
            unique_gt = torch.unique(all_targets[b], dim=0)
            Ki = unique_gt.shape[0]
            current_logits = logits[b]

            # Keep proposals with selection score >= 0.5.
            if self.hparams.estimate_mode_probabilities_filter_selection_score:
                keep_mask = current_logits.sigmoid() >= 0.5
            else:
                keep_mask = torch.ones_like(current_logits, dtype=torch.bool)

            # Apply the mask to both x0 and logits
            pre_filtered_x = x0[b][keep_mask]
            pre_filtered_l = current_logits[keep_mask]

            # 2. Filter duplicates for this specific item (Pass the pre-filtered versions)
            filtered_x, kept_idx = self.filter_duplicate_proposals(pre_filtered_x)
            filtered_l = pre_filtered_l[kept_idx]

            # 3. Ensure Mi >= Ki by padding with empty masks ONLY if necessary
            Mi = filtered_x.shape[0]
            if Mi < Ki:
                pad_size = Ki - Mi
                pad_x = torch.zeros((pad_size, H, W), device=x0.device, dtype=x0.dtype)
                # -100 ensures these 'empty' proposals have ~0 probability after sigmoid
                pad_l = torch.full(
                    (pad_size,), -100.0, device=logits.device, dtype=logits.dtype
                )

                filtered_x = torch.cat([filtered_x, pad_x], dim=0)
                filtered_l = torch.cat([filtered_l, pad_l], dim=0)

            filtered_x0_list.append(filtered_x)
            filtered_logits_list.append(filtered_l)

        return filtered_x0_list, filtered_logits_list

    def training_step(self, batch, batch_idx):
        batch_dict = self.batch_to_cond_and_mask(batch)

        if self.hparams.use_all_targets:
            # [B,K,H,W]
            all_targets = batch_dict["all_targets"]
        else:
            # [B,1,H,W]
            all_targets = batch_dict["target"]  # [B,1,H,W]
            if len(all_targets.shape) == 3:
                all_targets = all_targets.unsqueeze(1)
            assert len(all_targets.shape) == 4

        x0_pm1, probability_logits = self.forward(
            batch_dict["image"], batch_idx=batch_idx, step_type="train"
        )
        x0 = (x0_pm1 + 1) / 2

        (loss, unique_targets_list, matched_proposals_list, prob_targets_list, _, _) = (
            self.compute_hm_mse_unique_targets(all_targets, x0, probability_logits)
        )

        if loss.isnan().any():
            raise RuntimeError("Loss is NaN.")

        # Different batches can have different numbers of unique targets, but the metrics
        # accumulate global counts, so we can concatenate along the mode dimension and
        # update once instead of once per batch item.
        threshold = self.proposal_decision_threshold()
        self.train_metrics.update(
            (torch.cat(matched_proposals_list) > threshold).float(),
            torch.cat(unique_targets_list),
        )

        all_p_targets = torch.cat(prob_targets_list)
        all_logits = probability_logits.flatten().sigmoid()
        self.train_prob_metrics.update(all_logits, all_p_targets.long())
        self.train_prob_network_confidence_mse(all_logits, all_p_targets)

        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_confidence_MSE": self.train_prob_network_confidence_mse,
                **self.train_metrics,
                **self.train_prob_metrics,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Debug logging
        if (
            batch_idx == 0
            and self.trainer.is_global_zero
            and self.trainer.current_epoch % 5 == 0
        ):  # Only log on main process
            all_targets = batch_dict["all_targets"][[0]].mean(1)
            # log x0s
            images_to_log = [wandb.Image(all_targets)] + [
                wandb.Image(x0[0, [i]].clip(0, 1)) for i in range(x0.shape[1])
            ]
            wandb.log({f"Train images": images_to_log})

        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self.batch_to_cond_and_mask(batch)
        all_targets = batch_dict["all_targets"]

        x0_pm1, probability_logits = self.forward(batch_dict["image"])
        # Binarize the proposals (see proposal_decision_threshold).
        x0 = ((x0_pm1 + 1) / 2 >= self.proposal_decision_threshold()).float()

        # Pre-filter: get lists of [Mi, H, W] and [Mi]
        x0_list, logits_list = self.prepare_filtered_proposals_list(
            x0, probability_logits, all_targets
        )

        (
            loss,
            unique_targets_list,
            matched_proposals_list,
            prob_targets_list,
            prior_targets_list,
            unique_groups_list,
        ) = self.compute_hm_mse_unique_targets(all_targets, x0_list, logits_list)

        # Metrics accumulate globally, so update once on concatenated tensors instead of per item.
        self.val_metrics.update(
            torch.cat(matched_proposals_list), torch.cat(unique_targets_list)
        )
        all_probs = torch.cat(logits_list).sigmoid()
        all_prob_targets = torch.cat(prob_targets_list)
        self.val_prob_network_confidence_mse(all_probs, all_prob_targets)
        self.val_prob_metrics.update(all_probs, all_prob_targets.long())

        if self.hparams.estimate_mode_probabilities:
            for b in range(len(x0_list)):
                n_gts = all_targets[b].shape[0]
                n_unique_gts = torch.unique(all_targets[b], dim=0).shape[0]
                all_gts_unique = (n_unique_gts == n_gts)

                if (
                    self.hparams.estimate_mode_probabilities_from_gt
                    or self.hparams.estimate_cityscapes_priors
                ):
                    if self.hparams.estimate_cityscapes_priors:
                        gt_distribution = self.cityscapes_gt_class_distribution(
                            all_targets[b]
                        )
                    else:
                        gt_distribution = self.gt_prior_distribution
                    self._estimate_and_log_priors_for_proposals(
                        all_targets[b],
                        gt_distribution,
                        batch_dict["image"][b],
                        "val",
                        allow_reliability_diagram=all_gts_unique,
                    )
                else:
                    prior_t = prior_targets_list[b].clone()
                    if not self.hparams.estimate_mode_probabilities_hm_only:
                        prior_t[prior_t == -1] = 0.0

                    self._estimate_and_log_priors_for_proposals(
                        x0_list[b], prior_t, batch_dict["image"][b], "val",
                        allow_reliability_diagram=all_gts_unique,
                    )

        val_logs = {
            "val_loss": loss,
            "val_macro_iou_with_duplicate_gts": compute_hm_iou_batch(all_targets, x0),
            "val_max_dice_with_duplicates": self.compute_max_dice(all_targets, x0_list),
            "val_max_dice_unique": self.compute_max_dice(unique_targets_list, x0_list),
            **self.val_metrics,
            **self.val_prob_metrics,
        }
        if self.hparams.train_with_unique_gts:
            # These metrics need the matching to unique GTs.
            val_logs["val_macro_iou_deduplicated"] = self.compute_macro_iou(
                unique_targets_list, matched_proposals_list
            )
            val_logs["val_macro_iou_with_duplicate_proposals"] = (
                self.compute_macro_iou_duplicate_proposals(
                    unique_targets_list, matched_proposals_list, unique_groups_list
                )
            )
        self.log_dict(val_logs, on_epoch=True)

        mean_number_of_selected_proposals = sum(len(l) for l in x0_list) / len(x0_list)
        self.log(
            "val_mean_selected_proposals",
            mean_number_of_selected_proposals,
            on_epoch=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        batch_dict = self.batch_to_cond_and_mask(batch)
        all_targets = batch_dict["all_targets"]

        # Forward pass yields full proposals [B, num_proposals, H, W]
        x0_pm1, probability_logits = self.forward(
            batch_dict["image"], batch_idx=batch_idx, step_type="test"
        )
        # Binarize the proposals (see proposal_decision_threshold).
        x0 = ((x0_pm1 + 1) / 2 >= self.proposal_decision_threshold()).float()

        # Filter duplicates and ensure M_i >= K_i per batch item
        # Returns List[Tensor(Mi, H, W)] and List[Tensor(Mi)]
        x0_list, logits_list = self.prepare_filtered_proposals_list(
            x0, probability_logits, all_targets
        )

        (
            loss,
            unique_targets_list,
            matched_proposals_list,
            prob_targets_list,
            prior_targets_list,
            unique_groups_list,
        ) = self.compute_hm_mse_unique_targets(all_targets, x0_list, logits_list)

        # 1. + 2. Segmentation and probability network metrics: these accumulate
        # globally, so update once on concatenated tensors instead of per item.
        self.test_metrics.update(
            torch.cat(matched_proposals_list), torch.cat(unique_targets_list)
        )
        all_probs = torch.cat(logits_list).sigmoid()
        all_prob_targets = torch.cat(prob_targets_list)
        self.test_prob_network_confidence_mse.update(all_probs, all_prob_targets)
        self.test_prob_metrics.update(all_probs, all_prob_targets.long())

        # 3. Prior Probability Estimator (Diffusion/Flow Perspective)
        if self.hparams.estimate_mode_probabilities:
            for b in range(len(x0_list)):
                if (
                    self.hparams.estimate_mode_probabilities_from_gt
                    or self.hparams.estimate_cityscapes_priors
                ):
                    if self.hparams.estimate_cityscapes_priors:
                        gt_distribution = self.cityscapes_gt_class_distribution(
                            all_targets[b]
                        )
                    else:
                        gt_distribution = self.gt_prior_distribution
                    self._estimate_and_log_priors_for_proposals(
                        all_targets[b],
                        gt_distribution,
                        batch_dict["image"][b],
                        "test",
                    )
                else:
                    prior_t = prior_targets_list[b].clone()
                    if not self.hparams.estimate_mode_probabilities_hm_only:
                        prior_t[prior_t == -1] = 0.0
                    self._estimate_and_log_priors_for_proposals(
                        x0_list[b], prior_t, batch_dict["image"][b], "test"
                    )

        macro_iou = self.compute_macro_iou(unique_targets_list, matched_proposals_list)
        macro_iou_with_duplicate_proposals = self.compute_macro_iou_duplicate_proposals(
            unique_targets_list, matched_proposals_list, unique_groups_list
        )

        macro_iou_with_duplicates = compute_hm_iou_batch(all_targets, x0)
        max_dice_with_duplicates = self.compute_max_dice(all_targets, x0_list)
        max_dice_unique = self.compute_max_dice(unique_targets_list, x0_list)
        self.log_dict(
            {
                "test_loss": loss.item(),
                "test_macro_iou_deduplicated": macro_iou,
                "test_macro_iou_with_duplicate_gts": macro_iou_with_duplicates,
                "test_macro_iou_with_duplicate_proposals": macro_iou_with_duplicate_proposals,
                "test_max_dice_with_duplicates": max_dice_with_duplicates,
                "test_max_dice_unique": max_dice_unique,
                "test_confidence_MSE": self.test_prob_network_confidence_mse,
                **self.test_metrics,
                **self.test_prob_metrics,
            },
            on_step=False,
            on_epoch=True,
        )

        mean_number_of_selected_proposals = sum(len(l) for l in x0_list) / len(x0_list)
        self.log(
            "test_mean_selected_proposals",
            mean_number_of_selected_proposals,
            on_epoch=True,
        )

        return loss

    def on_validation_epoch_end(self):
        groups = self.val_reliability_diagram.compute()
        self.val_reliability_diagram.reset()

        if self.trainer.is_global_zero and groups:
            tgt_vals = sorted(v for v in groups.keys() if v > 0)
            if not tgt_vals:
                return
            positions = list(range(len(tgt_vals)))
            y_lim = (min(tgt_vals) * 0.5, max(tgt_vals) * 2)
            fs = 6
            fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
            for i, tgt_val in enumerate(tgt_vals):
                ests = np.atleast_1d(groups[tgt_val].numpy())
                jitter = np.random.uniform(-0.2, 0.2, size=len(ests))
                ax.scatter(
                    i + jitter,
                    ests,
                    alpha=0.4,
                    s=4,
                    color="steelblue",
                    zorder=2,
                    label="Estimates" if i == 0 else None,
                )
                ax.hlines(
                    ests.mean(),
                    i - 0.2,
                    i + 0.2,
                    colors="red",
                    linewidths=1.5,
                    zorder=3,
                    label="Mean estimate" if i == 0 else None,
                )
                ax.hlines(
                    tgt_val,
                    i - 0.35,
                    i + 0.35,
                    colors="black",
                    linewidths=1.0,
                    linestyles="dashed",
                    zorder=1,
                    label="True probability" if i == 0 else None,
                )
            ax.set_yscale("log")
            ax.set_ylim(y_lim)
            ax.set_xlim(-0.5, len(tgt_vals) - 0.5)
            ax.set_xticks(positions)
            ax.set_xticklabels([f"{v:.4f}" for v in tgt_vals], rotation=90, fontsize=fs)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.4f}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="y", labelsize=fs)
            ax.set_xlabel("True probability", fontsize=fs)
            ax.set_ylabel("Estimated probability", fontsize=fs)
            ax.set_title("Reliability diagram — velocity decomposition (val)", fontsize=fs)
            ax.legend(fontsize=fs)
            pdf_path = os.path.join(self.trainer.log_dir, "val_reliability_diagram.pdf")
            fig.savefig(pdf_path)
            wandb.save(pdf_path)
            plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr
        )
        return optimizer

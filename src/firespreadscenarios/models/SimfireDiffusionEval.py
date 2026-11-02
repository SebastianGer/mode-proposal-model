import pytorch_lightning as pl
import torch

from .BaseDiffusionEval import BaseDiffusionEval
from .BaseModePredictionNetwork import BaseModePredictionNetwork
from .SimfireDiffusionTraining import SimfireMixin


class SimfireDiffusionEval(SimfireMixin, BaseDiffusionEval):

    def compute_dataset_specific_metrics(self, metrics_agg, batch_dict, i):

        N = len(self.dataset_evaluator.class_distribution)
        # turn list of dicts into dict of lists
        metrics_dict = {k: [d[k] for d in metrics_agg] for k in metrics_agg[0].keys()}

        # Compute cluster_distribution
        chosen_modes = torch.concatenate(metrics_dict["chosen_modes"])
        cluster_distribution = (
            torch.bincount(chosen_modes, minlength=N) / chosen_modes.numel()
        )

        uniform_cluster_distribution = self.dataset_evaluator.uniform_distribution
        true_cluster_distribution = self.dataset_evaluator.class_distribution

        tvd_true = (cluster_distribution - true_cluster_distribution).abs().sum().item()
        tvd_uniform = (
            (cluster_distribution - uniform_cluster_distribution).abs().sum().item()
        )

        expected_coupons = self.dataset_evaluator.compute_expected_number_of_coupons(
            cluster_distribution
        )
        n_distinct_modes = (cluster_distribution > 0).sum().cpu().float().item()

        general_metrics_dict = {
            "TVD true": tvd_true,
            "TVD uniform": tvd_uniform,
            "Wrong area IoU": torch.concatenate(metrics_dict["wrong_area_iou"])
            .mean()
            .cpu()
            .item(),
            "Expected Coupons": expected_coupons,
            "Distinct modes": n_distinct_modes,
            "Brier score": torch.stack(metrics_dict["brier_score"]).mean().cpu().item(),
        }

        return general_metrics_dict


class SimfireModePredictionNetwork(SimfireMixin, BaseModePredictionNetwork):
    diff_model_class = SimfireDiffusionEval
    gt_prior_distribution = torch.tensor(
        [2**i / (2**8 - 1) for i in range(8)], device="cuda"
    )

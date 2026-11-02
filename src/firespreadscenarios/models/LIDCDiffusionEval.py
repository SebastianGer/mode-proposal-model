import torch

from .BaseDiffusionEval import BaseDiffusionEval
from .BaseModePredictionNetwork import BaseModePredictionNetwork
from .LIDCDiffusionTraining import LIDCMixin


class LIDCDiffusionEval(LIDCMixin, BaseDiffusionEval):
    def compute_dataset_specific_metrics(self, metrics_agg, batch_dict, i):

        brier_score = (
            torch.stack([m["brier_score"] for m in metrics_agg]).mean().cpu().item()
        )
        return {"Brier score": brier_score}


class LIDCModePredictionNetwork(LIDCMixin, BaseModePredictionNetwork):
    diff_model_class = LIDCDiffusionEval
    gt_prior_distribution = torch.tensor(
        [0.25]*4, device="cuda"
    )
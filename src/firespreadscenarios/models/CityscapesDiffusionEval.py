import torch

from .BaseDiffusionEval import BaseDiffusionEval
from .BaseModePredictionNetwork import BaseModePredictionNetwork
from .CityscapesDiffusionTraining import CityscapesMixin


class CityscapesDiffusionEval(CityscapesMixin, BaseDiffusionEval):

    def compute_dataset_specific_metrics(self, metrics_agg, batch_dict, i):

        # turn list of dicts into dict of lists
        metrics_dict = {k: [d[k] for d in metrics_agg] for k in metrics_agg[0].keys()}

        # Sum across all generated samples to get the distribution of the modes per class
        chosen_modes_all = torch.concatenate(metrics_dict["chosen_modes"], dim=0)
        cluster_distribution = chosen_modes_all.nanmean(dim=0).float().nan_to_num_(0.0)

        # Turn binary mode choices into concatenated binary strings, then decode into mode ids
        def float_tensor_to_mode_id(chosen_modes_i):
            string_list = map(lambda i: str(i.item()), chosen_modes_i.int())
            binary_string = "".join(string_list)
            decoded_decimal = int(binary_string, 2)
            return decoded_decimal

        chosen_modes = torch.tensor(
            [
                float_tensor_to_mode_id(chosen_modes_i)
                for chosen_modes_i in chosen_modes_all.nan_to_num_(0.0)
            ],
            device=self.device,
        )
        n_distinct_modes = float(len(chosen_modes.unique()))

        cluster_ids = batch_dict["target_summary"][i].unique()
        cluster_ids = cluster_ids[cluster_ids != 0]
        uniform_cluster_distribution = (
            self.dataset_evaluator.get_uniform_cluster_distribution(cluster_ids)
        )
        true_cluster_distribution = (
            self.dataset_evaluator.get_true_cluster_distribution(cluster_ids)
        )

        tvd_true = (cluster_distribution - true_cluster_distribution).abs().sum()
        tvd_uniform = (cluster_distribution - uniform_cluster_distribution).abs().sum()

        mode_strictness_dict = {
            f"Mode strictness {self.dataset_evaluator.flipping_classes[i].item()-1}": strictness_float.item()
            for i, strictness_float in enumerate(
                torch.concatenate(metrics_dict["mode_strictness"]).mean(0).cpu().numpy()
            )
        }
        chosen_modes_dict = {
            f"Mode probability {self.dataset_evaluator.flipping_classes[i].item()-1}": cluster_prob.item()
            for i, cluster_prob in enumerate(cluster_distribution.cpu().numpy())
        }

        general_metrics_dict = {
            "L1 true marginal distribution": tvd_true.item(),
            "L1 uniform marginal distribution": tvd_uniform.item(),
            "Mode strictness": torch.concatenate(metrics_dict["mode_strictness"])
            .nanmean()
            .cpu()
            .item(),
            "Wrong area IoU": torch.concatenate(metrics_dict["wrong_area_iou"])
            .mean()
            .cpu()
            .item(),
            "Distinct modes": n_distinct_modes,
        }

        return general_metrics_dict | mode_strictness_dict | chosen_modes_dict


class CityscapesModePredictionNetwork(CityscapesMixin, BaseModePredictionNetwork):
    diff_model_class = CityscapesDiffusionEval
    # Computed via dataloader.CityscapesEval.CityscapesEval.get_probability_per_mode
    # Both probability distributions only valid for four flipping classes.
    gt_prior_distribution = torch.tensor(
        [
            8.9063e-03,
            1.6922e-01,
            2.6719e-02,
            5.0766e-01,
            2.9688e-03,
            5.6406e-02,
            8.9063e-03,
            1.6922e-01,
            4.6875e-04,
            8.9062e-03,
            1.4063e-03,
            2.6719e-02,
            1.5625e-04,
            2.9687e-03,
            4.6875e-04,
            8.9062e-03,
        ],
        device="cuda",
    )
    gt_class_distribution = torch.tensor([0.05, 0.25, 0.75, 0.95], device="cuda")

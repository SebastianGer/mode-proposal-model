import torch

from .CityscapesDiffusionTraining import CityscapesMixin
from .LIDCDiffusionTraining import LIDCMixin
from .ProbabilisticUNet import BaseProbabilisticUNetLightning
from .SimfireDiffusionTraining import SimfireMixin


class SimfirePUNet(SimfireMixin, BaseProbabilisticUNetLightning):

    def compute_dataset_specific_metrics(self, metrics_agg, batch_dict, i):

        N = len(self.dataset_evaluator.class_distribution)
        # turn list of dicts into dict of lists
        metrics_dict = {k: [d[k] for d in metrics_agg] for k in metrics_agg[0].keys()}

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


class LIDCPUNet(LIDCMixin, BaseProbabilisticUNetLightning):

    def compute_dataset_specific_metrics(self, metrics_agg, batch_dict, i):

        brier_score = (
            torch.stack([m["brier_score"] for m in metrics_agg]).mean().cpu().item()
        )
        return {"Brier score": brier_score}


class CityscapesPUNet(CityscapesMixin, BaseProbabilisticUNetLightning):

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

        # Changed from previous experiments: Not actually TVD anymore, since we don't deal with proper distributions.
        # Our distributions variables contain lists of bernoulli distributions
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

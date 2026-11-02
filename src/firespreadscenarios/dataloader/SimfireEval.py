from itertools import combinations
from typing import Literal

import torch


class SimfireEval:

    def __init__(
        self, distribution_mode: Literal["uniform", "asymmetric"], device="cuda"
    ):
        self.device = device

        self.uniform_distribution = (
            torch.ones(8, device=self.device) / 8
        )  # Uniform distribution over 8 classes
        if distribution_mode == "uniform":
            self.class_distribution = self.uniform_distribution
        elif distribution_mode == "asymmetric":
            class_distribution = torch.tensor(
                [2**i for i in range(8)], device=self.device
            )
            self.class_distribution = (
                class_distribution / class_distribution.sum()
            )  # Normalize to sum to 1

    def compute_metrics(self, generated_seg_maps, full_seg_map):
        # full_seg_map is mask_dict["all_targets"], shape B x 8 x 240 x 240
        # generated_seg_maps is the output of the model, shape: B x C x H x W ??

        # Add background class to the three main classes
        all_targets = full_seg_map
        background_class = full_seg_map.sum(dim=1, keepdim=True) == 0
        one_hot_seg_map = torch.cat([background_class, all_targets], dim=1)

        ious = generated_seg_maps * one_hot_seg_map
        ious = ious.sum(dim=(2, 3)) / one_hot_seg_map.sum(dim=(2, 3))

        # Class 0 catches all areas that should never be 1 because they correspond to classes that we don't flip
        wrong_area_iou = ious[:, 0]
        good_areas_ious = ious[:, 1:]

        ### Compute diversity metrics
        # determine chosen mode per class
        chosen_modes = good_areas_ious.argmax(dim=1)

        ### Brier score
        weighted_gt_mask = all_targets * self.class_distribution[None, :, None, None]
        weighted_gt_mask = weighted_gt_mask.sum(1)

        brier_score = (
            (generated_seg_maps.float().mean(0) - weighted_gt_mask) ** 2
        ).mean()

        return {
            "wrong_area_iou": wrong_area_iou,
            "chosen_modes": chosen_modes,
            "brier_score": brier_score,
        }

    def compute_expected_number_of_coupons(self, dist):
        # Compute the expected number of samples to draw from the given non-uniform distribution until one of each item has been seen.
        # https://en.wikipedia.org/wiki/Coupon_collector%27s_problem

        # Reduce to non-zero clusters
        dist = dist[dist > 0]
        n = len(dist)

        # Source: "Birthday paradox, coupon collectors, caching algorithms and self-organizing search" 1992, eq 14b
        # https://www.sciencedirect.com/science/article/pii/0166218X9290177C
        agg = 1
        for i in range(1, n):
            plus_minus = (-1) ** (n - 1 - i)
            # Sum over all subsets with length i
            for subset in list(combinations(range(n), i)):
                agg += plus_minus * 1 / (1 - dist[list(subset)].sum())

        return agg

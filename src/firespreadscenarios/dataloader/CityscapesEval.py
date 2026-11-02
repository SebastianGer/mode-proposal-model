from itertools import combinations
from typing import Literal

import torch

BINARIZATION_MODES = Literal[
    "none", "cars", "five", "four", "four-v2", "four-v3", "four-v4"
]


def get_transforming_classes_and_distribution(
    binarization_mode: BINARIZATION_MODES = "five",
):
    class_distribution = torch.zeros(
        (35,)
    )  # We start at -1, so we shift by 1, and we go up until class index 33
    transforming_classes = (
        torch.tensor([7, 8, 21, 24, 26]) + 1
    )  # We start at -1, which we shift to 0, so it works as a regular index
    if binarization_mode == "none":
        probs = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    elif binarization_mode == "cars":
        probs = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.5])
    elif binarization_mode == "five":
        probs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    elif binarization_mode == "four":
        probs = torch.tensor([0.05, 0.25, 0.75, 0.95])
        transforming_classes = torch.tensor([7, 8, 21, 26]) + 1
    elif binarization_mode == "four-v2":
        probs = torch.tensor([0.2, 0.35, 0.65, 0.8])
        transforming_classes = torch.tensor([7, 8, 21, 26]) + 1
    elif binarization_mode == "four-v3":
        probs = torch.tensor([0.4, 0.45, 0.55, 0.6])
        transforming_classes = torch.tensor([7, 8, 21, 26]) + 1
    elif binarization_mode == "four-v4":
        probs = torch.tensor([0.5, 0.5, 0.5, 0.5])
        transforming_classes = torch.tensor([7, 8, 21, 26]) + 1
    class_distribution[transforming_classes] = probs
    return class_distribution, transforming_classes


class CityscapesEval:

    def __init__(self, binarization_mode: BINARIZATION_MODES = "five", device="cuda"):
        self.device = device

        self.class_distribution, self.flipping_classes = (
            get_transforming_classes_and_distribution(binarization_mode)
        )
        self.class_distribution = self.class_distribution.to(self.device)
        self.flipping_classes = self.flipping_classes.to(self.device)

        self.n_flipping_classes = len(self.flipping_classes)

    def reduce_segmap_to_one_hot_flipping_classes(self, full_seg_map):
        filtered_seg_map = torch.where(
            torch.isin(full_seg_map + 1, self.flipping_classes),
            full_seg_map + 1,
            torch.zeros_like(full_seg_map),
        )

        # Map the flipping classes to consecutive indices. If a position is not overwritten in this loop,
        # it contains a zero, which will be understood as the error class.
        flipping_classes_as_idx = torch.zeros_like(filtered_seg_map)
        for i, flipping_class in enumerate(self.flipping_classes):
            flipping_classes_as_idx[filtered_seg_map == flipping_class] = i + 1

        # One hot encoding of all classes to easily compute overlap between the correct area and predictions
        # Class index zero from the step before is also turned into ones here.
        one_hot = torch.nn.functional.one_hot(
            flipping_classes_as_idx, num_classes=self.n_flipping_classes + 1
        ).permute(0, 3, 1, 2)

        return one_hot.float()

    def compute_metrics(self, generated_seg_maps, full_seg_map):

        one_hot_seg_map = self.reduce_segmap_to_one_hot_flipping_classes(full_seg_map)
        if len(generated_seg_maps.shape) == 3:
            # If the predictions are not in the format B x C x H x W, add a channel dimension
            generated_seg_maps_4dim = generated_seg_maps.unsqueeze(1)
        else:
            generated_seg_maps_4dim = generated_seg_maps
        ious = (generated_seg_maps_4dim * one_hot_seg_map).sum(
            dim=(2, 3)
        ) / one_hot_seg_map.sum(dim=(2, 3))

        # Class 0 catches all areas that should never be 1 because they correspond to classes that we don't flip
        wrong_area_iou = ious[:, 0]
        good_areas_ious = ious[:, 1:]

        ### Compute image quality metric
        # For each class, the model can choose to predict all of it as true or false
        mode_strictness = torch.maximum(good_areas_ious, 1 - good_areas_ious)
        # Renormalize to [0,1], since the maximum above is always >= 0.5
        mode_strictness = (mode_strictness - 0.5) * 2

        ### Compute diversity metrics
        # determine chosen mode per class
        chosen_modes = (good_areas_ious > 0.5).float()

        # Set chosen mode to nan where it's nan in good_class_ious, to catch the cases of a class not being present in a given image.
        chosen_modes = torch.where(
            torch.isnan(good_areas_ious),
            torch.tensor(float("nan"), device=self.device),
            chosen_modes,
        )

        ### Brier score
        # Assign the flipping probabilities to the respective areas occupied by the flipping classes
        prob_dist = self.class_distribution[self.flipping_classes]
        # :-1 excludes the error class
        weighted_gt_mask = one_hot_seg_map[:, :-1] * prob_dist[None, :, None, None]
        weighted_gt_mask = weighted_gt_mask.sum(1)

        brier_score = (
            (generated_seg_maps.float().mean(0) - weighted_gt_mask) ** 2
        ).mean()

        return {
            "mode_strictness": mode_strictness,
            "wrong_area_iou": wrong_area_iou,
            "chosen_modes": chosen_modes,
            "brier_score": brier_score,
        }

    def get_true_cluster_distribution(self, cluster_ids):
        # Return the flipping probabilities of the five flipping classes.
        return self.class_distribution[self.flipping_classes]

    def get_uniform_cluster_distribution(self, cluster_ids=None):
        # Flipping decisions are seen as independent of each other, so they all have probability 1/2.
        n = len(self.flipping_classes)
        dist = torch.zeros((n,), device=self.device)
        dist[:] = 1 / 2

        return dist

    def get_probability_per_mode(self):
        # Compute the probabilities of all 2**n_classes combinations of flipping classes either to the positive or negative class
        prob_dist = self.class_distribution[self.flipping_classes]
        binary_strings = self.gen_all_binary_vectors(len(prob_dist))
        prob_per_mode = (
            prob_dist[None, :] * binary_strings
            + (1 - prob_dist[None, :]) * (1 - binary_strings)
        ).prod(1)
        return prob_per_mode

    def gen_all_binary_vectors(self, n_bits: int):
        return torch.tensor(
            [[int(x) for x in format(i, f"0{n_bits}b")] for i in range(2**n_bits)],
            device=self.device,
        )

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

    def get_batch_of_possible_modes(self, full_seg_map):
        # full_seg_map is the regular unfiltered segmentation map with all classes
        # We reduce this to the five flipping classes and return all possible modes
        # for this segmentation map of size [W,H] by iterating the 2**5 possible combinations
        # of flipping each class on or off.
        # Returns a binary tensor of size [2**5, W, H], where each row is a possible mode.
        # The number of rows is reduced if some classes are always on or always off.

        # Reduce to flipping classes, drop zero-class
        one_hot_seg_map = self.reduce_segmap_to_one_hot_flipping_classes(
            full_seg_map[None]
        )[
            :, 1:
        ]  # Size [1, 5, H, W]

        n = len(self.flipping_classes)

        # Produce all binary strings of length 5 as float tensor
        binary_combinations = torch.tensor(
            [list(map(int, bin(i)[2:].zfill(n))) for i in range(2**n)],
            device=self.device,
        ).float()  # Size [2**5, 5]

        # Correct for possibe always-on or always-off classes, in case we ever decide to have them

        for i, prob in enumerate(self.class_distribution[self.flipping_classes]):
            if prob == 0:
                binary_combinations[:, i] = 0
            elif prob == 1:
                binary_combinations[:, i] = 1

        # Remove duplicates in batch dim
        binary_combinations = torch.unique(binary_combinations, dim=0)

        possible_modes = one_hot_seg_map * binary_combinations[:, :, None, None]
        return possible_modes.sum(1)

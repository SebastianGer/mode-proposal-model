from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class SimfireDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        distribution_mode: Literal["uniform", "asymmetric"],
        indices: slice,
    ):
        """
        Args:
            x_path (str): Path to the X.npy file.
            y_path (str): Path to the Y.npy file.
            indices (slice): Slice object to select the subset of data.
        """
        self.distribution_mode = distribution_mode
        if distribution_mode == "uniform":
            class_distribution = torch.ones(8)
        elif distribution_mode == "asymmetric":
            class_distribution = torch.tensor([2**i for i in range(8)])

        class_distribution = (
            class_distribution / class_distribution.sum()
        )  # Normalize to sum to 1
        self.class_distribution = class_distribution

        self.X = torch.tensor(np.load(x_path)[indices])
        self.Y = torch.tensor(np.load(y_path)[indices])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        mask = self.Y[idx]

        chosen_mode = torch.multinomial(self.class_distribution, 1)

        mask_dict = {
            "all_targets": mask,
            "random_target": mask[chosen_mode],
            "target_summary": mask.float().mean(0),
            "chosen_mode": chosen_mode,
            "idx": idx,
        }
        return x, mask_dict

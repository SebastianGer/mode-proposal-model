from typing import Literal

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .SimfireDataset import SimfireDataset


class SimfireDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        distribution_mode: Literal["uniform", "asymmetric"],
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.distribution_mode = distribution_mode
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.x_path = f"{self.root_dir}/X.npy"
        self.y_path = f"{self.root_dir}/Y.npy"

    def setup(self, stage=None):
        self.train_dataset = SimfireDataset(
            self.x_path,
            self.y_path,
            distribution_mode=self.distribution_mode,
            indices=slice(0, 5000),
        )
        self.val_dataset = SimfireDataset(
            self.x_path,
            self.y_path,
            distribution_mode=self.distribution_mode,
            indices=slice(5000, 7500),
        )
        self.test_dataset = SimfireDataset(
            self.x_path,
            self.y_path,
            distribution_mode=self.distribution_mode,
            indices=slice(7500, None),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

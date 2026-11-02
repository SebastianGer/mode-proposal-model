import os

import pytorch_lightning as pl
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

from MoSEAUSeg.lidc_transforms import RandomHorizontalFlip
from MoSEAUSeg.lidc_transforms import RandomRotation
from MoSEAUSeg.lidc_transforms import RandomVerticalFlip
from MoSEAUSeg.LIDCDataset import lidc_Dataset


class LIDCDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = lidc_Dataset(
            os.path.join(self.data_dir, "train"),
            transforms.Compose(
                [RandomHorizontalFlip(), RandomVerticalFlip(), RandomRotation(angle=10)]
            ),
        )
        self.val_dataset = lidc_Dataset(os.path.join(self.data_dir, "val"))
        self.test_dataset = lidc_Dataset(os.path.join(self.data_dir, "test"))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

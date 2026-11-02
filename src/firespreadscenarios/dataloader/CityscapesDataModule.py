# Source: https://github.com/Lightning-Universe/lightning-bolts/blob/2c4602aa684e7b90e7ffdcea1d3f93a20f9c2ead/src/pl_bolts/datamodules/cityscapes_datamodule.py
"""
                              Apache License
                        Version 2.0, January 2004
                     http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

   "License" shall mean the terms and conditions for use, reproduction,
   and distribution as defined by Sections 1 through 9 of this document.

   "Licensor" shall mean the copyright owner or entity authorized by
   the copyright owner that is granting the License.

   "Legal Entity" shall mean the union of the acting entity and all
   other entities that control, are controlled by, or are under common
   control with that entity. For the purposes of this definition,
   "control" means (i) the power, direct or indirect, to cause the
   direction or management of such entity, whether by contract or
   otherwise, or (ii) ownership of fifty percent (50%) or more of the
   outstanding shares, or (iii) beneficial ownership of such entity.

   "You" (or "Your") shall mean an individual or Legal Entity
   exercising permissions granted by this License.

   "Source" form shall mean the preferred form for making modifications,
   including but not limited to software source code, documentation
   source, and configuration files.

   "Object" form shall mean any form resulting from mechanical
   transformation or translation of a Source form, including but
   not limited to compiled object code, generated documentation,
   and conversions to other media types.

   "Work" shall mean the work of authorship, whether in Source or
   Object form, made available under the License, as indicated by a
   copyright notice that is included in or attached to the work
   (an example is provided in the Appendix below).

   "Derivative Works" shall mean any work, whether in Source or Object
   form, that is based on (or derived from) the Work and for which the
   editorial revisions, annotations, elaborations, or other modifications
   represent, as a whole, an original work of authorship. For the purposes
   of this License, Derivative Works shall not include works that remain
   separable from, or merely link (or bind by name) to the interfaces of,
   the Work and Derivative Works thereof.

   "Contribution" shall mean any work of authorship, including
   the original version of the Work and any modifications or additions
   to that Work or Derivative Works thereof, that is intentionally
   submitted to Licensor for inclusion in the Work by the copyright owner
   or by an individual or Legal Entity authorized to submit on behalf of
   the copyright owner. For the purposes of this definition, "submitted"
   means any form of electronic, verbal, or written communication sent
   to the Licensor or its representatives, including but not limited to
   communication on electronic mailing lists, source code control systems,
   and issue tracking systems that are managed by, or on behalf of, the
   Licensor for the purpose of discussing and improving the Work, but
   excluding communication that is conspicuously marked or otherwise
   designated in writing by the copyright owner as "Not a Contribution."

   "Contributor" shall mean Licensor and any individual or Legal Entity
   on behalf of whom a Contribution has been received by Licensor and
   subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   copyright license to reproduce, prepare Derivative Works of,
   publicly display, publicly perform, sublicense, and distribute the
   Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   (except as stated in this section) patent license to make, have made,
   use, offer to sell, sell, import, and otherwise transfer the Work,
   where such license applies only to those patent claims licensable
   by such Contributor that are necessarily infringed by their
   Contribution(s) alone or by combination of their Contribution(s)
   with the Work to which such Contribution(s) was submitted. If You
   institute patent litigation against any entity (including a
   cross-claim or counterclaim in a lawsuit) alleging that the Work
   or a Contribution incorporated within the Work constitutes direct
   or contributory patent infringement, then any patent licenses
   granted to You under this License for that Work shall terminate
   as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
   Work or Derivative Works thereof in any medium, with or without
   modifications, and in Source or Object form, provided that You
   meet the following conditions:

   (a) You must give any other recipients of the Work or
       Derivative Works a copy of this License; and

   (b) You must cause any modified files to carry prominent notices
       stating that You changed the files; and

   (c) You must retain, in the Source form of any Derivative Works
       that You distribute, all copyright, patent, trademark, and
       attribution notices from the Source form of the Work,
       excluding those notices that do not pertain to any part of
       the Derivative Works; and

   (d) If the Work includes a "NOTICE" text file as part of its
       distribution, then any Derivative Works that You distribute must
       include a readable copy of the attribution notices contained
       within such NOTICE file, excluding those notices that do not
       pertain to any part of the Derivative Works, in at least one
       of the following places: within a NOTICE text file distributed
       as part of the Derivative Works; within the Source form or
       documentation, if provided along with the Derivative Works; or,
       within a display generated by the Derivative Works, if and
       wherever such third-party notices normally appear. The contents
       of the NOTICE file are for informational purposes only and
       do not modify the License. You may add Your own attribution
       notices within Derivative Works that You distribute, alongside
       or as an addendum to the NOTICE text from the Work, provided
       that such additional attribution notices cannot be construed
       as modifying the License.

   You may add Your own copyright statement to Your modifications and
   may provide additional or different license terms and conditions
   for use, reproduction, or distribution of Your modifications, or
   for any such Derivative Works as a whole, provided Your use,
   reproduction, and distribution of the Work otherwise complies with
   the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
   names, trademarks, service marks, or product names of the Licensor,
   except as required for reasonable and customary use in describing the
   origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
   agreed to in writing, Licensor provides the Work (and each
   Contributor provides its Contributions) on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied, including, without limitation, any warranties or conditions
   of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
   PARTICULAR PURPOSE. You are solely responsible for determining the
   appropriateness of using or redistributing the Work and assume any
   risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
   whether in tort (including negligence), contract, or otherwise,
   unless required by applicable law (such as deliberate and grossly
   negligent acts) or agreed to in writing, shall any Contributor be
   liable to You for damages, including any direct, indirect, special,
   incidental, or consequential damages of any character arising as a
   result of this License or out of the use or inability to use the
   Work (including but not limited to damages for loss of goodwill,
   work stoppage, computer failure or malfunction, or any and all
   other commercial damages or losses), even if such Contributor
   has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
   the Work or Derivative Works thereof, You may choose to offer,
   and charge a fee for, acceptance of support, warranty, indemnity,
   or other liability obligations and/or rights consistent with this
   License. However, in accepting such obligations, You may act only
   on Your own behalf and on Your sole responsibility, not on behalf
   of any other Contributor, and only if You agree to indemnify,
   defend, and hold each Contributor harmless for any liability
   incurred by, or claims asserted against, such Contributor by reason
   of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

APPENDIX: How to apply the Apache License to your work.

   To apply the Apache License to your work, attach the following
   boilerplate notice, with the fields enclosed by brackets "[]"
   replaced with your own identifying information. (Don't include
   the brackets!)  The text should be enclosed in the appropriate
   comment syntax for the file format. We also recommend that a
   file or class name and description of purpose be included on the
   same "printed page" as the copyright notice for easier
   identification within third-party archives.

Copyright 2018-2021 William Falcon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Changes:
# Removed unneeded warnings. Removed seed that doesn't seem to be used.
# Changed the default transformations for images.
# Changed the default transformations for targets, to deal with multi-modal targets.

from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2 as transform_lib

from .CityscapesDataset import CityscapesDataset as Cityscapes
from .CityscapesEval import (
    BINARIZATION_MODES,
    CityscapesEval,
    get_transforming_classes_and_distribution,
)


class CityscapesDataModule(LightningDataModule):
    """
    .. figure:: https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/muenster00-1024x510.png
        :width: 400
        :alt: Cityscape

    Standard Cityscapes, train, val, test splits and transforms

    Note: You need to have downloaded the Cityscapes dataset first and provide the path to where it is saved.
        You can download the dataset here: https://www.cityscapes-dataset.com/

    Specs:
        - 30 classes (road, person, sidewalk, etc...)
        - (image, target) - image dims: (3 x 1024 x 2048), target dims: (1024 x 2048)

    Transforms::

        transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transform_lib.Normalize(
                mean=[0.28689554, 0.32513303, 0.28389177],
                std=[0.18696375, 0.19017339, 0.18720214]
            )
        ])

    Example::

        from pl_bolts.datamodules import CityscapesDataModule

        dm = CityscapesDataModule(PATH)
        model = LitModel()

        Trainer().fit(model, datamodule=dm)

    Or you can set your own transforms

    Example::

        dm.train_transforms = ...
        dm.test_transforms = ...
        dm.val_transforms  = ...
        dm.target_transforms = ...
    """

    name = "Cityscapes"
    extra_args: dict = {}

    def __init__(
        self,
        data_dir: str,
        quality_mode: str = "fine",
        target_type: str = "instance",
        num_workers: int = 0,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        target_h: int = 64,
        target_w: int = 128,
        img_h: int = 64,
        img_w: int = 128,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        target_transforms: Optional[Callable] = None,
        binarization_mode: BINARIZATION_MODES = "five",
        resize_input: bool = True,
        use_alternate_split: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: where to load the data from path, i.e. where directory leftImg8bit and gtFine or gtCoarse
                are located
            quality_mode: the quality mode to use, either 'fine' or 'coarse'
            target_type: targets to use, either 'instance' or 'semantic'
            num_workers: how many workers to use for loading data
            batch_size: number of examples per training/eval step
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
            use_alternate_split: If true, splits the official training set into a new train and validation set according to val_split,
            and uses the official validation set as the test set. If false, uses the official train/val/test splits.
        """
        super().__init__(*args, **kwargs)

        if target_type not in ["instance", "semantic"]:
            raise ValueError(
                f'Only "semantic" and "instance" target types are supported. Got {target_type}.'
            )

        self.dims = (3, 1024, 2048)
        self.data_dir = data_dir
        self.quality_mode = quality_mode
        self.target_type = target_type
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.target_h = target_h
        self.target_w = target_w
        self.img_h = img_h
        self.img_w = img_w
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.target_transforms = target_transforms
        self.resize_input = resize_input
        self.use_alternate_split = use_alternate_split
        self.class_distribution, _ = get_transforming_classes_and_distribution(
            binarization_mode
        )

        self.cityscapes_eval = CityscapesEval(
            binarization_mode=binarization_mode, device="cpu"
        )

    @property
    def num_classes(self) -> int:
        """
        Return:
            30
        """
        return 30

    def _get_alternate_split_subset(self, full_dataset, subset_type: str):
        """
        Compute deterministic train/val split indices if not already cached,
        and return the appropriate Subset for the given type.

        Args:
            full_dataset: The full Cityscapes dataset object (with desired transforms).
            subset_type: Either 'train' or 'val' to specify which subset to return.

        Returns:
            A Subset wrapping full_dataset with the appropriate indices.
        """
        # Compute indices if not already cached
        if not hasattr(self, "_train_split_indices"):
            n = len(full_dataset)
            val_n = 500  # split off 500 samples for validation

            generator = torch.Generator().manual_seed(0)
            perm = torch.randperm(n, generator=generator)
            val_idx = perm[:val_n].tolist()
            train_idx = perm[val_n:].tolist()

            # Store indices for reuse
            self._train_split_indices = (train_idx, val_idx)

        train_idx, val_idx = self._train_split_indices

        # Return the appropriate subset
        if subset_type == "train":
            return Subset(full_dataset, train_idx)
        elif subset_type == "val":
            return Subset(full_dataset, val_idx)
        else:
            raise ValueError(f"subset_type must be 'train' or 'val', got {subset_type}")

    def train_dataset(self) -> Cityscapes:
        transforms = self.train_transforms or self._default_train_transforms()
        target_transforms = self.target_transforms or self._default_target_transforms(
            torch.Generator()
        )

        if self.use_alternate_split:
            # Create full dataset with training transforms, then get train subset
            train_full = Cityscapes(
                self.data_dir,
                split="train",
                target_type=self.target_type,
                mode=self.quality_mode,
                transform=transforms,
                target_transform=target_transforms,
                **self.extra_args,
            )
            return self._get_alternate_split_subset(train_full, "train")

        # default: return the full official training split
        dataset = Cityscapes(
            self.data_dir,
            split="train",
            target_type=self.target_type,
            mode=self.quality_mode,
            transform=transforms,
            target_transform=target_transforms,
            **self.extra_args,
        )

        return dataset

    def train_dataloader(self) -> DataLoader:
        """Cityscapes train set."""
        dataset = self.train_dataset()

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataset(self) -> Cityscapes:
        transforms = self.val_transforms or self._default_transforms()
        target_transforms = self.target_transforms or self._default_target_transforms(
            torch.Generator().manual_seed(0)
        )

        if self.use_alternate_split:
            # Create full dataset with validation transforms, then get val subset
            val_full = Cityscapes(
                self.data_dir,
                split="train",
                target_type=self.target_type,
                mode=self.quality_mode,
                transform=transforms,
                target_transform=target_transforms,
                **self.extra_args,
            )
            return self._get_alternate_split_subset(val_full, "val")

        # default: use the official validation split
        dataset = Cityscapes(
            self.data_dir,
            split="val",
            target_type=self.target_type,
            mode=self.quality_mode,
            transform=transforms,
            target_transform=target_transforms,
            **self.extra_args,
        )

        return dataset

    def val_dataloader(self) -> DataLoader:
        """Cityscapes val set."""

        dataset = self.val_dataset()

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=True,
        )

    def test_dataset(self) -> Cityscapes:
        transforms = self.test_transforms or self._default_transforms()
        target_transforms = self.target_transforms or self._default_target_transforms(
            torch.Generator().manual_seed(0)
        )

        if self.use_alternate_split:
            # use the official validation split as the test dataset
            dataset = Cityscapes(
                self.data_dir,
                split="val",
                target_type=self.target_type,
                mode=self.quality_mode,
                transform=transforms,
                target_transform=target_transforms,
                **self.extra_args,
            )
            return dataset

        dataset = Cityscapes(
            self.data_dir,
            split="test",
            target_type=self.target_type,
            mode=self.quality_mode,
            transform=transforms,
            target_transform=target_transforms,
            **self.extra_args,
        )

        return dataset

    def test_dataloader(self) -> DataLoader:
        """Cityscapes test set."""
        dataset = self.test_dataset()

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def _default_transforms(self) -> Callable:
        tfs = [
            transform_lib.ToImage(),
            transform_lib.ToDtype(torch.float32, scale=True),
        ]
        if self.resize_input:
            # Pretrained detectron model takes full size image, but image-conditioned diffusion model does not.
            tfs += [transform_lib.Resize((self.img_h, self.img_w))]

        # ImageNet normalization, is required by detectron model
        tfs += [
            transform_lib.Normalize(
                mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
            )
        ]
        return transform_lib.Compose(tfs)

    def _default_train_transforms(self) -> Callable:
        tfs = [
            transform_lib.ToImage(),
            transform_lib.ToDtype(torch.float32, scale=True),
        ]
        if self.resize_input:
            # Pretrained detectron model takes full size image, but image-conditioned diffusion model does not.
            tfs += [transform_lib.Resize((self.img_h, self.img_w))]

        # ImageNet normalization, is required by detectron model
        tfs += [
            transform_lib.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transform_lib.Normalize(
                mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
            ),
        ]
        return transform_lib.Compose(tfs)

    def old_default_target_transforms(self, torch_rand_generator) -> Callable:
        transform_list = [
            transform_lib.ToImage(),
            transform_lib.ToDtype(torch.long),
            transform_lib.Resize(
                (self.target_h, self.target_w),
                interpolation=transform_lib.InterpolationMode.NEAREST,
            ),
            transform_lib.Lambda(lambda t: t.long().squeeze()),
        ]

        def randomly_binarize_mask(mask):
            # Randomly flip classes to on or off
            binarization_mask = torch.bernoulli(
                self.class_distribution, generator=torch_rand_generator
            )
            return binarization_mask[mask + 1]

        def deterministically_binarize_mask(mask):
            # Set all flipping classes to on, rest to off, to be able to see which areas are okay to flip.
            binarization_mask = (self.class_distribution > 0).float()
            return binarization_mask[mask + 1]

        transform_list.append(
            transform_lib.Lambda(
                lambda img: (
                    randomly_binarize_mask(img),
                    deterministically_binarize_mask(img),
                    img,
                )
            )
        )

        return transform_lib.Compose(transform_list)

    def _default_target_transforms(self, torch_rand_generator) -> Callable:
        transform_list = [
            transform_lib.ToImage(),
            transform_lib.ToDtype(torch.long),
            transform_lib.Resize(
                (self.target_h, self.target_w),
                interpolation=transform_lib.InterpolationMode.NEAREST,
            ),
            transform_lib.Lambda(lambda t: t.long().squeeze()),
        ]

        def randomly_binarize_mask(mask):
            # Randomly flip classes to on or off, given the probabilities
            binarization_mask = torch.bernoulli(
                self.class_distribution, generator=torch_rand_generator
            )
            return binarization_mask[mask + 1]

        def deterministically_binarize_mask(mask):
            # Set all flipping classes to on, rest to off, to be able to see which areas are okay to flip.
            binarization_mask = (self.class_distribution > 0).float()
            return binarization_mask[mask + 1]

        def get_all_modes(mask):
            return self.cityscapes_eval.get_batch_of_possible_modes(mask)

        transform_list.append(
            transform_lib.Lambda(
                lambda img: {
                    "random_target": randomly_binarize_mask(img),
                    "target_summary": deterministically_binarize_mask(img) * img,
                    "all_targets": get_all_modes(img).int(),
                    "full_seg_mask": img,
                }
            )
        )

        return transform_lib.Compose(transform_list)

from typing import Any

import torch

from firespreadscenarios.dataloader.LIDCEval import LIDCEval

from .BaseDiffusionTraining import BaseDiffusionTraining


class LIDCMixin:

    def get_dataset_evaluator(self, dataset_evaluator_arg: Any):
        return LIDCEval()

    def load_concrete_backbone_model(self):
        raise NotImplementedError()

    def unbatch(self, batch):
        # Should return a dictionary with:
        # - image: B x C x H x W
        # - target: B x H x W, randomly chosen if multiple exist
        # - target_summary: B x H x W, contains some kind of summary of the target, e.g. the mean over all modes; used for visualization
        # - all_targets: B x N x H x W, all targets

        if len(batch) == 3:
            image, targets, _ = batch
        else:
            image, targets = batch
        batch_size, n_targets, _, _ = targets.shape

        target_idcs = torch.randint(low=0, high=n_targets, size=(batch_size,))
        chosen_targets = targets[torch.arange(batch_size), target_idcs]

        batch_dict = {
            "image": image,
            "target": chosen_targets,
            "target_summary": targets.float().mean(1),
            "all_targets": targets,
            "dataset_eval_input": targets,
            "logging_target": targets.float().mean(1),
        }
        return batch_dict


class LIDCDiffusionTraining(LIDCMixin, BaseDiffusionTraining):
    pass

from typing import Any

from firespreadscenarios.dataloader.SimfireEval import SimfireEval

from .BaseDiffusionTraining import BaseDiffusionTraining


class SimfireMixin:

    def get_dataset_evaluator(self, dataset_evaluator_arg: Any):
        return SimfireEval(dataset_evaluator_arg)

    def load_concrete_backbone_model(self):
        raise NotImplementedError()

    def unbatch(self, batch):
        # Should return a dictionary with:
        # - image: B x C x H x W
        # - target: B x H x W, randomly chosen if multiple exist
        # - target_summary: B x H x W, contains some kind of summary of the target, e.g. the mean over all modes; used for visualization
        # - all_targets: B x N x H x W, all targets

        image, mask_dict = batch

        batch_dict = {
            "image": image,
            "target": mask_dict["random_target"],
            "target_summary": mask_dict["target_summary"],
            "all_targets": mask_dict["all_targets"],
            "dataset_eval_input": mask_dict["all_targets"],
            "logging_target": mask_dict["target_summary"],
            "idx": mask_dict["idx"],
        }
        return batch_dict


class SimfireDiffusionTraining(SimfireMixin, BaseDiffusionTraining):
    pass

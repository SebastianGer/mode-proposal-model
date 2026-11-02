from firespreadscenarios.dataloader.CityscapesEval import BINARIZATION_MODES, CityscapesEval

from .BaseDiffusionTraining import BaseDiffusionTraining


class CityscapesMixin:
    def get_dataset_evaluator(
        self, binarization_mode: BINARIZATION_MODES = "four"
    ):
        return CityscapesEval(binarization_mode)

    def load_concrete_backbone_model(self):
        raise NotImplementedError("We only condition on image inputs at the moment.")

    def unbatch(self, batch):
        # Should return a dictionary with:
        # - image: B x C x H x W
        # - target: B x H x W, randomly chosen if multiple exist
        # - target_summary: B x H x W, contains some kind of summary of the target, e.g. the mean over all modes; used for visualization
        # - all_targets: B x N x H x W, all targets

        image, targets, _ = batch
        batch_size = image.shape[0]

        batch_dict = {
            "image": image,
            "target": targets["random_target"],
            "target_summary": targets["target_summary"],
            "all_targets": targets["all_targets"],
            "dataset_eval_input": targets["full_seg_mask"],
            "logging_target": targets["target_summary"],
        }
        return batch_dict


class CityscapesDiffusionTraining(CityscapesMixin, BaseDiffusionTraining):
    pass

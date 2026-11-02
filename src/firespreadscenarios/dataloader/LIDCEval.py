from typing import Any


class LIDCEval:

    def __init__(self, dummy_arg: Any = None, device="cuda"):
        # Dummy arg only used to match the signature of the CityscapesEval class
        self.device = device

    def compute_metrics(self, generated_seg_maps, all_targets):
        if len(generated_seg_maps.shape) == 3 and len(all_targets.shape) == 4:
            # If the generated segmentation maps are 3D and the targets are 4D, we need to add a channel dimension to the generated maps
            generated_seg_maps = generated_seg_maps.unsqueeze(1)

        # Compare mean generated and mean target segmentation maps as a way to assess the calibration of the model
        # This is a very simple metric, but it can be useful to get a rough idea of how well the model is calibrated
        # We compute the square of the pixelwise difference, to arrive at something resembling the Brier score.
        brier_score = (
            (generated_seg_maps.float().mean(0) - all_targets.float().mean(0)) ** 2
        ).mean()

        return {"brier_score": brier_score}

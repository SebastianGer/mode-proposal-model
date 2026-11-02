from typing import Literal

from ..generate_diverse import edm_sampler_shielding_guidance
from .SamplingStrategy import SamplingStrategy


class ShieldingGuidance(SamplingStrategy):
    """
    Based on http://arxiv.org/abs/2410.06025

    ShieldingGuidance encourages sample diversity by repelling denoised predictions
    from one another whenever they fall within a radius `shield_radius` in pixel space.
    The repulsion is applied as a shift to the denoising target at each diffusion step.
    """

    def __init__(
        self,
        pg_s_min: float = 0.0,  # lowest noise level at which shielding guidance is applied
        repulsion_mode: Literal[
            "samples_only", "memory_bank_only", "samples_and_memory_bank"
        ] = "samples_only",
        guidance_factor: float = 0.0,
        shield_radius: float = 0.0,
    ):
        self.pg_s_min = pg_s_min
        self.repulsion_mode = repulsion_mode
        self.guidance_factor = guidance_factor
        self.shield_radius = shield_radius

    def sample(
        self, diffusion_model, latents, replicated_cond, existing_samples, hparams
    ):
        return edm_sampler_shielding_guidance(
            diffusion_model,
            latents,
            conditioning=replicated_cond,
            rho=hparams.rho,
            num_steps=hparams.timesteps,
            sigma_max=hparams.sigma_max,
            S_churn=hparams.s_churn,
            S_max=hparams.s_max,
            S_min=hparams.s_min,
            use_second_order_correction=hparams.use_second_order_correction,
            asymmetric_time_difference=hparams.asymmetric_time_difference,
            conditioning_noise_factor=hparams.conditioning_noise_factor,
            guidance_factor=self.guidance_factor,
            shield_radius=self.shield_radius,
            pg_s_min=self.pg_s_min,
            repulsion_mode=self.repulsion_mode,
            repulsive_items=existing_samples,
        )

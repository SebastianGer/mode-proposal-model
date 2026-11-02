from typing import Literal

from ..generate_diverse import edm_sampler2
from .SamplingStrategy import SamplingStrategy


class ParticleGuidance(SamplingStrategy):
    """
    Based on https://openreview.net/forum?id=hEyIHsyZ9F

    ParticleGuidance is a sampling strategy that uses particle guidance to sample from the diffusion model.
    Should also be used as the naive sampling strategy, by simply leaving particle_guidance factor at 0.0.
    """

    def __init__(
        self,
        particle_guidance_factor: float = 0.0,
        pg_s_min: float = 0.0,  # particle guidance: S_min; lowest noise level at which we still use particle guidance
        pg_kernel_type: Literal["L2", "log_det"] = "L2",
        pg_clip_gradient_norm: float = -1.0,  # clip gradients during particle guidance if value is positive
        pg_as_guidance: bool = True,
        repulsion_mode: Literal[
            "samples_only", "memory_bank_only", "samples_and_memory_bank"
        ] = "samples_only",
        conditioning_noise_factor: float = 0.0,
    ):
        self.particle_guidance_factor = particle_guidance_factor
        self.pg_s_min = pg_s_min
        self.pg_kernel_type = pg_kernel_type
        self.pg_clip_gradient_norm = pg_clip_gradient_norm
        self.pg_as_guidance = pg_as_guidance
        self.repulsion_mode = repulsion_mode
        self.conditioning_noise_factor = conditioning_noise_factor

    def sample(
        self, diffusion_model, latents, replicated_cond, existing_samples, hparams
    ):
        return edm_sampler2(
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
            conditioning_noise_factor=self.conditioning_noise_factor,
            particle_guidance_factor=self.particle_guidance_factor,
            pg_s_min=self.pg_s_min,
            pg_kernel_type=self.pg_kernel_type,
            pg_clip_gradient_norm=self.pg_clip_gradient_norm,
            pg_as_guidance=self.pg_as_guidance,
            repulsion_mode=self.repulsion_mode,
            repulsive_items=existing_samples,
        )

# Written by Claude Code (Anthropic). The sampler itself is in
# generate_diverse.py:edm_sampler_dpmsolver_pp, which is adapted from k-diffusion (MIT License, see NOTICE).
from ..generate_diverse import edm_sampler_dpmsolver_pp
from .SamplingStrategy import SamplingStrategy


class DPMSolverPlusPlus(SamplingStrategy):
    """
    Naive (non-guided) sampling using DPM-Solver++(2M), a deterministic
    multistep ODE solver that reaches 2nd-order accuracy with only one
    network evaluation per step instead of the two needed by edm_sampler2's
    Heun correction. Unlike DPM-Solver-v3, it needs no calibration step,
    since its update coefficients come directly from the EDM noise schedule.

    Does not support particle/shielding guidance; use ParticleGuidance with
    particle_guidance_factor=0.0 or ShieldingGuidance with guidance_factor=0.0
    instead if guided sampling is needed.
    """

    def sample(
        self, diffusion_model, latents, replicated_cond, existing_samples, hparams
    ):
        return edm_sampler_dpmsolver_pp(
            diffusion_model,
            latents,
            conditioning=replicated_cond,
            rho=hparams.rho,
            num_steps=hparams.timesteps,
            sigma_max=hparams.sigma_max,
        )

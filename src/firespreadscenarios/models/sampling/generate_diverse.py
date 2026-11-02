# In this file, various derived versions of the original EDM sampler are implemented.
# The original sampler code can be found at third_party/edm/generate.py:edm_sampler.
# The parts of the methods in this file that are the same as in the original code fall under the original copyright:
#     Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

from typing import Literal

import torch

from edm.generate import euler_step, time_step_discretization

from .clustering import clustering
from .diversity_functions import (compute_particle_guidance_grad,
                                  compute_shielding_guidance_grad)
from .utils import noisy_conditioning


def edm_sampler2(
    net,
    latents,
    conditioning=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    use_second_order_correction=True,
    asymmetric_time_difference: float = 0.0,
    conditioning_noise_factor: float = 0.0,
    particle_guidance_factor: float = 0.0,
    pg_s_min: float = 0.0,
    pg_s_max: float = float("inf"),
    pg_kernel_type: Literal["L2", "log_det"] = "L2",
    pg_clip_gradient_norm: float = -1,
    pg_as_guidance: bool = False,
    repulsion_mode: Literal[
        "samples_only", "memory_bank_only", "samples_and_memory_bank"
    ] = "samples_only",
    repulsive_items: torch.Tensor | None = None,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    t_steps, t_step_pairs = time_step_discretization(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        num_steps=num_steps,
        rho=rho,
        device=latents.device,
    )

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    x_steps = [x_next]
    x0_preds, pg_grads = [], []
    for i, (t_cur, t_next) in enumerate(t_step_pairs):
        x_cur = x_next

        # Only use particle guidance while we have a noise level of at least pg_s_min.
        # We want to encourage diversity when switching between modes, but after that we should not
        # push images in the same mode away from each other.
        if (
            not pg_as_guidance
            and particle_guidance_factor != 0
            and pg_s_min <= t_cur <= pg_s_max
        ):
            denoised, pg_grad, _ = compute_particle_guidance_grad(
                net,
                x_cur,
                t_cur,
                t_cur,
                conditioning,
                conditioning_noise_factor,
                pg_kernel_type,
                pg_clip_gradient_norm,
                repulsion_mode,
                repulsive_items,
            )
            particle_guidance_grad = particle_guidance_factor * t_cur * pg_grad
            pg_grads.append(particle_guidance_grad.detach().clone())

            x_cur = x_cur + particle_guidance_grad

        # SDE Euler step: Add noise, denoise, then take a big step in the direction of the denoised image.
        x_hat, t_hat, d_cur, denoised = euler_step(
            x_cur,
            t_cur,
            net,
            S_churn,
            num_steps,
            S_min,
            S_max,
            S_noise,
            sigma_min,
            asymmetric_time_difference,
            conditioning,
            conditioning_noise_factor,
        )
        x0_preds.append(denoised)

        # If used as guidance, we need to compute PG from the image with the added SDE noise (x_hat), not from x_cur.
        particle_guidance_grad = torch.zeros_like(x_cur)
        if (
            pg_as_guidance
            and particle_guidance_factor != 0
            and pg_s_min <= t_cur <= pg_s_max
        ):
            denoised, pg_grad, _ = compute_particle_guidance_grad(
                net,
                x_hat,
                t_hat,
                t_hat,
                conditioning,
                conditioning_noise_factor,
                pg_kernel_type,
                pg_clip_gradient_norm,
                repulsion_mode,
                repulsive_items,
            )
            particle_guidance_grad = particle_guidance_factor * t_hat * pg_grad
            pg_grads.append(particle_guidance_grad.detach().clone())

        x_next = x_hat + (t_next - t_hat) * (d_cur - particle_guidance_grad)

        # Apply 2nd order correction.
        if use_second_order_correction and (i < num_steps - 1):
            t_async = max(
                torch.full_like(t_next, fill_value=sigma_min),
                t_next - asymmetric_time_difference,
            )
            denoised = net(
                x_next,
                t_async,
                noisy_conditioning(t_next, conditioning, conditioning_noise_factor),
            ).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        x_steps.append(x_next)

    return {
        "x_next": x_next,
        "x_steps": x_steps,
        "x0_preds": x0_preds,
        "pg_grads": pg_grads,
    }


def edm_sampler_clustering(
    net,
    latents,
    conditioning=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    use_second_order_correction=True,
    asymmetric_time_difference: float = 0.0,
    conditioning_noise_factor: float = 0.0,
    clustering_distance_metric: Literal["chamfer", "L2"] = "chamfer",
    n_clusters: int = 8,
    prune_clusters: bool = False,
    repulsion_mode: Literal[
        "samples_only", "memory_bank_only", "samples_and_memory_bank"
    ] = "samples_only",
    repulsive_items: torch.Tensor | None = None,
):
    """
    Clustering could be used as an add-on to all ways of sampling, but we focus on the basic version here, instead of spending effort to implement more complex functions.
    """

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    t_steps, t_step_pairs = time_step_discretization(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        num_steps=num_steps,
        rho=rho,
        device=latents.device,
    )

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    x_steps = [x_next]
    x0_preds = []
    pg_grads = []
    cluster_ids = []
    for i, (t_cur, t_next) in enumerate(t_step_pairs):
        x_cur = x_next

        # SDE Euler step: Add noise, denoise, then take a big step in the direction of the denoised image.
        x_hat, t_hat, d_cur, denoised = euler_step(
            x_cur,
            t_cur,
            net,
            S_churn,
            num_steps,
            S_min,
            S_max,
            S_noise,
            sigma_min,
            asymmetric_time_difference,
            conditioning,
            conditioning_noise_factor,
        )
        x0_preds.append(denoised)

        x_next = x_hat + (t_next - t_hat) * (d_cur)

        # Apply 2nd order correction.
        if use_second_order_correction and (i < num_steps - 1):
            t_async = max(
                torch.full_like(t_next, fill_value=sigma_min),
                t_next - asymmetric_time_difference,
            )
            denoised = net(
                x_next,
                t_async,
                noisy_conditioning(t_next, conditioning, conditioning_noise_factor),
            ).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        x_steps.append(x_next)

        denoised_cluster_medoids_ids = clustering(
            (denoised > 0).int().detach(),
            distance_metric=clustering_distance_metric,
            n_clusters=n_clusters,
            existing_samples=None,
        )
        cluster_ids.append(denoised_cluster_medoids_ids)
        if prune_clusters and x_next.shape[0] > n_clusters:
            # Only continue denoising the cluster medoids, not the whole batch, saving computational cost.
            x_next = x_next[denoised_cluster_medoids_ids]
            conditioning = conditioning[denoised_cluster_medoids_ids]

    return {
        "x_next": x_next,
        "x_steps": x_steps,
        "x0_preds": x0_preds,
        "pg_grads": pg_grads,
        "cluster_ids": cluster_ids,
    }


def edm_sampler_shielding_guidance(
    net,
    latents,
    conditioning=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    use_second_order_correction=True,
    asymmetric_time_difference: float = 0.0,
    conditioning_noise_factor: float = 0.0,
    guidance_factor: float = 0.0,
    pg_s_min: float = 0.0,
    pg_s_max: float = float("inf"),
    pg_kernel_type: Literal["L2", "log_det"] = "L2",
    repulsion_mode: Literal[
        "samples_only", "memory_bank_only", "samples_and_memory_bank"
    ] = "samples_only",
    repulsive_items: torch.Tensor | None = None,
    shield_radius: float = 0.0,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    t_steps, t_step_pairs = time_step_discretization(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        num_steps=num_steps,
        rho=rho,
        device=latents.device,
    )

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    x_steps = [x_next]
    x0_preds, pg_grads = [], []
    for i, (t_cur, t_next) in enumerate(t_step_pairs):
        x_cur = x_next

        # Only use particle guidance while we have a noise level of at least pg_s_min.
        # We want to encourage diversity when switching between modes, but after that we should not
        # push images in the same mode away from each other.
        if guidance_factor != 0 and pg_s_min <= t_cur <= pg_s_max:
            denoised, sg_grad = compute_shielding_guidance_grad(
                net,
                x_cur,
                t_cur,
                conditioning,
                conditioning_noise_factor,
                shield_radius,
                repulsion_mode,
                repulsive_items,
            )
            shielding_guidance_repellency = t_cur * sg_grad
            pg_grads.append(shielding_guidance_repellency.detach().clone())
            x0_preds.append(denoised)

            # Shift the target towards which we're denoising!
            denoised = denoised + shielding_guidance_repellency
        else:
            # SDE Euler step: Add noise, denoise, then take a big step in the direction of the denoised image.
            _, _, _, denoised = euler_step(
                x_cur,
                t_cur,
                net,
                S_churn,
                num_steps,
                S_min,
                S_max,
                S_noise,
                sigma_min,
                asymmetric_time_difference,
                conditioning,
                conditioning_noise_factor,
            )
            x0_preds.append(denoised)

        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        # Apply 2nd order correction.
        if use_second_order_correction and (i < num_steps - 1):
            denoised = net(
                x_next,
                t_next,
                noisy_conditioning(t_next, conditioning, conditioning_noise_factor),
            ).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

        x_steps.append(x_next)

    return {
        "x_next": x_next,
        "x_steps": x_steps,
        "x0_preds": x0_preds,
        "pg_grads": pg_grads,
    }


# edm_sampler_dpmsolver_pp was written by Claude Code (Anthropic).
# It is adapted from sample_dpmpp_2m in k-diffusion
# (https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py),
# Copyright (c) 2022 Katherine Crowson, licensed under the MIT License.
# The full MIT license text is in the NOTICE file.
def edm_sampler_dpmsolver_pp(
    net,
    latents,
    conditioning=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
):
    """
    DPM-Solver++(2M): a deterministic, calibration-free multistep ODE solver.
    Reaches 2nd-order accuracy using only one network evaluation per step
    (after the first), unlike the Heun correction in edm_sampler2, which needs
    two. No diversity-promoting guidance is implemented here, since this
    sampler is intended purely for naive (non-guided) sampling.

    Source: C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu, "DPM-Solver++:
    Fast Solver for Guided Sampling of Diffusion Probability Models,"
    arXiv:2211.01095. Closely follows the reference implementation
    `sample_dpmpp_2m` in https://github.com/crowsonkb/k-diffusion, adapted to
    this repo's net()/conditioning calling convention and EDM time schedule.

    Derivation note: EDM uses a VE schedule (alpha_t=1), so half-logSNR
    lambda_t = -log(sigma_t). The data-prediction update below thus only
    needs sigma ratios and log-sigma differences, never alpha_t terms.
    """
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    t_steps, t_step_pairs = time_step_discretization(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        num_steps=num_steps,
        rho=rho,
        device=latents.device,
    )

    x_next = latents.to(torch.float64) * t_steps[0]
    x_steps = [x_next]
    x0_preds = []
    old_denoised = None

    for i, (t_cur, t_next) in enumerate(t_step_pairs):
        x_cur = x_next

        denoised = net(x_cur, t_cur, conditioning).to(torch.float64)
        x0_preds.append(denoised)

        if old_denoised is None or t_next == 0:
            # First step (no history yet) or the final step to sigma=0:
            # falls back to the 1st-order update, equivalent to a DDIM/Euler step.
            denoised_d = denoised
        else:
            lambda_cur = -t_cur.log()
            lambda_next = -t_next.log()
            lambda_prev = -t_steps[i - 1].log()
            h = lambda_next - lambda_cur
            h_last = lambda_cur - lambda_prev
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised

        x_next = (t_next / t_cur) * x_cur + (1 - t_next / t_cur) * denoised_d
        old_denoised = denoised
        x_steps.append(x_next)

    return {
        "x_next": x_next,
        "x_steps": x_steps,
        "x0_preds": x0_preds,
        "pg_grads": [],
    }

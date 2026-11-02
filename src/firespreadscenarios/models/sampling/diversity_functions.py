import math
from typing import Literal

import torch

from .utils import noisy_conditioning


def compute_particle_guidance_grad(
    net,
    x_hat_,
    t_async,
    t_hat,
    conditioning,
    conditioning_noise_factor,
    pg_kernel_type: Literal["L2", "log_det"],
    pg_clip_gradient_norm: float = -1.0,
    repulsion_mode: Literal[
        "samples_only", "memory_bank_only", "samples_and_memory_bank"
    ] = "samples_only",
    repulsive_items=None,
):
    # Base version according to:
    # G. Corso, Y. Xu, V. de Bortoli, R. Barzilay, and T. Jaakkola,
    # “Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models,” Nov. 24, 2023, arXiv: arXiv:2310.13102.
    # http://arxiv.org/abs/2310.13102
    #

    with torch.enable_grad():

        x_hat = x_hat_.detach().clone()
        x_hat.requires_grad = True
        denoised = net(
            x_hat,
            t_async,
            noisy_conditioning(t_hat, conditioning, conditioning_noise_factor),
        ).to(torch.float64)

        return compute_particle_guidance_grad_from_denoised(
            x_hat=x_hat,
            denoised=denoised,
            pg_kernel_type=pg_kernel_type,
            pg_clip_gradient_norm=pg_clip_gradient_norm,
            repulsion_mode=repulsion_mode,
            repulsive_items=repulsive_items,
        )


def compute_particle_guidance_grad_from_denoised(
    x_hat,
    denoised,
    pg_kernel_type: Literal["L2", "log_det"],
    pg_clip_gradient_norm: float = -1.0,
    repulsion_mode: Literal[
        "samples_only", "memory_bank_only", "samples_and_memory_bank"
    ] = "samples_only",
    repulsive_items=None,
):
    # Base version according to:
    # G. Corso, Y. Xu, V. de Bortoli, R. Barzilay, and T. Jaakkola,
    # “Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models,” Nov. 24, 2023, arXiv: arXiv:2310.13102.
    # http://arxiv.org/abs/2310.13102
    #

    with torch.enable_grad():

        xs = (denoised + 1) * 0.5  # go from [-1,1] to [0,1]
        n = xs.shape[0]

        # Default case: Repulse items from each other
        # Special case: Batch of items is given from which we want to repulse the items, e.g. already generated samples.
        samples_flattened = xs.flatten(1)
        if repulsive_items is None or repulsion_mode == "samples_only":
            repulsive_items_flattened = xs.flatten(1)
        elif repulsion_mode == "memory_bank_only":
            repulsive_items_flattened = repulsive_items.flatten(1)
        elif repulsion_mode == "samples_and_memory_bank":
            repulsive_items_flattened = torch.cat(
                [xs.flatten(1), repulsive_items.flatten(1)], dim=0
            )
        # Only backprop to samples.
        repulsive_items_flattened = repulsive_items_flattened.detach()

        distance_matrix = torch.cdist(samples_flattened, repulsive_items_flattened, p=2)

        # Normalizing factor
        h_t = distance_matrix.median().detach() ** 2 / math.log(n)
        rbf_kernel = (-distance_matrix / h_t).exp()

        if pg_kernel_type == "log_det":
            log_det = -torch.logdet(rbf_kernel)
            log_det.backward()
        elif pg_kernel_type == "L2":
            rbf_sum = rbf_kernel.sum()
            rbf_sum.backward()

        if pg_clip_gradient_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters=x_hat, max_norm=pg_clip_gradient_norm
            )

        return denoised.detach(), x_hat.grad, distance_matrix.detach()


def compute_shielding_guidance_grad(
    net,
    x_hat,
    t_hat,
    conditioning,
    conditioning_noise_factor,
    shield_radius: float = 0.0,
    repulsion_mode: Literal[
        "samples_only", "memory_bank_only", "samples_and_memory_bank"
    ] = "samples_only",
    repulsive_items=None,
):

    # Code based on the Shielded Diffusion paper: http://arxiv.org/abs/2410.06025

    denoised = net(
        x_hat, t_hat, noisy_conditioning(t_hat, conditioning, conditioning_noise_factor)
    ).to(torch.float64)
    xs = (denoised + 1) * 0.5  # go from [-1,1] to [0,1]

    relu = torch.nn.ReLU()
    repellency_term = 0

    # Repel within batch
    samples_flattened = xs.flatten(1)
    if repulsive_items is None or repulsion_mode in [
        "samples_only",
        "samples_and_memory_bank",
    ]:
        # diff_vec has size [batch, batch, dimensions]
        diff_vec = samples_flattened.unsqueeze(1) - samples_flattened.unsqueeze(0)
        weight = (diff_vec**2).sum(dim=2).sqrt()
        trunc_weight = relu(shield_radius / weight - 1)
        trunc_weight.diagonal().fill_(0)  # Don’t repel from the image itself
        trunc_weight = trunc_weight.unsqueeze(-1)
        repellency_term += (diff_vec * trunc_weight).sum(dim=1)

    if (
        repulsion_mode in ["memory_bank_only", "samples_and_memory_bank"]
        and repulsive_items is not None
    ):
        repulsive_items_flattened = repulsive_items.flatten(1)

        diff_vec = samples_flattened.unsqueeze(1) - repulsive_items_flattened.unsqueeze(
            0
        )  # diff_vec has size [batch, num_protection_images, dimensions]
        weight = (diff_vec**2).sum(dim=2).sqrt()
        trunc_weight = relu(shield_radius / weight - 1).unsqueeze(-1)
        repellency_term += (diff_vec * trunc_weight).sum(dim=1)

    # Convert to repellency in [-1,1] image space
    repellency_term = 2 * repellency_term.reshape(x_hat.shape)

    return denoised.detach(), repellency_term

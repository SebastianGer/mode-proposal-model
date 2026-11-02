# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

# Changes made:
# - Added separate functions timestep_schedule, time_step_discretization, euler_step to make it easier to reuse parts of the code. 
#   The functionality of these functions is essentially the same as in the original code. 
# - euler_step allows for asymmetric time differences, as used in Analog Bits, and noisy conditioning (CADS). 

import numpy as np
import torch

from firespreadscenarios.models.sampling.utils import noisy_conditioning


def timestep_schedule(sigma_max=80, sigma_min=0.002, num_steps=10, rho=7):
    step_indices = torch.arange(num_steps)
    return (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho


def time_step_discretization(num_steps, sigma_max, sigma_min, rho, device):
    if num_steps > 1:
        t_steps = timestep_schedule(sigma_max=sigma_max, sigma_min=sigma_min, num_steps=num_steps, rho=rho).to(device)
        t_steps = torch.cat([(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        t_step_pairs = zip(t_steps[:-1], t_steps[1:])
    else:
        assert num_steps == 1
        t_step0 = torch.tensor([sigma_max], device=device)
        t_step1 = torch.zeros_like(t_step0)
        t_steps = (t_step0, t_step1)
        t_step_pairs = [t_steps]

    return t_steps, t_step_pairs


def euler_step(x_cur, t_cur, net, S_churn, num_steps, S_min, S_max, S_noise, sigma_min, asymmetric_time_difference, conditioning, conditioning_noise_factor):
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = torch.as_tensor(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
    
    # Euler step preparations.
    t_async = max(torch.full_like(t_hat, fill_value=sigma_min), t_hat - asymmetric_time_difference)
    denoised = net(x_hat, t_async, noisy_conditioning(t_hat, conditioning, conditioning_noise_factor)).to(torch.float64)
        
    d_cur = (x_hat - denoised) / t_hat

    return x_hat, t_hat, d_cur, denoised


#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


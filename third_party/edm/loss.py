# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# Changes made:
# - Removed references to torch_utils persistence decorators
# - Modified EDM loss in various ways (see respective comment). Modified class is EDMLoss2.

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import random
from .generate import noisy_conditioning
#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

class EDMLoss2:
    # Changes made compared to EDMLoss:
    # - Added option to add noise to the conditioning information (CADS)
    # - Added option to use step unrolling
    # - Added option to provide a fixed sigma instead of sampling it
    # - Renamed labels to conditioning
    # - Changed return value to return the prediction and the noised target as well

    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,
                 conditioning_noise_factor:float=0.0,
                 conditioning_noise_prob:float=0.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.conditioning_noise_factor = conditioning_noise_factor
        self.conditioning_noise_prob = conditioning_noise_prob

    def __call__(self, net, images, conditioning=None, fixed_sigma=None, use_step_unrolling:bool=False):
        # Output target is what the prediction should look like. 
        # Input target is the informatoin we put into the model together with the noise
        input_target = images        
        output_target = images

        if use_step_unrolling:

            if self.conditioning_noise_factor > 0:
                raise RuntimeError("WARNING: Noise schedule for conditioning is requested but not implemented for step unrolling!")

            # Make the first prediction step from pure noise and conditioning information
            sigma = torch.tensor([net.sigma_max]*images.shape[0], device=images.device, dtype=images.dtype)[:, None, None, None]
            n = torch.randn_like(output_target) * sigma

            with torch.no_grad():
                D_yn_first_pred = net(n, sigma, conditioning)
                
            # Add noise to the first prediction, instead of the ground truth, since this is closer to what happens in sampling.
            input_target = D_yn_first_pred

        if fixed_sigma is not None:
            sigma = fixed_sigma
        else:
            rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp() # shape [B,1,1,1], log-normal distribution

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2 # shape [B,1,1,1]

        # Forward noise process
        n = torch.randn_like(output_target) * sigma
        x_t = input_target + n

        # CADS: Add noise to the conditioning information
        if conditioning is not None and random.random() < self.conditioning_noise_prob:
            conditioning = noisy_conditioning(sigma, conditioning, conditioning_noise_factor=self.conditioning_noise_factor)

        D_yn = net(x_t, sigma, conditioning)
        loss = weight * ((D_yn - output_target) ** 2)

        return loss.mean(), D_yn, output_target+n
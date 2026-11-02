import random

import torch


def noisy_conditioning(t, conditioning, conditioning_noise_factor):
    if conditioning_noise_factor == 0:
        return conditioning
    # Add noise to the conditioning, following the sampling noise schedule for x_t, but multiplied by a factor.
    # Using a piece-wise linear function, like in the CADS paper should not change much, because low noise is
    # basically equivalent to no noise for binary conditioning, and very high noise is similar to completely random conditioning.
    #
    # Cityscapes data is normalized to mean 0, std 1, so the formula below keeps the std the same, while obscuring the original signal with noise.
    return (
        conditioning
        + conditioning_noise_factor * t * torch.randn_like(conditioning.float())
    ) / (t**2 + 1).sqrt()


def normalize_max_per_image(t):
    # Expects a tensor of shape [B,C,H,W], computes the maximum for each of the B images
    # and normalizes each image to have maximum 1.
    assert len(t.shape) == 4
    flattened = t.view(*t.shape[:2], -1)
    image_wise_max = flattened.max(dim=-1).values
    # Ensure that we don't invert all-negative images
    image_wise_max[image_wise_max <= 0] = 1
    return t / image_wise_max[:, :, None, None]


def extract_groups(adj_matrix):

    num_nodes = adj_matrix.shape[0]

    groups = []
    visited = [False] * num_nodes

    for i in range(num_nodes):
        if not visited[i]:
            group = []
            for j in range(i, num_nodes):
                if adj_matrix[i, j] and not visited[j]:
                    group.append(j)
                    visited[j] = True
            groups.append(group)

    return groups


def extract_unique_and_replaceable_ids(groups, keep_random_unique=False):
    # Replace all but one sample in each group. If keep_random_unique is True, the unique sample is chosen randomly.
    # Otherwise, the first sample in the group is kept as unique.
    unique_ids = []
    replaceable_ids = []
    for g in groups:
        if len(g) > 0:
            if keep_random_unique:
                random.shuffle(g)
            unique_ids.append(g[0])
            replaceable_ids.extend(g[1:])  # All but the first one are replaceable
    return unique_ids, replaceable_ids

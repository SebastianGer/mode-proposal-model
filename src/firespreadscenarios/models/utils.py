import torch
import torchvision.utils as vutils


def from_unit_interval_to_bitscale(x, bit_scale):
    # Transforms x from [0,1] to [-bit_scale, bit_scale]
    return ((x * 2 - 1) * bit_scale).clamp(-bit_scale, bit_scale)


def from_unit_interval_to_pm1(x):
    # Transforms x from [0,1] to [-1, 1], used for target segmentation maps
    return ((x * 2 - 1)).clamp(-1, 1)


def from_pm1_to_unit_interval(x):
    # Transforms x from [-1, 1] to [0,1], used for predicted segmentation maps
    return ((x + 1) / 2).clamp(0, 1)


def from_bitscale_to_unit_interval(x, bit_scale, clamp=True):
    # Transforms x from [-bit_scale, bit_scale] to [0,1]
    rescaled = (x / bit_scale + 1) / 2
    if clamp:
        rescaled = rescaled.clamp(0, 1)
    return rescaled


def to_unit_interval(x):
    return (x - x.min()) / (x.max() - x.min())


def visualize_batch_sampling_progression(tensor_list):

    N = len(tensor_list)  # Number of tensors in the list
    B, C, H, W = tensor_list[0].size()  # Assuming all tensors have the same size

    # Stack tensors along a new dimension (batch axis) to make transposing easier
    stacked_images = torch.stack(tensor_list)  # Shape: [N, B, C, H, W]

    # Transpose to swap the first two dimensions (N and B -> B and N)
    transposed_images = stacked_images.permute(1, 0, 2, 3, 4)  # Shape: [B, N, C, H, W]

    # Reshape back to a large batch for make_grid, i.e., [B * N, C, H, W]
    all_images = transposed_images.reshape(B * N, C, H, W)

    # Create a grid of BxN images (B rows, N images per row)
    grid = vutils.make_grid(
        all_images, nrow=N, padding=2, normalize=True, pad_value=0.25
    )[0]

    return grid

import torch
from scipy.optimize import linear_sum_assignment


def compute_iou_matrix(outputs, targets):
    # Flatten masks to (N, H*W)
    masks1 = outputs.flatten(start_dim=1).float()
    masks2 = targets.flatten(start_dim=1).float()

    # Calculate intersection: (N, H*W) @ (H*W, M) -> (N, M)
    intersection = torch.matmul(masks1, masks2.t())

    # Calculate areas (sums of pixels)
    area1 = masks1.sum(dim=1).view(-1, 1)  # (N, 1)
    area2 = masks2.sum(dim=1).view(1, -1)  # (1, M)

    # Union = Area1 + Area2 - Intersection
    union = area1 + area2 - intersection

    # Avoid division by zero
    iou = intersection / (union + 1e-6)

    # If both prediction and target are empty (union == 0), treat as perfect match
    both_empty_mask = union == 0
    if both_empty_mask.any():
        iou = iou.clone()
        iou[both_empty_mask] = 1.0
    return iou


def match_targets(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    return row_ind, col_ind

def compute_hm_iou_batch(true_imgs: torch.tensor, gen_imgs: torch.tensor):
    assert true_imgs.shape[0] == gen_imgs.shape[0]
    batch_hm_iou = []
    for b in range(true_imgs.shape[0]):
        hm_iou = compute_hm_iou(true_imgs[b], gen_imgs[b])
        batch_hm_iou.append(hm_iou)
    mean_hm_iou = torch.stack(batch_hm_iou).mean()
    return mean_hm_iou
    
def compute_hm_iou(
    true_imgs: torch.tensor, gen_imgs: torch.tensor, return_ious: bool = False
):
    iou_matrix = compute_iou_matrix(true_imgs, gen_imgs)
    row_ind, col_ind = match_targets(1 - iou_matrix)  # need cost matrix, so 1-IoU
    ious = iou_matrix[row_ind, col_ind]
    if return_ious:
        return ious.mean(), ious
    else:
        return ious.mean()

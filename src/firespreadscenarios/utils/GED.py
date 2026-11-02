import torch

from firespreadscenarios.utils.hm_iou import compute_iou_matrix


def compute_ged(true_imgs: torch.tensor, gen_imgs: torch.tensor):
    # Computes the generalized energy distance between two probability distributions, based on the Jaccard index as the distance function d.

    # Threshold the images
    true_imgs = (true_imgs > 0.5).float().cuda()
    gen_imgs = (gen_imgs > 0.5).float().cuda()

    def compute_mean_distance(imgs1, imgs2):
        # Distance is 1-IoU(x,y)
        iou_matrix = compute_iou_matrix(imgs1, imgs2)
        ious = iou_matrix.flatten()
        return 1 - ious.nan_to_num(1.0).mean()

    def compute_mean_distance_within(imgs):
        # Use to compute the expected distance within the same set of images
        # Distance is 1-IoU(x,y)
        iou_matrix = compute_iou_matrix(imgs, imgs)
        ious = iou_matrix.flatten()
        return 1 - ious.nan_to_num(1.0).mean()

    dist_true_gen = compute_mean_distance(true_imgs, gen_imgs)
    dist_true_true = compute_mean_distance_within(true_imgs)
    dist_gen_gen = compute_mean_distance_within(gen_imgs)

    return {
        "dist_true_gen": dist_true_gen,
        "dist_true_true": dist_true_true,
        "dist_gen_gen": dist_gen_gen,
        "ged": 2 * dist_true_gen - dist_true_true - dist_gen_gen,
    }

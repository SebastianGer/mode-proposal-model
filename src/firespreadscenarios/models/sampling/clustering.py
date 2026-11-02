import warnings
from typing import Literal

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_cdt
from sklearn_extra.cluster import KMedoids


def prepare_data_for_chamfer(y):
    B = y.shape[0]
    y_npy = y.numpy().astype(np.uint8) * 255
    y_edges = np.zeros_like(y_npy)
    y_dist_transform = np.zeros_like(y_npy)
    for b in range(B):
        y_edges[b] = cv2.Canny(y_npy[b, 0], 30, 200)
        y_dist_transform[b] = distance_transform_cdt(255 - y_npy[b])
    return y_edges, y_dist_transform


def create_pairwise_chamfer_dist(y_edges, y_dist_transform):
    def pairwise_chamfer_dist(i, j):
        return y_dist_transform[int(i[0])][y_edges[int(j[0])] > 0].sum()

    return pairwise_chamfer_dist


def chamfer_dist(y_edges, y_dist_transform):
    B = y_edges.shape[0]

    dist_mat = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            chamfer_dist_ij = y_dist_transform[i][y_edges[j] > 0].sum()
            chamfer_dist_ji = y_dist_transform[j][y_edges[i] > 0].sum()
            dist_mat[i, j] = (chamfer_dist_ij + chamfer_dist_ji) / 2.0
    return dist_mat


def compute_chamfer_dist_mat(y):
    # Compute pairwise chamfer distances
    # 1. detect edges with canny filter
    # 2. compute distance transform, that indicates distance to edge for each pixel
    # 3. compute chamfer distance based on edge positions and distance transform
    y_edges, y_dist_transform = prepare_data_for_chamfer(y)
    return chamfer_dist(y_edges, y_dist_transform)


def clustering(
    X_img,
    n_clusters=8,
    distance_metric: Literal["chamfer", "L2"] = "chamfer",
    existing_samples=None,
):
    unique_vals = X_img.unique()
    if len(unique_vals) == 1 and 0 in unique_vals:
        # Empty images, can't cluster. Return arbitrary indices as medoids.
        return torch.arange(n_clusters, dtype=int)
    if unique_vals.shape[0] != 2 or 0 not in unique_vals or 1 not in unique_vals:
        raise ValueError(
            f"Input image must be binary, i.e. contain only two unique values (0 and 1), but contains: {unique_vals}."
        )

    all_samples = X_img
    n_clusters_new = n_clusters

    # If existing samples are provided, include them in clustering, but increase the number of clusters accordingly.
    # We want to find new clusters, so any clusters belonging to the existing samples will be ignored.
    if existing_samples is not None:
        all_samples = torch.cat((X_img, existing_samples.unsqueeze(1)), dim=0)
        n_clusters_new = n_clusters + existing_samples.shape[0]

    if distance_metric == "chamfer":
        dist_mat = compute_chamfer_dist_mat(all_samples.cpu())
    elif distance_metric == "L2":
        dist_mat_input = all_samples.float().flatten(1, 3)
        dist_mat = torch.cdist(dist_mat_input, dist_mat_input).cpu().numpy()
    else:
        raise ValueError(
            f"Unsupported distance metric: {distance_metric}. Supported metrics are 'chamfer' and 'L2'."
        )

    # Sometimes medoids are very unluckily chosen, but mostly they result in similar cost (inertia). Precomputed distances perform better than computing them as part of the clustering.
    best_clustering = None
    for _ in range(3):
        # Catch and ignore warnings about empty clusters.
        # In case of empty clusters, it seems like the empty cluster will be represented by a random other element, which is acceptable for our use case.
        with warnings.catch_warnings(record=False) as w:
            warnings.simplefilter("ignore")

            kmedoids = KMedoids(
                n_clusters=n_clusters_new,
                method="alternate",
                init="k-medoids++",
                metric="precomputed",
            ).fit(dist_mat)
            if best_clustering is None or best_clustering.inertia_ < kmedoids.inertia_:
                best_clustering = kmedoids

    if existing_samples is not None:
        # Get indices of the medoids that correspond to the existing samples.
        existing_samples_medoids_ids = best_clustering.labels_[
            -existing_samples.shape[0] :
        ]
        new_medoids_mask = np.ones(n_clusters_new, dtype=bool)
        new_medoids_mask[existing_samples_medoids_ids] = False

        # Get all indices wrt X_img that are not in the existing samples medoids.
        # Only take the first n_clusters, since it could happen that multiple existing samples ended up in the same cluster,
        # but we only want to return n_clusters new samples in the end.
        new_medoids = best_clustering.medoid_indices_[new_medoids_mask][:n_clusters]
    else:
        new_medoids = best_clustering.medoid_indices_

    return new_medoids

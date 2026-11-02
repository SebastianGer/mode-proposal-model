"""
Offline evaluation of diffusion model mode-probability estimation via HDBSCAN.

After running BaseDiffusionEval with `write_samples_dir` set, call this script
to cluster the binarized samples per conditioning with HDBSCAN (distance = 1 - IoU),
Hungarian-match cluster centroids to ground-truth modes, and compute global and
median Brier Skill Score together with a reliability diagram.

Usage::

    python -m firespreadscenarios.evaluate_diffusion_brier \\
        --dataset mmfire \\
        --data_dir /path/to/simfire \\
        --split val \\
        --samples_dir /path/to/saved_samples \\
        --eps 0.3 --min_samples 5

The --batch_size must match the value used during sample generation so that the
(batch_idx, item_i) filename keys map to the same data-loader items.
"""

import argparse
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Callable, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import torch
import wandb
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN
from tqdm import tqdm

from firespreadscenarios.models.BaseModePredictionNetwork import (
    GlobalBrierSkillScore,
    MedianMetric,
)
from firespreadscenarios.utils.hm_iou import compute_hm_iou, compute_iou_matrix

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── GT probability helpers ────────────────────────────────────────────────────

def _gt_probs_empirical(all_targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Derive GT probabilities as empirical frequency over K annotator masks.
    Correct for LIDC (4 radiologists with equal weight) because all_targets
    contains one entry per annotator, with duplicates when they agree.
    """
    unique_modes, inverse = torch.unique(all_targets, dim=0, return_inverse=True)
    K = all_targets.shape[0]
    probs = torch.zeros(unique_modes.shape[0], dtype=torch.float32)
    for i in range(unique_modes.shape[0]):
        probs[i] = (inverse == i).sum().float() / K
    return unique_modes, probs


def _gt_probs_fixed(
    all_targets: torch.Tensor,
    distribution: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GT probabilities from a precomputed distribution over K modes in all_targets.
    The i-th entry of `distribution` is the probability of all_targets[i].
    Used for mmfire/Simfire where all_targets already contains the K unique modes
    in a fixed order and the actual distribution (uniform or asymmetric) is a
    dataset-level constant rather than an empirical frequency.

    If some modes are identical (degenerate case), their probabilities are summed.
    """
    unique_modes, inverse = torch.unique(all_targets, dim=0, return_inverse=True)
    dist = distribution.to(dtype=torch.float32, device="cpu")
    probs = torch.zeros(unique_modes.shape[0], dtype=torch.float32)
    for i in range(unique_modes.shape[0]):
        orig_positions = (inverse == i).nonzero(as_tuple=True)[0]
        probs[i] = dist[orig_positions].sum()
    return unique_modes.float(), probs


# ── HDBSCAN clustering and matching ───────────────────────────────────────────

def _hdbscan_labels(samples: torch.Tensor, eps: float, min_samples: int):
    """
    HDBSCAN cluster label per sample, using 1 - IoU as the distance metric.
    -1 denotes noise. Exposed separately from `_cluster_and_match` so callers
    can inspect noise samples even when no valid (non-noise) cluster is found.

    `min_samples` maps to HDBSCAN's `min_cluster_size` and `eps` to its
    `cluster_selection_epsilon` (clusters within this distance of each other
    are merged, giving DBSCAN-like behaviour at that threshold) — this avoids
    having to hand-tune a single global density radius across items whose
    mask size/shape vary a lot, which plain DBSCAN was sensitive to.
    """
    print("Computing IoU matrix...")
    samples_dev = samples.float().to(_DEVICE)
    iou = compute_iou_matrix(samples_dev, samples_dev)
    print("Computing IoU matrix finished.")
    dist = (1.0 - iou).clamp(0.0, 1.0).cpu().numpy()
    return HDBSCAN(
        min_cluster_size=min_samples,
        cluster_selection_epsilon=eps,  
        metric="precomputed",
        cluster_selection_method="eom"
    ).fit_predict(dist)


def _aggregate_best_case(
    centroids: torch.Tensor,
    cluster_freqs: torch.Tensor,
    unique_gt_modes: torch.Tensor,
) -> torch.Tensor:
    """
    Many-to-one "best case" assignment: each cluster is independently assigned
    to its single best-IoU-matching GT mode (no one-to-one constraint), and
    cluster frequencies landing on the same GT mode are summed.

    This is the "_best_case" counterpart to the one-to-one Hungarian matching
    in `_cluster_and_match`: it tells you how well-calibrated the underlying
    generative distribution is if HDBSCAN's intentional oversegmentation of a
    single true mode into several clusters didn't cost anything (each piece
    is correctly re-attributed to the one true mode it came from). It is
    unrealistic as a deployable estimator, since it requires knowing which GT
    mode every sample truly belongs to -- ground truth that isn't available at
    inference time -- but it isolates clustering-granularity penalties from
    genuine model miscalibration.

    Returns estimated_probs [M] (M = number of GT modes), zeros if there are
    no clusters or no GT modes.
    """
    M = unique_gt_modes.shape[0]
    C = centroids.shape[0]
    best_case_est = torch.zeros(M, dtype=torch.float32)
    if C == 0 or M == 0:
        return best_case_est
    iou = compute_iou_matrix(centroids, unique_gt_modes.float())  # [C, M]
    best_match = iou.argmax(dim=1)  # [C]
    best_case_est.scatter_add_(0, best_match, cluster_freqs)
    return best_case_est


def _cluster_and_match(
    samples: torch.Tensor,
    labels: np.ndarray,
    unique_gt_modes: torch.Tensor,
    gt_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Hungarian-match HDBSCAN cluster centroids to GT modes.

    `labels` are precomputed by `_hdbscan_labels` (1 - IoU distance) — passed
    in rather than recomputed here so callers that also need the raw labels
    for clustering diagnostics (e.g. `evaluate_mode_probabilities`) only pay
    for HDBSCAN once per item.

    Every GT mode and every non-noise cluster is preserved in the output:
    - A cluster with no matching GT mode (more clusters than GT modes) gets
      target_prob = 0, as before.
    - A GT mode with no matching cluster (more GT modes than clusters) gets
      an all-zero "empty" centroid and estimated frequency 0, so it still
      shows up in evaluation/plots instead of being silently dropped.

    Also returns a many-to-one "best case" estimate (see `_aggregate_best_case`)
    over the M GT modes, decoupled from the one-to-one matching above.

    Returns (estimated_probs [L], target_probs [L], centroids [L,H,W],
    matched_gt [L,H,W], best_case_est [M]) where L = max(C, M), C = number of
    non-noise clusters, M = number of GT modes. Frequencies are normalised
    over non-noise samples. Returns None if there are no samples and no GT
    modes to report.
    """
    N = samples.shape[0]
    if N == 0:
        return None

    H, W = samples.shape[-2:]
    M = unique_gt_modes.shape[0]

    unique_labels = sorted(set(labels.tolist()) - {-1})
    C = len(unique_labels)

    if C == 0 and M == 0:
        return None

    n_non_noise = int((labels != -1).sum())
    cluster_freqs = torch.tensor(
        [(labels == lbl).sum() / max(n_non_noise, 1) for lbl in unique_labels],
        dtype=torch.float32,
    )

    if C > 0:
        centroids = torch.stack([
            (samples[labels == lbl].float().mean(0) >= 0.5).float()
            for lbl in unique_labels
        ])  # [C, H, W]
    else:
        centroids = samples.new_zeros((0, H, W))

    best_case_est = _aggregate_best_case(centroids, cluster_freqs, unique_gt_modes)

    if C > 0 and M > 0:
        # Minimise 1 − IoU ↔ maximise IoU; rows = clusters, cols = GT modes
        cost = (1.0 - compute_iou_matrix(centroids, unique_gt_modes.float())).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        row_ind, col_ind = [], []

    target_probs = torch.zeros(C, dtype=torch.float32)
    matched_gt = torch.zeros_like(centroids)
    for c_idx, m_idx in zip(row_ind, col_ind):
        target_probs[c_idx] = gt_probs[m_idx]
        matched_gt[c_idx] = unique_gt_modes[m_idx]

    # GT modes left unmatched (more GT modes than clusters) get an empty
    # centroid with estimated frequency 0, instead of being dropped.
    unmatched_gt_idx = sorted(set(range(M)) - set(col_ind))
    if unmatched_gt_idx:
        cluster_freqs = torch.cat(
            [cluster_freqs, torch.zeros(len(unmatched_gt_idx), dtype=torch.float32)]
        )
        target_probs = torch.cat([target_probs, gt_probs[unmatched_gt_idx]])
        centroids = torch.cat(
            [centroids, torch.zeros((len(unmatched_gt_idx), H, W), dtype=centroids.dtype)]
        )
        matched_gt = torch.cat(
            [matched_gt, unique_gt_modes[unmatched_gt_idx].float()]
        )

    return cluster_freqs, target_probs, centroids, matched_gt, best_case_est


# ── Reliability diagram ───────────────────────────────────────────────────────

def _plot_reliability_diagram(groups: dict, output_path: str):
    tgt_vals = sorted(v for v in groups if v > 0)
    if not tgt_vals:
        return

    fs = 6
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    y_lim = (min(tgt_vals) * 0.5, max(tgt_vals) * 2)

    for i, tgt_val in enumerate(tgt_vals):
        ests = np.atleast_1d(np.array(groups[tgt_val]))
        jitter = np.random.uniform(-0.2, 0.2, size=len(ests))
        ax.scatter(
            i + jitter, ests, alpha=0.4, s=4, color="steelblue", zorder=2,
            label="Estimates" if i == 0 else None,
        )
        ax.hlines(
            ests.mean(), i - 0.2, i + 0.2, colors="red", linewidths=1.5, zorder=3,
            label="Mean estimate" if i == 0 else None,
        )
        ax.hlines(
            tgt_val, i - 0.35, i + 0.35, colors="black", linewidths=1.0,
            linestyles="dashed", zorder=1,
            label="True probability" if i == 0 else None,
        )

    ax.set_yscale("log")
    ax.set_ylim(y_lim)
    ax.set_xlim(-0.5, len(tgt_vals) - 0.5)
    ax.set_xticks(list(range(len(tgt_vals))))
    ax.set_xticklabels([f"{v:.4f}" for v in tgt_vals], rotation=90, fontsize=fs)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.4f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_xlabel("True probability", fontsize=fs)
    ax.set_ylabel("Estimated probability", fontsize=fs)
    ax.set_title("Reliability diagram — HDBSCAN (diffusion)", fontsize=fs)
    ax.legend(fontsize=fs, loc="lower right")
    fig.savefig(output_path)
    plt.close(fig)


def _plot_debug_centroids(
    centroids: torch.Tensor,
    matched_gt: torch.Tensor,
    est_freqs: torch.Tensor,
    target_probs: torch.Tensor,
    item_label: str,
    output_path: str,
):
    """
    Sanity-check figure: each HDBSCAN cluster centroid next to its Hungarian-matched
    ground-truth mode (or "unmatched" if none was assigned).
    """
    C = centroids.shape[0]
    if C > 50:
        return None
    fig, axes = plt.subplots(C, 2, figsize=(4, 2 * C), squeeze=False)
    for c in range(C):
        axes[c, 0].imshow(centroids[c].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[c, 0].set_title(f"Centroid {c} (est={est_freqs[c]:.3f})", fontsize=7)
        axes[c, 0].axis("off")

        is_matched = target_probs[c] > 0
        axes[c, 1].imshow(matched_gt[c].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        gt_label = f"GT (tgt={target_probs[c]:.3f})" if is_matched else "unmatched"
        axes[c, 1].set_title(gt_label, fontsize=7)
        axes[c, 1].axis("off")

    fig.suptitle(item_label, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ── Core evaluation loop ──────────────────────────────────────────────────────

def evaluate_mode_probabilities(
    samples_dir: str,
    data_loader,
    unbatch_fn: Callable,
    get_gt_fn: Callable,
    eps: float = 0.3,
    min_samples: int = 5,
    require_unique_modes: bool = True,
) -> dict:
    """
    Compute Brier Skill Score and reliability diagram for a diffusion model
    whose samples were saved by BaseDiffusionEval (``write_samples_dir``).

    Args:
        samples_dir:  Directory of ``{batch_idx:05d}_{item_i:04d}_rep{j:04d}.npz``
                      files (converted from BaseDiffusionEval's ``.pt`` output via
                      ``firespreadscenarios.pt_to_npz``). The reliability diagram
                      is written to ``{samples_dir}/plots``.
        data_loader:  DataLoader (no shuffle) with the same batch_size and order
                      as during sample generation.
        unbatch_fn:   ``callable(batch) -> dict`` with at least ``"all_targets"``
                      [B, K, H, W].
        get_gt_fn:    ``callable(batch_dict, i) -> (unique_gt_modes, gt_probs)``
                      Returns the unique ground-truth modes [M, H, W] and their
                      probabilities [M] for batch item i.
        eps:          HDBSCAN cluster_selection_epsilon in 1-IoU space.
        min_samples:  HDBSCAN min_cluster_size.
        require_unique_modes: If True (default), skip items whose raw
                      ``all_targets[i]`` contains any duplicate GT masks (e.g.
                      two annotators who agree exactly, or a flipped class
                      that's absent from the image so the flip is a no-op).
                      Such items have fewer genuinely distinct modes than their
                      nominal K, which can make them a less meaningful test of
                      multimodal calibration. Affects every metric below, not
                      just the Brier scores.

    Also tracks, averaged over every item with loaded samples:
    - the number of non-noise HDBSCAN clusters found and the fraction of
      samples assigned to the noise cluster ("avg_n_clusters" / "avg_noise_rate"),
      the clustering-based analogue of the "Distinct modes" metric in
      SimfireDiffusionEval/CityscapesDiffusionEval (those datasets can count
      distinct modes directly against a known, enumerable mode set; here the
      set of modes isn't known up front, so HDBSCAN's cluster count stands in).
    - "avg_hm_iou": Hungarian-matched IoU between the (non-deduplicated)
      ground-truth masks and the raw generated samples, computed the same way
      as `compute_hm_iou(possible_modes, binarized_predictions)` in
      BaseDiffusionEval.compute_tvd / `compute_hm_iou_batch` in
      BaseModePredictionNetwork — i.e. directly on samples, independent of
      the HDBSCAN clustering used for the Brier score above.
    - "global_bss_best_case" / "median_bss_best_case": same Brier Skill Score
      formula, but computed on the many-to-one "best case" aggregation (see
      `_aggregate_best_case`) instead of the one-to-one Hungarian matching.
      The one-to-one metrics above charge a full penalty whenever HDBSCAN
      intentionally splits one true mode into several clusters (only one
      cluster can win the assignment; the rest count as spurious extra modes
      with target_prob=0). The best-case metrics instead re-attribute every
      cluster to its best-matching GT mode and sum the pieces, so deliberate
      oversegmentation no longer costs anything. This requires knowing the
      true GT-mode membership of each cluster, which is unavailable at
      inference time, so it is not a deployable estimator — it is an upper
      bound on the underlying generative distribution's calibration, with
      clustering-granularity penalties removed.
    - "worst_items" / "best_items": the 5 lowest/highest-scoring items by
      per-item best-case BSS, as (batch_idx, item_i, bss, bss_best_case)
      tuples, for manual inspection.
    """
    plots_dir = os.path.join(samples_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Built once instead of glob()-ing samples_dir per item: with samples
    # accumulated from multiple parallel eval runs (filenames carry a wandb
    # run id), a per-item glob rescans an ever-growing directory every time.
    rep_file_re = re.compile(r"^(\d{5})_(\d{4})_rep.*\.npz$")
    rep_files_by_item = defaultdict(list)
    for fname in os.listdir(samples_dir):
        m = rep_file_re.match(fname)
        if m:
            rep_files_by_item[(int(m.group(1)), int(m.group(2)))].append(
                os.path.join(samples_dir, fname)
            )

    global_bss_metric = GlobalBrierSkillScore()
    median_bss_metric = MedianMetric()
    global_bss_metric_best_case = GlobalBrierSkillScore()
    median_bss_metric_best_case = MedianMetric()
    reliability_groups: dict = {}
    # Per-item (batch_idx, item_i, brier_skill_score, brier_skill_score_best_case),
    # logged so the worst/best items can be inspected manually afterwards.
    per_item_scores = []
    n_items = 0
    n_skipped = 0
    n_skipped_duplicate_modes = 0
    n_debug_figs = 0
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    # Per-item clustering diagnostics, averaged over every item with loaded samples
    # (regardless of whether matching against GT modes later succeeds).
    sum_n_clusters = 0
    sum_noise_rate = 0.0
    sum_hm_iou = 0.0
    n_clustering_items = 0

    outer_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Batches", unit="batch")
    for batch_idx, batch in outer_bar:
        batch_dict = unbatch_fn(batch)
        B = batch_dict["all_targets"].shape[0]

        inner_bar = tqdm(range(B), desc=f"  batch {batch_idx} items", unit="item", leave=False)
        for i in inner_bar:
            inner_bar.set_postfix(items=n_items, skipped=n_skipped, dup_modes=n_skipped_duplicate_modes)
            all_targets_i = batch_dict["all_targets"][i]
            n_unique = torch.unique(all_targets_i, dim=0).shape[0]
            has_unique_modes = n_unique == all_targets_i.shape[0]
            if require_unique_modes and not has_unique_modes:
                n_skipped_duplicate_modes += 1
                continue

            rep_files = sorted(rep_files_by_item.get((batch_idx, i), []))
            if not rep_files:
                n_skipped += 1
                continue

            samples = torch.cat(
                [torch.from_numpy(np.load(f)["arr_0"]) for f in rep_files],
                dim=0,
            ).float()

            labels = _hdbscan_labels(samples, eps, min_samples)
            sum_n_clusters += len(set(labels.tolist()) - {-1})
            sum_noise_rate += float((labels == -1).sum()) / len(labels)
            # Same call as `possible_modes`/`binarized_predictions` in
            # BaseDiffusionEval.compute_tvd: HM IoU against all (possibly
            # duplicated) GT masks, not the deduplicated/clustered ones.
            sum_hm_iou += compute_hm_iou(batch_dict["all_targets"][i], samples).item()
            n_clustering_items += 1

            unique_gt_modes, gt_probs = get_gt_fn(batch_dict, i)

            result = _cluster_and_match(samples, labels, unique_gt_modes, gt_probs)
            if result is None:
                n_skipped += 1
                continue

            est, tgt, centroids, matched_gt, best_case_est = result

            # Matches GlobalBrierSkillScore.update's own bs/bs_ref formula, so the
            # per-item score fed to MedianMetric is consistent with the global one.
            bs = torch.sum((est - tgt) ** 2)
            ref_preds = torch.full_like(est, 1.0 / len(est))
            bs_ref = torch.sum((ref_preds - tgt) ** 2)
            brier_skill_score = 1.0 - bs / (bs_ref + 1e-8)

            global_bss_metric.update(est, tgt)
            median_bss_metric.update(brier_skill_score)

            # Best-case: many-to-one aggregation over the M GT modes directly
            # (see `_aggregate_best_case`), scored the same way but without the
            # one-to-one matching penalty for deliberate oversegmentation.
            bs_best_case = torch.sum((best_case_est - gt_probs) ** 2)
            ref_preds_best_case = torch.full_like(best_case_est, 1.0 / len(best_case_est))
            bs_ref_best_case = torch.sum((ref_preds_best_case - gt_probs) ** 2)
            brier_skill_score_best_case = 1.0 - bs_best_case / (bs_ref_best_case + 1e-8)

            global_bss_metric_best_case.update(best_case_est, gt_probs)
            median_bss_metric_best_case.update(brier_skill_score_best_case)
            per_item_scores.append((
                batch_idx, i,
                brier_skill_score.item(), brier_skill_score_best_case.item(),
            ))
            n_items += 1

            if n_debug_figs < 5:
                debug_path = os.path.join(
                    plots_dir,
                    f"debug_centroids_{run_id}_eps{eps}_min{min_samples}"
                    f"_b{batch_idx:05d}_i{i:04d}.pdf",
                )
                _plot_debug_centroids(
                    centroids, matched_gt, est, tgt,
                    f"batch {batch_idx} item {i} (run {run_id})",
                    debug_path,
                )
                n_debug_figs += 1

            # Reliability diagram: only matched clusters (tgt > 0) from items with
            # unique GT modes, regardless of require_unique_modes -- duplicate modes
            # distort the per-target grouping by mixing probabilities for what are
            # really separate GT masks. Otherwise consistent with
            # BaseModePredictionNetwork.on_validation_epoch_end
            if has_unique_modes:
                for e_val, t_val in zip(est.tolist(), tgt.tolist()):
                    if t_val > 0:
                        t_rounded = round(t_val, 4)
                        reliability_groups.setdefault(t_rounded, []).append(e_val)

    global_bss = global_bss_metric.compute().item()
    median_bss = median_bss_metric.compute().item()
    global_bss_best_case = global_bss_metric_best_case.compute().item()
    median_bss_best_case = median_bss_metric_best_case.compute().item()
    avg_n_clusters = sum_n_clusters / n_clustering_items if n_clustering_items else float("nan")
    avg_noise_rate = sum_noise_rate / n_clustering_items if n_clustering_items else float("nan")
    avg_hm_iou = sum_hm_iou / n_clustering_items if n_clustering_items else float("nan")

    print(
        f"Items evaluated      : {n_items}  |  skipped : {n_skipped}"
        f"  |  skipped (duplicate modes) : {n_skipped_duplicate_modes}"
    )
    print(f"Global BSS           : {global_bss:.4f}")
    print(f"Median BSS           : {median_bss:.4f}")
    print(f"Global BSS best case : {global_bss_best_case:.4f}")
    print(f"Median BSS best case : {median_bss_best_case:.4f}")
    print(f"Avg # clusters       : {avg_n_clusters:.2f}")
    print(f"Avg noise rate       : {avg_noise_rate:.4f}")
    print(f"Avg HM IoU           : {avg_hm_iou:.4f}")

    # Ranked by best-case BSS (i.e. with the one-to-one matching penalty for
    # deliberate oversegmentation removed), since that's the score that isolates
    # genuine model miscalibration from clustering artifacts.
    ranked = sorted(per_item_scores, key=lambda r: r[3])
    worst_items = ranked[:5]
    best_items = ranked[-5:][::-1]

    print("Worst 5 items (by best-case BSS):")
    for batch_idx, i, bss, bss_best_case in worst_items:
        print(f"  batch {batch_idx:5d} item {i:4d}  |  BSS={bss:.4f}  BSS_best_case={bss_best_case:.4f}")
    print("Best 5 items (by best-case BSS):")
    for batch_idx, i, bss, bss_best_case in best_items:
        print(f"  batch {batch_idx:5d} item {i:4d}  |  BSS={bss:.4f}  BSS_best_case={bss_best_case:.4f}")

    diagram_path = os.path.join(
        plots_dir, f"reliability_eps{eps}_min{min_samples}.pdf"
    )
    _plot_reliability_diagram(reliability_groups, diagram_path)
    print(f"Reliability diagram → {diagram_path}")

    if wandb.run is not None:
        wandb.log({
            "global_bss": global_bss,
            "median_bss": median_bss,
            "global_bss_best_case": global_bss_best_case,
            "median_bss_best_case": median_bss_best_case,
            "n_items": n_items,
            "n_skipped": n_skipped,
            "n_skipped_duplicate_modes": n_skipped_duplicate_modes,
            "avg_n_clusters": avg_n_clusters,
            "avg_noise_rate": avg_noise_rate,
            "avg_hm_iou": avg_hm_iou,
            "worst_items": wandb.Table(
                columns=["batch_idx", "item_idx", "bss", "bss_best_case"],
                data=[list(row) for row in worst_items],
            ),
            "best_items": wandb.Table(
                columns=["batch_idx", "item_idx", "bss", "bss_best_case"],
                data=[list(row) for row in best_items],
            ),
        })

    return {
        "global_bss": global_bss,
        "median_bss": median_bss,
        "global_bss_best_case": global_bss_best_case,
        "median_bss_best_case": median_bss_best_case,
        "n_items": n_items,
        "n_skipped": n_skipped,
        "n_skipped_duplicate_modes": n_skipped_duplicate_modes,
        "worst_items": worst_items,
        "best_items": best_items,
        "avg_n_clusters": avg_n_clusters,
        "avg_noise_rate": avg_noise_rate,
        "avg_hm_iou": avg_hm_iou,
    }


# ── Dataset builders ──────────────────────────────────────────────────────────

def _build_mmfire(args):
    from firespreadscenarios.dataloader.SimfireDataModule import SimfireDataModule
    from firespreadscenarios.dataloader.SimfireDataset import SimfireDataset

    dm = SimfireDataModule(
        root_dir=args.data_dir,
        distribution_mode="asymmetric",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()
    dl = {"train": dm.train_dataloader, "val": dm.val_dataloader, "test": dm.test_dataloader}[
        args.split
    ]()

    # Instantiate one sample just to read class_distribution — no data loaded
    ref_ds = SimfireDataset(
        x_path=os.path.join(args.data_dir, "X.npy"),
        y_path=os.path.join(args.data_dir, "Y.npy"),
        distribution_mode="asymmetric",
        indices=slice(0, 1),
    )
    class_distribution = ref_ds.class_distribution  # [K]

    def unbatch(batch):
        image, mask_dict = batch
        return {"all_targets": mask_dict["all_targets"].float()}

    def get_gt(batch_dict, i):
        return _gt_probs_fixed(batch_dict["all_targets"][i], class_distribution)

    return dl, unbatch, get_gt


def _build_lidc(args):
    from firespreadscenarios.dataloader.LIDCDataModule import LIDCDataModule

    dm = LIDCDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()
    dl = {"train": dm.train_dataloader, "val": dm.val_dataloader, "test": dm.test_dataloader}[
        args.split
    ]()

    def unbatch(batch):
        image, targets = batch[:2]
        return {"all_targets": targets.float()}

    def get_gt(batch_dict, i):
        # LIDC: 4 radiologist masks, empirical frequency = fraction who agree on a mode
        return _gt_probs_empirical(batch_dict["all_targets"][i])

    return dl, unbatch, get_gt


def _build_cityscapes(args):
    from firespreadscenarios.dataloader.CityscapesDataModule import CityscapesDataModule

    dm = CityscapesDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        binarization_mode="four",
        quality_mode="fine",
        target_type="semantic",
    )
    dm.setup(stage=None)
    dl = {"train": dm.train_dataloader, "val": dm.val_dataloader, "test": dm.test_dataloader}[
        args.split
    ]()

    # Joint distribution over all 16 binary combinations of the 4 flipping classes.
    # Precomputed via CityscapesEval.get_probability_per_mode; see CityscapesModePredictionNetwork.
    gt_prior = torch.tensor([
        8.9063e-03, 1.6922e-01, 2.6719e-02, 5.0766e-01,
        2.9688e-03, 5.6406e-02, 8.9063e-03, 1.6922e-01,
        4.6875e-04, 8.9062e-03, 1.4063e-03, 2.6719e-02,
        1.5625e-04, 2.9687e-03, 4.6875e-04, 8.9062e-03,
    ])

    def unbatch(batch):
        image, targets, _ = batch
        return {"all_targets": targets["all_targets"].float()}

    def get_gt(batch_dict, i):
        # all_targets is ordered so position j corresponds to gt_prior[j]
        return _gt_probs_fixed(batch_dict["all_targets"][i], gt_prior)

    return dl, unbatch, get_gt


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion model mode-probability estimation via HDBSCAN."
    )
    parser.add_argument(
        "--dataset", choices=["mmfire", "cityscapes", "lidc"], required=True,
        help="Dataset used during sample generation.",
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="val",
        help="Data split whose samples were generated.",
    )
    parser.add_argument(
        "--data_dir", "--data-dir", required=True,
        help="Dataset root directory (same value as data_dir/root_dir in cfgs/<dataset>/dataset.yaml).",
    )
    parser.add_argument(
        "--samples_dir", required=True,
        help="Directory of samples saved by BaseDiffusionEval (write_samples_dir).",
    )
    parser.add_argument(
        "--eps", type=float, default=0.3,
        help="HDBSCAN cluster_selection_epsilon in 1-IoU space (merges clusters within this distance).",
    )
    parser.add_argument(
        "--min_samples", type=int, default=5,
        help="HDBSCAN min_cluster_size.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Must match the batch_size used during sample generation.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--wandb_project", default=None,
        help="If set, starts a wandb run (wandb.init) and logs metrics to it. "
        "Needed because a sweep agent only sets env vars; it doesn't call wandb.init() for you.",
    )
    parser.add_argument(
        "--require_unique_modes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Only evaluate items whose raw all_targets contains no duplicate GT "
            "masks (default: True). Use --no-require_unique_modes to also "
            "evaluate items with duplicate GT modes (e.g. annotators who agree "
            "exactly, or a no-op class flip)."
        ),
    )

    args = parser.parse_args()

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, config=vars(args))

    builders = {"mmfire": _build_mmfire, "lidc": _build_lidc, "cityscapes": _build_cityscapes}
    dl, unbatch_fn, get_gt_fn = builders[args.dataset](args)

    evaluate_mode_probabilities(
        samples_dir=args.samples_dir,
        data_loader=dl,
        unbatch_fn=unbatch_fn,
        get_gt_fn=get_gt_fn,
        eps=args.eps,
        min_samples=args.min_samples,
        require_unique_modes=args.require_unique_modes,
    )


if __name__ == "__main__":
    main()

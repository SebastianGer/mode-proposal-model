import numpy as np
import torch
from scipy.optimize import minimize, nnls


def estimate_mode_probabilities_from_velocity(
    diff_model,
    segmentation_masks,
    cond,
    n_samples,
    estimation_method,
    additional_arg,
    n_subsample,
    device="cuda",
):
    """
    Estimates prior mode probabilities by switching the perspective from diffusion to flow modeling.
    The velocity at each point drawn from the prior is assumed to be a linear combination
    of the vectors to each target segmentation mask, weighted by the prior mode probabilities.
    This perspective is directly motivated by the training objective of flow models.
    By setting up such linear systems of equations for multiple points drawn from the prior,
    we can solve for the prior mode probabilities using least squares.

    The initial draft of this method was written with Gemini.

    Args:
        segmentation_masks: [K_i, 1, H, W] tensor of proposals with value range roughly [-1,1]
        cond: [C, H, W] Conditioning for the model.
    """

    sigma_max = 80.0

    K_i, H, W = segmentation_masks.shape
    dim = H * W
    x_0 = segmentation_masks.reshape(K_i, dim).float()  # effectively flattening

    # 1. Sample random points from the prior distribution
    # Shape: [n_samples, dim]
    x_t = torch.randn(n_samples, dim, device=device) * sigma_max

    # 2. Compute model velocity at these points
    t = torch.full((n_samples, 1), sigma_max, device=device)
    cond = cond.repeat(n_samples, 1, 1, 1)

    with torch.no_grad():

        x_t_reshaped = x_t.reshape(-1, 1, H, W)
        D_hat = diff_model.forward(x_t_reshaped, t.reshape(-1), cond)
        D_hat_hat = D_hat.reshape(-1, dim)
        v_est = (x_t - D_hat_hat) / t

    # 3. Compute conditional flow directions (Target Vectors)
    # For every noise sample i, compute vector to every target mask j
    # target_vectors[i, j] = -x_0[j] + x_t[i]
    # Shape: [n_samples, K, dim]
    target_vectors = -x_0.unsqueeze(0) + x_t.unsqueeze(1)

    # 4. Solve the resulting linear system using non-negative least squares (NNLS).
    # We want non-negative mode weights w_j such that for each sampled point i:
    # v_est[i] ≈ sum_j w_j * target_vectors[i, j].
    # Rearranged into a single large linear system:
    # A @ w ≈ b, where A has shape [n_samples * dim, K] and b has shape [n_samples * dim].

    A = target_vectors.permute(1, 0, 2).reshape(K_i, -1).T
    B = v_est.reshape(-1)

    # Remove rows where all entries in A are identical (no information to infer weights)
    matches_first = A == A[:, [0]]
    non_uniform_rows = ~matches_first.all(dim=1)
    A = A[non_uniform_rows, :]
    B = v_est.reshape(-1, 1)[non_uniform_rows, :]

    # Normalize to std=1, otherwise the constraint of summing to 1.0 does not permit good fits.
    A /= 80.0

    if estimation_method == "NNLS":
        A_np = np.asarray(A.detach().cpu().numpy(), dtype=np.float64)
        B_np = np.asarray(B.detach().cpu().numpy(), dtype=np.float64)

        weights, _ = nnls(A_np, B_np.squeeze(1))

    elif estimation_method in [
        "NNLS_FAST",
        "NNLS_SUM_TO_ONE",
        "SMOOTH_L1_SUM_TO_ONE",
    ]:

        if estimation_method == "NNLS_FAST":
            cons = None
            bounds = [(0, None) for _ in range(K_i)]

        else:
            # Constraint: sum(w) = 1
            cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(K_i)]

        # Initial guess: uniform distribution
        w0 = np.ones(K_i) / K_i

        if estimation_method in [
            "NNLS_FAST",
            "NNLS_SUM_TO_ONE",
        ]:
            alpha = additional_arg
            if additional_arg is None:
                alpha = 0.0

            # Reduced K-dim QP: minimize ||A w - B||^2 = w^T @ ATA @ w - 2 * w^T @ ATb + const
            # This avoids repeated O(n_rows * K) matrix-vector products in each objective eval.
            # Instead, objective evals are O(K^2), and SLSQP converges in 2-5 iterations.
            ATA = A.T @ A
            ATB = A.T @ B.flatten()
            ATA_reg = ATA + alpha * torch.eye(K_i, device=ATA.device)

            ATA_reg_np = np.asarray(ATA_reg.detach().cpu().numpy(), dtype=np.float64)
            ATB_np = np.asarray(ATB.detach().cpu().numpy(), dtype=np.float64)

            def objective(w):
                return (w @ ATA_reg_np @ w) - 2.0 * (w @ ATB_np)

        elif estimation_method == "SMOOTH_L1_SUM_TO_ONE":

            def objective(w):

                # Smooth L1 logic:
                # 0.5 * x^2 if |x| < beta
                # |x| - 0.5 * beta if |x| >= beta
                # beta is a float threshold where behavior switches from quadratic to linear
                beta = additional_arg
                num_subsamples = n_subsample

                A_np = np.asarray(A.detach().cpu().numpy(), dtype=np.float64)
                B_np = np.asarray(B.detach().cpu().numpy(), dtype=np.float64)

                if num_subsamples > 0 and A_np.shape[0] > num_subsamples:
                    # random subsampling
                    indices = np.random.choice(
                        A_np.shape[0], size=num_subsamples, replace=False
                    )
                    A_np = A_np[indices]
                    B_np = B_np[indices]

                error = A_np @ w - B_np.flatten()
                abs_error = np.abs(error)

                loss = np.where(
                    abs_error < beta,
                    0.5 * (error**2) / beta,
                    abs_error - 0.5 * beta,
                )

                return np.sum(loss)

        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)

        if res.success:
            weights = res.x
        else:
            print(f"Optimization failed: {res.message}")
            weights = res.x  # Returns the last best guess

    weights = torch.from_numpy(weights).to(
        device=segmentation_masks.device, dtype=torch.float32
    )

    total = weights.sum()
    # Guard against failure cases by returning uniform distribution.
    if total <= 0 or torch.isnan(total):
        prior_probs = torch.full_like(weights, 1.0 / weights.numel())
    else:
        # Ensure summing to 1 to receive proper probability distribution. Most important for NNLS.
        prior_probs = weights / total

    return prior_probs


def estimate_binary_codes(proposals, components):
    """
    Estimates the composition of each proposal in terms of the binary components.
    Used for Cityscapes.

    proposals: [N, H, W], binary CPU tensor
    components: [4, H, W], binary CPU tensor
    """
    N, H, W = proposals.shape
    K = components.shape[0]

    P_flat = proposals.reshape(N, -1).T  # [Pixels, N]
    C_flat = components.reshape(K, -1).T  # [Pixels, K]

    # alpha_continuous is [K, N]
    alpha_continuous, _, _, _ = np.linalg.lstsq(C_flat, P_flat, rcond=None)
    alpha_binary = (alpha_continuous > 0.5).astype(int)

    # Returns [N, 4] matrix of binary codes
    return alpha_binary.T


def estimated_mode_weights_to_bernoulli(w_raw, alpha_configs):
    """
    Projects the raw weights from the NNLS velocity solver to Bernoulli distribution probabilities in the 4 classes.

    w_raw: array of 16 weights from the NNLS velocity solver.
    alpha_configs: (16, 4) binary matrix where [m, i] is 1 if class i is 'on' in mode m.
    """
    # 1. Normalize raw weights to ensure they are a valid distribution
    w_raw = np.clip(w_raw, 1e-10, None)
    w_raw /= w_raw.sum()

    # 2. Define the objective: KL-Divergence
    # We want to find p (size 4) such that the resulting 16 mode probs
    # match w_raw as closely as possible.
    def objective(p):
        # Calculate constructed weights w_prime for all 16 modes
        # p is array of 4 probabilities [p_road, p_sidewalk, p_car, p_veg]

        # Matrix of (p_i if config==1 else 1-p_i)
        probs_matrix = alpha_configs * p + (1 - alpha_configs) * (1 - p)

        # Product across the 4 classes for each of the 16 modes
        w_prime = np.prod(probs_matrix, axis=1)

        # KL Divergence: sum(w_raw * log(w_raw / w_prime))
        # Minimizing this is equivalent to minimizing Cross-Entropy
        return -np.sum(w_raw * np.log(w_prime + 1e-10))

    # 3. Constraints: each p_i must be between 0 and 1
    bounds = [(0, 1)] * 4
    init_p = np.full(4, 0.5)  # Start at 50/50

    res = minimize(objective, init_p, bounds=bounds)

    return res.x if res.success else None


def estimate_cityscapes_priors(
    gt_seg_masks,
    diff_model,
    cond,
    n_samples,
    estimation_method,
    additional_arg,
    n_subsample,
    device="cuda",
):
    """
    Decompose the velocity at several noise points into a linear combination of the vectors to each target ground truth segmentation mask.
    This generally will not return the true mode probabilities, since the linear system is underdetermined.

    Then, we make use of the underlying lower-dimensional structure in the Cityscapes modes: Each mode is a combination of 4 binary components (road, sidewalk, car, vegetation),
    that are either "on" or "off", i.e. assigned to the positive or negative class, following a Bernoulli distribution per class.
    The weights we estimated beforehand would ideally be the result of combining the probabilities of each class being on or off according to the mode configurations.
    We thus solve a second least squares problem to find the Bernoulli probabilities for each class, that best explain the estimated mode weights from the first step.

    We assume that gt_seg_masks is a tensor with 16 modes in the first dimension and pick the single-class masks from that. This method could also be run from proposals,
    if we could reliably estimate the single-class masks from the set of proposals. In preliminary experiments, this almost worked by applying a PCA to the proposals,
    but it required manual thresholding to distinguish between the car and vegetation components. Thus, we stick with using the GT single-class masks to demonstrate that
    the velocity decomposition in principle also works for Cityscapes.
    """

    single_classes = gt_seg_masks[[8, 4, 2, 1]]
    valid_classes_mask = single_classes.bool().any(dim=(1, 2))
    valid_single_classes = single_classes  # [valid_classes_mask]

    unique_gt_seg_masks = gt_seg_masks.unique(dim=0) * 2 - 1

    estimated_mode_weights = estimate_mode_probabilities_from_velocity(
        diff_model,
        unique_gt_seg_masks,
        cond,
        n_samples,
        estimation_method,
        additional_arg,
        n_subsample,
        device,
    )

    # Estimate the binary codes for each proposal in terms of the 4 components, to receive the alpha_configs matrix for the second optimization step.
    # We could use the ground truth here, but this function could also work if we replaced gt_seg_masks with the proposals, so we keep it general.
    alpha_est = estimate_binary_codes(
        (unique_gt_seg_masks.cpu() > 0), valid_single_classes.cpu() > 0
    )
    estimated_bernoulli_weights = estimated_mode_weights_to_bernoulli(
        estimated_mode_weights.cpu().numpy(), alpha_est
    )

    # Fill computed weights in at right spot in the full class distribution, including single classes that are not present in the ground truths.
    estimated_bernoulli_weights_full = torch.zeros(
        len(single_classes), device=gt_seg_masks.device
    )
    estimated_bernoulli_weights_full[valid_classes_mask] = torch.tensor(
        estimated_bernoulli_weights, device=gt_seg_masks.device
    ).float()[valid_classes_mask]
    return estimated_bernoulli_weights_full

# analysis/kernels.py
import numpy as np


def get_first_conv_weights(model):
    return model.get_first_conv().weight.detach().cpu().numpy()


def run_kernel_stats(models_by_space, color_spaces):
    """
    Computes:
      - kernel variance per space
      - similarity of kernels to RGB (cosine)
    Returns:
      kernel_stats_by_space: {space: {"var": float, "mean_abs": float}}
      kernel_similarity_to_rgb: {space: float (cosine similarity)}
    """
    kernel_stats_by_space = {}
    kernel_similarity_to_rgb = {}

    w_rgb = get_first_conv_weights(models_by_space["rgb"]).reshape(-1)
    norm_rgb = np.linalg.norm(w_rgb) + 1e-8

    for space in color_spaces:
        w = get_first_conv_weights(models_by_space[space]).reshape(-1)
        var = float(np.var(w))
        mean_abs = float(np.mean(np.abs(w)))
        kernel_stats_by_space[space] = {"var": var, "mean_abs": mean_abs}

        sim = float(np.dot(w_rgb, w) / (norm_rgb * (np.linalg.norm(w) + 1e-8)))
        kernel_similarity_to_rgb[space] = sim

    return kernel_stats_by_space, kernel_similarity_to_rgb


def run_feature_redundancy_from_corr(corr, thresh=0.8):
    """
    Given a channel correlation matrix corr (C x C),
    compute ratio of |corr_ij|>thresh off-diagonal.
    """
    C = corr.shape[0]
    off_diag = corr[np.triu_indices(C, k=1)]
    return float(np.mean(np.abs(off_diag) > thresh))

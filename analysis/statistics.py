# analysis/statistics.py
import numpy as np
from scipy.stats import f_oneway, wilcoxon, pearsonr


def run_anova_and_wilcoxon(results, per_class_acc, color_spaces):
    """
    Runs:
      - ANOVA on per-class accuracy across color spaces
      - Wilcoxon on ΔE and SSIM vs RGB

    Returns:
      {
        "anova_p": float,
        "wilcoxon_deltaE": {space: p},
        "wilcoxon_ssim": {space: p}
      }
    """
    # ANOVA on per-class accuracy
    acc_mat = np.vstack([per_class_acc[s] for s in color_spaces])
    anova_p = f_oneway(*acc_mat).pvalue

    # Wilcoxon vs RGB for ΔE & SSIM
    wilcoxon_p_deltaE = {}
    wilcoxon_p_ssim = {}

    d_rgb = np.array(results["rgb"]["deltaE_samples"])
    s_rgb = np.array(results["rgb"]["ssim_samples"])

    for space in color_spaces:
        if space == "rgb":
            continue

        d_sp = np.array(results[space]["deltaE_samples"])
        n = min(len(d_rgb), len(d_sp))
        if n > 0:
            wilcoxon_p_deltaE[space] = wilcoxon(d_rgb[:n], d_sp[:n]).pvalue

        s_sp = np.array(results[space]["ssim_samples"])
        n2 = min(len(s_rgb), len(s_sp))
        if n2 > 0:
            wilcoxon_p_ssim[space] = wilcoxon(s_rgb[:n2], s_sp[:n2]).pvalue

    return {
        "anova_p": anova_p,
        "wilcoxon_deltaE": wilcoxon_p_deltaE,
        "wilcoxon_ssim": wilcoxon_p_ssim
    }


def run_correlations(results, color_spaces):
    """
    Pearson correlations:
      corr(acc, mean ΔE)
      corr(acc, mean SSIM)
    """
    acc_array = np.array([results[s]["test_acc"][-1] for s in color_spaces])
    deltaE_array = np.array([results[s]["deltaE_mean"] for s in color_spaces])
    ssim_array = np.array([results[s]["ssim"] for s in color_spaces])

    corr_acc_deltaE, _ = pearsonr(acc_array, deltaE_array)
    corr_acc_ssim, _ = pearsonr(acc_array, ssim_array)

    return {
        "corr_acc_deltaE": float(corr_acc_deltaE),
        "corr_acc_ssim": float(corr_acc_ssim)
    }


def bootstrap_ci(values, n_boot=2000, alpha=0.05):
    vals = np.array(values)
    boot = []
    for _ in range(n_boot):
        sample = np.random.choice(vals, size=len(vals), replace=True)
        boot.append(sample.mean())
    lower = np.percentile(boot, 100 * alpha / 2)
    upper = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def run_accuracy_ci(per_class_acc, color_spaces):
    """
    Computes 95% CI for per-class mean accuracy per space.

    Returns:
      dict[space] -> (low, high)
    """
    accuracy_ci = {}
    for s in color_spaces:
        accuracy_ci[s] = bootstrap_ci(per_class_acc[s])
    return accuracy_ci

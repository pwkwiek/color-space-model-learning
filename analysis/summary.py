# analysis/summary.py

import numpy as np
import pandas as pd
from .statistics import bootstrap_ci


def _mean_dict_values(d):
    """Return the mean of a dict's values, safely."""
    if d is None or not isinstance(d, dict) or len(d) == 0:
        return float("nan")
    try:
        return float(np.mean(list(d.values())))
    except Exception:
        return float("nan")


def build_summary_table(
    color_spaces,
    results,
    analysis_outputs,
    per_class_acc=None,
    accuracy_ci=None,
):
    """
    Build a summary DataFrame with columns:

    ColorSpace, TestAcc, TrainAcc, ClassAccMean,
    TrainLossLast, TestLossLast, Acc_CI_low, Acc_CI_high,
    Mean_ΔE, ΔE_Q25, ΔE_Q50, ΔE_Q75, SSIM,
    FLOPs_first_conv, Params, Activations, FeatVar,
    EpochTimeMean, ConvergenceRate,
    KernelVar, KernelMeanAbs, KernelSim_to_RGB,
    GCAM_IoU_vs_RGB, AttentionEntropy, Redundancy,
    EmbeddingStabilityMean,
    ColorComplexityMean, TextureComplexityMean,
    Robust_Hue, Robust_Sat, Robust_Bright,
    PruningDrop25, PruningDrop50
    """

    confusions              = analysis_outputs.get("confusions", {})
    per_class_acc_all       = per_class_acc if per_class_acc is not None else analysis_outputs.get("per_class_acc", {})
    embedding_stability_all = analysis_outputs.get("embedding_stability", {})
    color_complexity_all    = analysis_outputs.get("color_complexity", {})
    texture_complexity_all  = analysis_outputs.get("texture_complexity", {})
    color_sensitivity_all   = analysis_outputs.get("color_sensitivity", {})
    kernel_stats_by_space   = analysis_outputs.get("kernel_stats", {})
    kernel_similarity_to_rgb= analysis_outputs.get("kernel_similarity", {})
    complexity_metrics_all  = analysis_outputs.get("complexity_metrics", {})
    pruning_sensitivity_all = analysis_outputs.get("pruning_sensitivity", {})
    gcam_iou_to_rgb         = analysis_outputs.get("gcam_iou", {})
    attention_entropy       = analysis_outputs.get("attention_entropy", {})
    feature_redundancy      = analysis_outputs.get("feature_redundancy", {})
    flops_by_space          = analysis_outputs.get("flops", {})
    deltaE_stats_all        = analysis_outputs.get("deltaE_stats", {})
    ssim_stats_all          = analysis_outputs.get("ssim_stats", {})

    # ------------------------------------
    # If not provided, compute bootstrap CIs
    # ------------------------------------
    if accuracy_ci is None and per_class_acc_all:
        accuracy_ci = {
            s: bootstrap_ci(per_class_acc_all[s])
            for s in per_class_acc_all
        }

    # ------------------------------------
    # Build rows
    # ------------------------------------
    rows = []

    for space in color_spaces:
        r = results[space]
        stats = complexity_metrics_all.get(space, {})

        # --------------------------
        # Basic training metrics
        # --------------------------
        test_acc_last   = float(r["test_acc"][-1]) if "test_acc" in r else float("nan")
        train_acc_last  = float(r["train_acc"][-1]) if "train_acc" in r else float("nan")
        train_loss_last = float(r["train_loss"][-1]) if "train_loss" in r else float("nan")
        test_loss_last  = float(r["test_loss"][-1]) if "test_loss" in r else float("nan")

        pc_acc = per_class_acc_all.get(space, None)
        class_acc_mean = float(np.mean(pc_acc)) if pc_acc is not None else float("nan")

        ci_low, ci_high = (float("nan"), float("nan"))
        if accuracy_ci and space in accuracy_ci:
            ci_low, ci_high = accuracy_ci[space]

        # --------------------------
        # ΔE and SSIM metrics
        # --------------------------
        dE_mean = float(r.get("deltaE_mean", float("nan")))

        dE_q25 = dE_q50 = dE_q75 = float("nan")
        if space in deltaE_stats_all:
            q = deltaE_stats_all[space].get("quantiles", None)
            # Correct: quantiles = [q05, q25, q50, q75, q95]
            if q is not None and len(q) >= 4:
                dE_q25 = float(q[1])
                dE_q50 = float(q[2])
                dE_q75 = float(q[3])

        ssim_mean = float(r.get("ssim", float("nan")))

        flops_first_conv = float(flops_by_space.get(space, float("nan")))

        # --------------------------
        # Complexity metrics
        # --------------------------
        params          = float(stats.get("params", float("nan")))
        activations     = float(stats.get("activations", float("nan")))
        feat_var        = float(stats.get("feat_var", float("nan")))
        epoch_time_mean = float(stats.get("epoch_time_mean", float("nan")))
        convergence_rate= float(stats.get("convergence_rate", float("nan")))

        # --------------------------
        # Kernel statistics
        # --------------------------
        kstats = kernel_stats_by_space.get(space, {})
        kernel_var       = float(kstats.get("var", float("nan")))
        kernel_mean_abs  = float(kstats.get("mean_abs", float("nan")))
        kernel_sim_to_rgb= float(kernel_similarity_to_rgb.get(space, float("nan")))

        # --------------------------
        # Saliency / Attention
        # --------------------------
        gcam_iou_val = float(gcam_iou_to_rgb.get(space, float("nan")))
        att_ent      = float(attention_entropy.get(space, float("nan")))
        redund       = float(feature_redundancy.get(space, float("nan")))

        # --------------------------
        # Embedding stability
        # --------------------------
        emb = embedding_stability_all.get(space, {})
        emb_stab_mean = _mean_dict_values(emb)

        # --------------------------
        # Per-space color + texture complexity
        # --------------------------
        cc = color_complexity_all.get(space, None)
        tc = texture_complexity_all.get(space, None)
        color_complex_mean   = _mean_dict_values(cc)
        texture_complex_mean = _mean_dict_values(tc)

        # --------------------------
        # Robustness metrics
        # --------------------------
        sens = color_sensitivity_all.get(space, {})
        robust_hue    = float(sens.get("hue", float("nan")))
        robust_sat    = float(sens.get("sat", float("nan")))
        robust_bright = float(sens.get("bright", float("nan")))

        # --------------------------
        # Pruning sensitivity
        # --------------------------
        prune = pruning_sensitivity_all.get(space, {})
        drop25 = float(prune.get("drop25", float("nan")))
        drop50 = float(prune.get("drop50", float("nan")))

        # --------------------------
        # Build row
        # --------------------------
        row = {
            "ColorSpace": space.upper(),
            "TestAcc": test_acc_last,
            "TrainAcc": train_acc_last,
            "ClassAccMean": class_acc_mean,
            "TrainLossLast": train_loss_last,
            "TestLossLast": test_loss_last,
            "Acc_CI_low": ci_low,
            "Acc_CI_high": ci_high,
            "Mean_ΔE": dE_mean,
            "ΔE_Q25": dE_q25,
            "ΔE_Q50": dE_q50,
            "ΔE_Q75": dE_q75,
            "SSIM": ssim_mean,
            "FLOPs_first_conv": flops_first_conv,
            "Params": params,
            "Activations": activations,
            "FeatVar": feat_var,
            "EpochTimeMean": epoch_time_mean,
            "ConvergenceRate": convergence_rate,
            "KernelVar": kernel_var,
            "KernelMeanAbs": kernel_mean_abs,
            "KernelSim_to_RGB": kernel_sim_to_rgb,
            "GCAM_IoU_vs_RGB": gcam_iou_val,
            "AttentionEntropy": att_ent,
            "Redundancy": redund,
            "EmbeddingStabilityMean": emb_stab_mean,
            "ColorComplexityMean": color_complex_mean,
            "TextureComplexityMean": texture_complex_mean,
            "Robust_Hue": robust_hue,
            "Robust_Sat": robust_sat,
            "Robust_Bright": robust_bright,
            "PruningDrop25": drop25,
            "PruningDrop50": drop50,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("TestAcc", ascending=False).reset_index(drop=True)
    return df

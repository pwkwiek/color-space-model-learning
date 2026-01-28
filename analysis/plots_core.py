# analysis/plots_core.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
#  Save directory for core plots
# -----------------------------------------------------------------------------
PLOT_DIR_CORE = os.path.join("results", "plots", "core")
os.makedirs(PLOT_DIR_CORE, exist_ok=True)


def _sanitize_filename(filename: str) -> str:
    """
    Make a filename safe for Windows:
    - remove newlines and weird unicode
    - allow only alnum + . _ - ( )
    - replace others with '_'
    """
    # Replace newlines and tabs with spaces
    filename = filename.replace("\n", "_").replace("\r", "_").replace("\t", "_")

    # Optionally normalize some symbols
    filename = filename.replace("↔", "-")

    # Keep only safe characters
    safe = []
    for ch in filename:
        if ch.isalnum() or ch in "._-()":
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _savefig_core(filename: str):
    """
    Save current matplotlib figure into the core plots directory.
    """
    safe_name = _sanitize_filename(filename)
    path = os.path.join(PLOT_DIR_CORE, safe_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")



# ============================================================
#  Helper radar plot utilities
# ============================================================

def radar_plot_multi_space(metric_dict, title):
    """
    metric_dict:
        space -> scalar
        OR
        space -> {label: value}
    Automatically handles both cases.
    """

    labels_spaces = list(metric_dict.keys())
    first_val = next(iter(metric_dict.values()))

    # ============================================================
    # CASE 1: dict of scalars  (e.g., {"rgb": 0.8, "lab": 0.7, ...})
    # ============================================================
    if isinstance(first_val, (int, float, np.floating)):
        labels = ["value"]  # one axis
        values = {s: [metric_dict[s]] for s in labels_spaces}

    # ============================================================
    # CASE 2: dict of dicts (e.g., {"rgb": {"class1": ..., "class2": ...}, ...})
    # ============================================================
    else:
        labels = list(first_val.keys())
        values = {
            s: [metric_dict[s][lab] for lab in labels]
            for s in labels_spaces
        }

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    cmap = plt.get_cmap("tab10")

    for i, space in enumerate(labels_spaces):
        vals = values[space]
        vals = vals + vals[:1]  # close shape

        ax.plot(angles, vals, linewidth=2, label=space.upper(), color=cmap(i))
        ax.fill(angles, vals, alpha=0.1, color=cmap(i))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=14, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save (filename sanitized by _savefig_core)
    _savefig_core(title.replace(" ", "_").lower() + ".png")
    plt.show()



def radar_plot_single_metric(metric_dict, title):
    """
    metric_dict: label -> scalar (e.g. SSIM per color space)
    Single polygon, each axis = color space.
    """
    if not metric_dict:
        return

    labels = list(metric_dict.keys())
    values = [metric_dict[k] for k in labels]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_rlabel_position(0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig_core(title.replace(" ", "_").lower() + ".png")
    plt.show()


# ============================================================
# 1. LOSS & ACCURACY PLOTS
# ============================================================

def plot_loss_and_accuracy(results, color_spaces):
    """
    Plots:
      - Train & Test loss per epoch
      - Train & Test accuracy per epoch

    Uses consistent colors per color space:
      solid = train, dashed = test.
    """
    cmap = plt.get_cmap("tab10")
    color_map = {space: cmap(i % 10) for i, space in enumerate(color_spaces)}

    # ---------- LOSS ----------
    plt.figure(figsize=(10, 6))
    for space in color_spaces:
        r = results[space]
        c = color_map[space]
        train_loss = r.get("train_loss", r.get("loss", []))
        test_loss = r.get("test_loss", None)

        epochs = range(1, len(train_loss) + 1)

        # train loss (solid)
        plt.plot(
            epochs,
            train_loss,
            label=f"{space.upper()} Train Loss",
            color=c,
            linestyle="-",
            linewidth=2,
        )

        # test loss (dashed) if available
        if test_loss is not None:
            plt.plot(
                epochs,
                test_loss,
                label=f"{space.upper()} Test Loss",
                color=c,
                linestyle="--",
                linewidth=2,
            )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train & Test Loss per Epoch")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    _savefig_core("loss_train_test.png")
    plt.show()

    # ---------- ACCURACY ----------
    plt.figure(figsize=(10, 6))
    for space in color_spaces:
        r = results[space]
        c = color_map[space]

        train_acc = r["train_acc"]
        test_acc = r["test_acc"]
        epochs = range(1, len(train_acc) + 1)

        plt.plot(
            epochs,
            train_acc,
            label=f"{space.upper()} Train Acc",
            color=c,
            linestyle="-",
            linewidth=2,
        )
        plt.plot(
            epochs,
            test_acc,
            label=f"{space.upper()} Test Acc",
            color=c,
            linestyle="--",
            linewidth=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train & Test Accuracy per Epoch")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    _savefig_core("accuracy_train_test.png")
    plt.show()


# ============================================================
# 2. CONFUSION MATRICES
# ============================================================

def plot_confusion_matrices(confusions, class_names, color_spaces):
    """
    confusions: dict[space] -> confusion matrix ndarray
    """
    for space in color_spaces:
        cm = confusions[space]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=False, cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix – {space.upper()}")
        plt.tight_layout()
        _savefig_core(f"confusion_{space}.png")
        plt.show()


# ============================================================
# 3. ΔE & SSIM PLOTS
# ============================================================

def plot_deltaE_distributions(results, color_spaces):
    """
    KDE plots of ΔE samples for each color space.
    """
    plt.figure(figsize=(10, 6))
    for space in color_spaces:
        samples = results[space]["deltaE_samples"]
        if len(samples) == 0:
            continue
        sns.kdeplot(samples, label=space.upper(), fill=True, alpha=0.2)
    plt.title("ΔE Distribution Across Color Spaces")
    plt.xlabel("ΔE (CIEDE2000)")
    plt.legend()
    plt.tight_layout()
    _savefig_core("deltaE_distribution.png")
    plt.show()


def plot_deltaE_per_class_bar(results, color_spaces, class_names):
    """
    Bar plot of per-class mean ΔE for each color space.
    """
    data = {}
    for space in color_spaces:
        stats = results[space]["deltaE_class_stats"]  # dict[class -> float]
        data[space.upper()] = [stats.get(cls, np.nan) for cls in class_names]

    df = pd.DataFrame(data, index=class_names)

    df.plot(kind="bar", figsize=(12, 6))
    plt.title("Per-Class Mean ΔE (CIEDE2000)")
    plt.ylabel("ΔE")
    plt.xlabel("Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    _savefig_core("deltaE_per_class_bar.png")
    plt.show()


def plot_representative_deltaE_maps(results, color_spaces):
    """
    Shows one representative ΔE map per color space.
    """
    n = len(color_spaces)
    plt.figure(figsize=(4 * n, 4))
    for i, space in enumerate(color_spaces):
        dE_map = results[space]["deltaE_rep_map"]
        plt.subplot(1, n, i + 1)
        plt.imshow(dE_map, cmap="magma")
        plt.title(space.upper())
        plt.axis("off")
    plt.suptitle("Representative ΔE (CIEDE2000) Maps – First-layer features", y=1.02)
    plt.tight_layout()
    _savefig_core("deltaE_rep_maps.png")
    plt.show()


def plot_ssim_distributions(results, color_spaces):
    """
    KDE plots of SSIM roundtrip samples for each color space.
    """
    plt.figure(figsize=(10, 6))
    for space in color_spaces:
        samples = results[space]["ssim_samples"]
        if len(samples) == 0:
            continue
        sns.kdeplot(samples, label=space.upper(), fill=True, alpha=0.2)
    plt.title("SSIM Distribution Across Color Spaces\n(RGB ↔ Colorspace Round Trip)")
    plt.xlabel("SSIM")
    plt.legend()
    plt.tight_layout()
    _savefig_core("ssim_distribution.png")
    plt.show()


def radar_ssim(results, color_spaces):
    """
    Radar plot of mean SSIM per color space.
    """
    ssim_metric = {space.upper(): results[space]["ssim"] for space in color_spaces}
    radar_plot_single_metric(ssim_metric,
                             "Mean SSIM Across Color Spaces\n(RGB ↔ Colorspace Round Trip)")


def radar_deltaE_per_class(results, color_spaces, class_names):
    """
    Radar of per-class ΔE, aggregated per space.
    """
    metric = {}
    for space in color_spaces:
        per_class = results[space]["deltaE_class_stats"]
        metric[space] = {cls: per_class.get(cls, 0.0) for cls in class_names}
    radar_plot_multi_space(metric, "Per-Class Mean ΔE Across Color Spaces")


# ============================================================
# 4. CORRELATION HEATMAPS
# ============================================================

def plot_correlation_heatmaps(results, color_spaces):
    """
    Plot channel-wise correlation heatmaps for the first-layer features.

    If results[space]["corr"] is present, use it.
    Otherwise, compute it from results[space]["first_layer_activation"].
    """

    for space in color_spaces:
        r = results[space]

        # 1. Try to use precomputed corr, if available
        corr = r.get("corr", None)

        # 2. If not available, compute from first_layer_activation
        if corr is None:
            feats = r.get("first_layer_activation", None)
            if feats is None:
                # nothing to plot for this space
                print(f"[plot_correlation_heatmaps] No first_layer_activation for {space}, skipping.")
                continue

            # feats: [B, C, H, W] -> compute channel-wise correlation
            feats = feats.detach().cpu()
            B, C, H, W = feats.shape
            # average spatially, leave (B, C)
            feats_flat = feats.view(B, C, -1).mean(dim=2).numpy()
            # normalize per-channel
            feats_flat = (feats_flat - feats_flat.mean(axis=0)) / (
                feats_flat.std(axis=0) + 1e-8
            )
            corr = np.corrcoef(feats_flat.T)
            # store back so next calls don't recompute
            r["corr"] = corr

        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"First-layer feature correlation – {space.upper()}")
        plt.tight_layout()
        _savefig_core(f"corr_heatmap_{space}.png")
        plt.show()

# ============================================================
# 5. KERNEL STATS & DIVERSITY
# ============================================================

def plot_kernel_variance_and_similarity(kernel_stats_by_space,
                                        kernel_similarity_to_rgb,
                                        color_spaces):
    """
    Two bar plots:
      - kernel variance
      - similarity to RGB
    """
    vars_raw = [kernel_stats_by_space[s]["var"] for s in color_spaces]
    sims_raw = [kernel_similarity_to_rgb[s] for s in color_spaces]
    xs = [s.upper() for s in color_spaces]

    plt.figure(figsize=(10, 4))
    sns.barplot(x=xs, y=vars_raw)
    plt.title("Kernel Weight Variance per Color Space")
    plt.ylabel("Variance")
    plt.tight_layout()
    _savefig_core("kernel_variance.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.barplot(x=xs, y=sims_raw)
    plt.title("Kernel Similarity to RGB (cosine)")
    plt.ylabel("Similarity")
    plt.tight_layout()
    _savefig_core("kernel_similarity_to_rgb.png")
    plt.show()


def plot_kernel_diversity_scatter(kernel_stats_by_space,
                                  kernel_similarity_to_rgb,
                                  color_spaces):
    """
    Scatter: x = variance, y = 1 - similarity_to_RGB
    """
    spaces = color_spaces
    vars_raw = [kernel_stats_by_space[s]["var"] for s in spaces]
    dis_raw = [1.0 - kernel_similarity_to_rgb[s] for s in spaces]

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")

    for i, s in enumerate(spaces):
        plt.scatter(vars_raw[i], dis_raw[i],
                    color=cmap(i), s=120, edgecolor="black")
        plt.text(vars_raw[i] * 1.02, dis_raw[i] * 1.02,
                 s.upper(), fontsize=12)

    plt.xlabel("Kernel Variance", fontsize=12)
    plt.ylabel("1 - Similarity to RGB", fontsize=12)
    plt.title("Kernel Diversity Across Color Spaces", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig_core("kernel_diversity_scatter.png")
    plt.show()


# ============================================================
# 6. EMBEDDING STABILITY & CLASS COMPLEXITY
# ============================================================

def plot_embedding_stability(embedding_stability, color_spaces, class_names):
    """
    embedding_stability: dict[space != "rgb"] -> {class_name: cosine}
    """
    # Radar
    radar_plot_multi_space(embedding_stability,
                           "Embedding Stability vs RGB (cosine similarity per class)")

    # KDE of all stability scores per space
    plt.figure(figsize=(10, 6))
    for space in embedding_stability:
        vals = list(embedding_stability[space].values())
        if len(vals) == 0:
            continue
        sns.kdeplot(vals, fill=True, alpha=0.3, label=space.upper())
    plt.title("Embedding Stability Distribution (Similarity to RGB Embeddings)")
    plt.xlabel("Cosine similarity")
    plt.legend()
    plt.tight_layout()
    _savefig_core("embedding_stability_kde.png")
    plt.show()


def plot_color_texture_complexity(color_complexity, texture_complexity):
    """
    Plot color and texture complexity per class.

    Accepts either:
      - color_complexity: {class_name: float}
        texture_complexity: {class_name: float}

      OR:
      - color_complexity: {space: {class_name: float}}
        texture_complexity: {space: {class_name: float}}
        In this case we pick 'rgb' if available, else the first space.
    """

    # ---------- Normalize shape to per-class dict ----------
    def _normalize_complexity_dict(comp):
        if not isinstance(comp, dict):
            return None  # unexpected type

        if not comp:  # empty dict
            return comp

        first_val = next(iter(comp.values()))

        # Case 1: per-class already (e.g. {"melanoma": 0.12, "nevus": 0.09})
        if isinstance(first_val, (int, float, np.floating)):
            return comp

        # Case 2: dict-of-dicts: {space: {class: value}}
        if isinstance(first_val, dict):
            if "rgb" in comp:
                return comp["rgb"]
            # else fallback to any space
            first_space = next(iter(comp.keys()))
            return comp[first_space]

        # Anything else: unsupported
        return None

    cc = _normalize_complexity_dict(color_complexity)
    tc = _normalize_complexity_dict(texture_complexity)

    if cc is None or tc is None:
        print("[plot_color_texture_complexity] Invalid input shapes, skipping plot.")
        return

    classes = list(cc.keys())
    color_vals = [cc[c] for c in classes]
    texture_vals = [tc.get(c, 0.0) for c in classes]

    # ---------- Color complexity ----------
    plt.figure(figsize=(8, 4))
    sns.barplot(x=classes, y=color_vals)
    plt.title("Color Complexity per Class")
    plt.ylabel("Color Std across channels")
    plt.xticks(rotation=45)
    plt.tight_layout()
    _savefig_core("color_complexity_per_class.png")
    plt.show()

    # ---------- Texture complexity ----------
    plt.figure(figsize=(8, 4))
    sns.barplot(x=classes, y=texture_vals)
    plt.title("Texture Complexity per Class")
    plt.ylabel("Mean Gradient Magnitude")
    plt.xticks(rotation=45)
    plt.tight_layout()
    _savefig_core("texture_complexity_per_class.png")
    plt.show()


# ============================================================
# 7. ROBUSTNESS PLOTS
# ============================================================

def radar_color_robustness(color_sensitivity, color_spaces):
    """
    color_sensitivity: dict[space] -> {"hue": ratio, "sat": ratio, "bright": ratio}
    """
    radar_plot_multi_space(color_sensitivity,
                           "Color Sensitivity (accuracy ratio after perturbation)")

    rob_df = pd.DataFrame(color_sensitivity).T
    rob_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Color Robustness Metrics (Hue / Sat / Brightness)")
    plt.ylabel("Accuracy Ratio")
    plt.xticks(rotation=0)
    plt.tight_layout()
    _savefig_core("color_robustness_bar.png")
    plt.show()


# ============================================================
# 8. PER-CLASS ACCURACY RADAR
# ============================================================

def radar_per_class_accuracy(per_class_acc, color_spaces, class_names):
    """
    per_class_acc: dict[space] -> np.array [num_classes]
    """
    metric = {
        space: {
            class_names[i]: float(per_class_acc[space][i])
            for i in range(len(class_names))
        }
        for space in color_spaces
    }
    radar_plot_multi_space(metric, "Per-Class Accuracy Across Color Spaces")


# ============================================================
# 9. COMPLEXITY METRICS BASIC PLOTS
# ============================================================

def plot_complexity_metrics(complexity_metrics, color_spaces):
    """
    complexity_metrics[space] = {
        "params", "activations", "feat_var",
        "epoch_time_mean", "convergence_rate"
    }
    """
    df = pd.DataFrame(complexity_metrics).T

    # Params & activations
    df[["params", "activations"]].plot(kind="bar", figsize=(10, 6))
    plt.title("Model Complexity: Parameters & Activations")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    _savefig_core("complexity_params_activations.png")
    plt.show()

    # Feature variance
    plt.figure(figsize=(8, 4))
    sns.barplot(x=df.index.str.upper(), y=df["feat_var"])
    plt.title("Average First-Layer Feature Variance")
    plt.xlabel("Color Space")
    plt.ylabel("Variance")
    plt.tight_layout()
    _savefig_core("complexity_feature_variance.png")
    plt.show()

    # Convergence rate
    plt.figure(figsize=(8, 4))
    sns.barplot(x=df.index.str.upper(), y=df["convergence_rate"])
    plt.title("Convergence Rate per Color Space")
    plt.xlabel("Color Space")
    plt.ylabel("ΔLoss per Epoch (approx)")
    plt.tight_layout()
    _savefig_core("complexity_convergence_rate.png")
    plt.show()


# ============================================================
# 10. PRUNING SENSITIVITY
# ============================================================

def plot_pruning_sensitivity(pruning_sensitivity, color_spaces):
    """
    pruning_sensitivity[space] = {"drop25": acc_drop, "drop50": acc_drop}
    """
    plt.figure(figsize=(10, 6))

    for space in color_spaces:
        vals = pruning_sensitivity[space]
        x = [25, 50]
        y = [vals.get("drop25", 0.0), vals.get("drop50", 0.0)]
        plt.plot(x, y, marker="o", label=space.upper())

    plt.title("Accuracy Drop After Pruning First Conv")
    plt.xlabel("Pruning Percentage (%)")
    plt.ylabel("Accuracy Drop (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _savefig_core("pruning_sensitivity.png")
    plt.show()

def plot_epoch_time_vs_train_accuracy(results, complexity_metrics, color_spaces):
    """
    Scatter plot: epoch_time_mean (x) vs final train_acc (y)
    + Also plots the full training curves normalized by epoch time.
    """
    plt.figure(figsize=(12, 5))

    # --- Scatter: Epoch time vs final training accuracy ---
    plt.subplot(1, 2, 1)
    xs, ys, labels = [], [], []

    for space in color_spaces:
        epoch_time = complexity_metrics[space]["epoch_time_mean"]
        final_acc  = results[space]["train_acc"][-1]

        xs.append(epoch_time)
        ys.append(final_acc)
        labels.append(space.upper())

        plt.scatter(epoch_time, final_acc, s=120)

    for x, y, label in zip(xs, ys, labels):
        plt.text(x, y, label, fontsize=10, ha="right")

    plt.xlabel("Mean Epoch Time (sec)")
    plt.ylabel("Final Training Accuracy")
    plt.title("Epoch Time vs Final Training Accuracy")

    # --- Training Accuracy Curve vs Time ---
    plt.subplot(1, 2, 2)

    for space in color_spaces:
        acc = np.array(results[space]["train_acc"])
        t = np.arange(len(acc)) * complexity_metrics[space]["epoch_time_mean"]
        plt.plot(t, acc, label=space.upper(), linewidth=2)

    plt.xlabel("Training Time (sec)")
    plt.ylabel("Train Accuracy")
    plt.title("Training Accuracy Curve Reparametrized by Time")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 11. MASTER FUNCTION: RUN ALL CORE PLOTS
# ============================================================

def run_all_core_plots(
    results,
    color_spaces,
    class_names,
    confusions,
    per_class_acc,
    embedding_stability,
    color_complexity,
    texture_complexity,
    color_sensitivity,
    kernel_stats_by_space,
    kernel_similarity_to_rgb,
    complexity_metrics,
    pruning_sensitivity,
):
    """
    Convenience wrapper to generate all core plots in one call.
    """
    # 1. Loss & accuracy
    plot_loss_and_accuracy(results, color_spaces)

    # 2. Confusions
    plot_confusion_matrices(confusions, class_names, color_spaces)

    # 3. ΔE & SSIM
    plot_deltaE_distributions(results, color_spaces)
    plot_deltaE_per_class_bar(results, color_spaces, class_names)
    plot_representative_deltaE_maps(results, color_spaces)
    plot_ssim_distributions(results, color_spaces)
    radar_ssim(results, color_spaces)
    radar_deltaE_per_class(results, color_spaces, class_names)

    # 4. Correlation
    plot_correlation_heatmaps(results, color_spaces)

    # 5. Kernels
    plot_kernel_variance_and_similarity(kernel_stats_by_space,
                                        kernel_similarity_to_rgb,
                                        color_spaces)
    plot_kernel_diversity_scatter(kernel_stats_by_space,
                                  kernel_similarity_to_rgb,
                                  color_spaces)

    # 6. Embeddings
    plot_embedding_stability(embedding_stability, color_spaces, class_names)
    plot_color_texture_complexity(color_complexity, texture_complexity)

    # 7. Robustness
    radar_color_robustness(color_sensitivity, color_spaces)

    # 8. Per-class accuracy
    radar_per_class_accuracy(per_class_acc, color_spaces, class_names)

    # 9. Complexity & pruning
    plot_complexity_metrics(complexity_metrics, color_spaces)
    plot_pruning_sensitivity(pruning_sensitivity, color_spaces)

    plot_epoch_time_vs_train_accuracy(results, complexity_metrics, color_spaces)


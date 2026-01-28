# analysis/plots_full.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .plots_core import run_all_core_plots
from analysis.gradcam import run_gradcam_maps_for_sample

PLOT_DIR_FULL = os.path.join("results", "plots", "full")
os.makedirs(PLOT_DIR_FULL, exist_ok=True)


def _savefig_full(filename: str):
    path = os.path.join(PLOT_DIR_FULL, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")


# ============================================================
# 1. GRAD-CAM OVERLAYS & DIFFERENCES
# ============================================================

def _overlay_cam_on_image(img, cam):
    """
    img: [H,W,3] in [0,1]
    cam: [H,W] in [0,1]
    returns overlayed [H,W,3]
    """
    import cv2
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = cv2.applyColorMap((cam_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = 0.5 * img + 0.5 * heatmap
    return np.clip(overlay, 0, 1)


def plot_gradcam_overlays_for_sample(models_by_space, loader, color_spaces, idx_in_batch=0):
    """
    Overlays Grad-CAM heatmaps onto the original image for one sample across all color spaces.
    """
    data, target = next(iter(loader))
    img_tensor = data[idx_in_batch].cpu()
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    cams = run_gradcam_maps_for_sample(models_by_space, loader, color_spaces, idx_in_batch)

    n = len(color_spaces)
    plt.figure(figsize=(4 * n, 4))

    for i, space in enumerate(color_spaces):
        cam = cams[space]["cam"]
        overlay = _overlay_cam_on_image(img_np, cam)
        true_label = cams[space]["true"]
        pred_label = cams[space]["pred"]

        plt.subplot(1, n, i + 1)
        plt.imshow(overlay)
        plt.title(f"{space.upper()}\ntrue={true_label}, pred={pred_label}")
        plt.axis("off")

    plt.suptitle("Grad-CAM Overlays Across Color Spaces", y=1.05)
    plt.tight_layout()
    _savefig_full(f"gradcam_overlays_idx{idx_in_batch}.png")
    plt.show()


def plot_gradcam_difference_vs_rgb(models_by_space, loader, color_spaces, idx_in_batch=0):
    """
    For one sample: |CAM(space) - CAM(RGB)|
    """
    cams = run_gradcam_maps_for_sample(models_by_space, loader, color_spaces, idx_in_batch)
    rgb_cam = cams["rgb"]["cam"]

    n = len(color_spaces) - 1
    plt.figure(figsize=(4 * n, 4))

    i = 0
    for space in color_spaces:
        if space == "rgb":
            continue
        cam = cams[space]["cam"]
        diff = np.abs(rgb_cam - cam)

        plt.subplot(1, n, i + 1)
        plt.imshow(diff, cmap="inferno")
        plt.title(f"{space.upper()} CAM Δ vs RGB")
        plt.axis("off")
        i += 1

    plt.suptitle("Grad-CAM Difference vs RGB", y=1.05)
    plt.tight_layout()
    _savefig_full(f"gradcam_diff_vs_rgb_idx{idx_in_batch}.png")
    plt.show()


# ============================================================
# 2. GRAD-CAM IoU & ATTENTION ENTROPY
# ============================================================

def plot_gradcam_iou_bar(gcam_iou_to_rgb):
    """
    gcam_iou_to_rgb: dict[space != "rgb"] -> IoU
    """
    spaces = list(gcam_iou_to_rgb.keys())
    vals = list(gcam_iou_to_rgb.values())

    plt.figure(figsize=(7, 4))
    sns.barplot(x=[s.upper() for s in spaces], y=vals)
    plt.title("Grad-CAM IoU vs RGB")
    plt.ylabel("IoU")
    plt.tight_layout()
    _savefig_full("gradcam_iou_vs_rgb.png")
    plt.show()


def plot_attention_entropy(attention_entropy, color_spaces):
    """
    attention_entropy: dict[space] -> mean entropy
    """
    xs = [s.upper() for s in color_spaces]
    ys = [attention_entropy[s] for s in color_spaces]

    plt.figure(figsize=(7, 4))
    sns.barplot(x=xs, y=ys)
    plt.title("Grad-CAM Attention Entropy per Color Space")
    plt.ylabel("Entropy")
    plt.tight_layout()
    _savefig_full("gradcam_attention_entropy.png")
    plt.show()


# ============================================================
# 3. FEATURE REDUNDANCY
# ============================================================

def plot_feature_redundancy(feature_redundancy, color_spaces):
    """
    feature_redundancy: dict[space] -> ratio
    """
    xs = [s.upper() for s in color_spaces]
    ys = [feature_redundancy[s] for s in color_spaces]

    plt.figure(figsize=(8, 4))
    sns.barplot(x=xs, y=ys)
    plt.title("Feature Redundancy Ratio per Color Space")
    plt.ylabel("Redundancy ratio (|corr|>0.8)")
    plt.tight_layout()
    _savefig_full("feature_redundancy_ratio.png")
    plt.show()


# ============================================================
# 4. COMPLEXITY: EXTRA VIEWS
# ============================================================

def plot_epoch_time_vs_accuracy(results, color_spaces):
    """
    Scatter: x = epoch_time_mean, y = final accuracy
    """
    xs, ys, labels = [], [], []
    for space in color_spaces:
        xs.append(np.mean(results[space]["epoch_times"]))
        ys.append(results[space]["test_acc"][-1])
        labels.append(space.upper())

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=80)
    for x, y, lbl in zip(xs, ys, labels):
        plt.text(x * 1.01, y * 1.01, lbl)
    plt.xlabel("Mean Epoch Time (s)")
    plt.ylabel("Final Test Accuracy (%)")
    plt.title("Epoch Time vs Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig_full("epoch_time_vs_accuracy.png")
    plt.show()


def plot_accuracy_vs_deltaE_ssim(results, color_spaces):
    """
    Scatter of accuracy vs ΔE and accuracy vs SSIM.
    """
    accs = [results[s]["test_acc"][-1] for s in color_spaces]
    dEs = [results[s]["deltaE_mean"] for s in color_spaces]
    ssim_vals = [results[s]["ssim"] for s in color_spaces]
    labels = [s.upper() for s in color_spaces]

    # Acc vs ΔE
    plt.figure(figsize=(7, 5))
    plt.scatter(dEs, accs, s=80)
    for x, y, lbl in zip(dEs, accs, labels):
        plt.text(x * 1.01, y * 1.01, lbl)
    plt.xlabel("Mean ΔE")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Mean ΔE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig_full("acc_vs_deltaE.png")
    plt.show()

    # Acc vs SSIM
    plt.figure(figsize=(7, 5))
    plt.scatter(ssim_vals, accs, s=80)
    for x, y, lbl in zip(ssim_vals, accs, labels):
        plt.text(x * 1.01, y * 1.01, lbl)
    plt.xlabel("Mean SSIM")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Mean SSIM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig_full("acc_vs_ssim.png")
    plt.show()


# ============================================================
# 5. MASTER FUNCTION: RUN ALL FULL PLOTS
# ============================================================

def run_all_full_plots(
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
    gcam_iou_to_rgb,
    attention_entropy,
    feature_redundancy,
    models_by_space,
    test_loader,
    gradcam_sample_indices=(0, 5, 10),
):
    """
    Calls:
      - all core plots
      - plus Grad-CAM, entropy, IoU, redundancy, extra complexity plots
    """
    # 1. Core suite
    run_all_core_plots(
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
    )

    # 2. Grad-CAM Overlays and Differences
    for idx in gradcam_sample_indices:
        print(f"\nGrad-CAM overlays for sample idx={idx}")
        plot_gradcam_overlays_for_sample(models_by_space, test_loader, color_spaces, idx_in_batch=idx)
        print(f"Grad-CAM differences vs RGB for idx={idx}")
        plot_gradcam_difference_vs_rgb(models_by_space, test_loader, color_spaces, idx_in_batch=idx)

    # 3. IoU & entropy
    plot_gradcam_iou_bar(gcam_iou_to_rgb)
    plot_attention_entropy(attention_entropy, color_spaces)

    # 4. Feature redundancy
    plot_feature_redundancy(feature_redundancy, color_spaces)

    # 5. Extra complexity / accuracy relations
    plot_epoch_time_vs_accuracy(results, color_spaces)
    plot_accuracy_vs_deltaE_ssim(results, color_spaces)

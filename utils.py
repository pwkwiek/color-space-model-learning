# utils.py
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# Device & Seeds
# ------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = 42):
    global SEED
    SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[utils] Using device: {DEVICE}")

# ------------------------------------------------------------------
# Simple plotting helpers
# ------------------------------------------------------------------
def plot_loss_and_accuracy(results, color_spaces):
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    color_map = {space: cmap(i) for i, space in enumerate(color_spaces)}

    for space in color_spaces:
        r = results[space]
        c = color_map[space]

        plt.plot(
            r["train_acc"],
            label=f"{space.upper()} Train",
            color=c,
            linewidth=2
        )
        plt.plot(
            r["test_acc"],
            label=f"{space.upper()} Test",
            color=c,
            linestyle="--",
            linewidth=2
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train/Test Accuracy per Epoch")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmaps(results):
    for c in results:
        corr = results[c]["corr"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"First-layer feature correlation – {c.upper()}")
        plt.tight_layout()
        plt.show()


def plot_representative_deltaE_maps(results):
    plt.figure(figsize=(4 * len(results), 4))
    for i, c in enumerate(results):
        dE_map = results[c]["deltaE_rep_map"]
        plt.subplot(1, len(results), i + 1)
        plt.imshow(dE_map, cmap="magma")
        plt.title(c.upper())
        plt.axis('off')
    plt.suptitle("Representative ΔE (CIEDE2000) Maps – First-layer features")
    plt.tight_layout()
    plt.show()


def radar_plot_multi_space(metric_dict, title):
    """
    metric_dict:
        space -> scalar
        OR
        space -> {label: value}
    """
    spaces = list(metric_dict.keys())
    first_val = next(iter(metric_dict.values()))

    import numpy as np

    if isinstance(first_val, (int, float, np.floating)):
        labels = ["value"]
        values = {s: [metric_dict[s]] for s in spaces}
    else:
        labels = list(first_val.keys())
        values = {s: [metric_dict[s][lab] for lab in labels] for s in spaces}

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    for space in spaces:
        vals = values[space]
        vals = vals + vals[:1]
        ax.plot(angles, vals, linewidth=2, label=space.upper())
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=14, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.show()


def radar_plot_single_metric(metric_dict, title):
    import numpy as np

    labels = list(metric_dict.keys())
    values = [metric_dict[k] for k in labels]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_rlabel_position(0)
    plt.tight_layout()
    plt.show()

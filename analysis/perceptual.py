# analysis/perceptual.py
import numpy as np
import torch
from skimage import color
from skimage.color import deltaE_ciede2000

from utils import DEVICE
from colorspace import rgb_to_color_batch_cpu, color_to_rgb_batch_cpu


def compute_deltaE_map_from_features(feat1, feat2):
    """
    feat1, feat2: [C,H,W] arrays or tensors (first-layer features).
    We:
      - take first 3 channels
      - normalize to [0,1]
      - convert to LAB
      - compute ΔE map
    """
    if isinstance(feat1, torch.Tensor):
        feat1 = feat1.numpy()
    if isinstance(feat2, torch.Tensor):
        feat2 = feat2.numpy()

    f1 = np.moveaxis(feat1[:3], 0, -1)
    f2 = np.moveaxis(feat2[:3], 0, -1)

    def norm01(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    f1 = norm01(f1)
    f2 = norm01(f2)

    f1_safe = np.clip(f1, 1e-6, 1 - 1e-6)
    f2_safe = np.clip(f2, 1e-6, 1 - 1e-6)
    lab1 = color.rgb2lab(f1_safe)
    lab2 = color.rgb2lab(f2_safe)

    return deltaE_ciede2000(lab1, lab2)


def run_deltaE(model, loader, class_names, color_space="rgb",
               samples_per_class=20):
    """
    High-level ΔE analysis.

    Returns dict:
    {
      "mean": float,
      "quantiles": list[5],
      "per_class": {cls: float},
      "rep_map": 2D array,
      "samples": list[float]
    }
    """
    model.eval()
    feats_by_class = {c: [] for c in class_names}

    # 1. Collect features
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE)
            _ = model(data)  # assumes model already uses convert_and_normalize inside training
            feats = model.first_layer_act.cpu()
            targets_np = targets.numpy()

            for i, cls_id in enumerate(targets_np):
                cls = class_names[int(cls_id)]
                if len(feats_by_class[cls]) < samples_per_class:
                    feats_by_class[cls].append(feats[i])

            if all(len(feats_by_class[c]) >= 2 for c in class_names):
                break

    # 2. Compute ΔE stats
    per_class_deltaE = {}
    all_deltaE = []
    rep_map = None

    for cls in class_names:
        vecs = feats_by_class[cls]
        if len(vecs) < 2:
            per_class_deltaE[cls] = np.nan
            continue

        deltas = []
        for i in range(0, len(vecs) - 1, 2):
            f1, f2 = vecs[i], vecs[i + 1]
            dE_map = compute_deltaE_map_from_features(f1, f2)
            dE_mean = float(np.nanmean(dE_map))
            deltas.append(dE_mean)
            all_deltaE.append(dE_mean)
            if rep_map is None:
                rep_map = dE_map

        per_class_deltaE[cls] = float(np.nanmean(deltas))

    all_deltaE = np.array(all_deltaE) if len(all_deltaE) > 0 else np.array([0.0])
    mean_deltaE = float(np.nanmean(all_deltaE))
    quantiles = np.nanpercentile(all_deltaE, [25, 50, 75, 90, 99])

    if rep_map is None:
        rep_map = np.zeros((8, 8))

    return {
        "mean": mean_deltaE,
        "quantiles": quantiles,
        "per_class": per_class_deltaE,
        "rep_map": rep_map,
        "samples": all_deltaE.tolist()
    }


def run_ssim_roundtrip(color_space, loader, sample_size=64):
    """
    Wrapper around RGB → color_space → RGB SSIM.

    Returns:
      {
        "mean": float,
        "samples": list[float]
      }
    """
    from skimage.metrics import structural_similarity as ssim

    ssim_vals = []
    total = 0

    for data, _ in loader:
        remaining = sample_size - total
        if remaining <= 0:
            break

        batch = data[:remaining]
        total += batch.size(0)

        rgb = batch.detach().cpu().float()
        cs = rgb_to_color_batch_cpu(rgb, color_space)
        rgb_round = color_to_rgb_batch_cpu(cs, color_space)

        for a, b in zip(rgb, rgb_round):
            img1 = np.transpose(a.numpy(), (1, 2, 0))
            img2 = np.transpose(b.numpy(), (1, 2, 0))
            img1 = np.clip(img1, 0, 1)
            img2 = np.clip(img2, 0, 1)
            val = ssim(img1, img2, channel_axis=2, data_range=1.0)
            ssim_vals.append(val)

    mean_ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    return {
        "mean": mean_ssim,
        "samples": ssim_vals
    }

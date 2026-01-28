# analysis/gradcam.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from utils import DEVICE
from colorspace import convert_and_normalize
from models import GradCAM


def run_gradcam_maps_for_sample(models_by_space, loader, color_spaces,
                                idx_in_batch=0):
    """
    Returns Grad-CAM maps for a single sample across color spaces.

    Returns:
      dict[space] -> {
          "cam": 2D np.array in [0,1],
          "pred": int,
          "true": int
      }
    """
    data, target = next(iter(loader))
    img = data[idx_in_batch:idx_in_batch + 1].to(DEVICE)
    true_label = int(target[idx_in_batch].item())

    cams_out = {}

    for space in color_spaces:
        model = models_by_space[space]
        model.to(DEVICE)
        model.eval()

        last_conv = model.get_last_conv_layer()
        gradcam = GradCAM(model, last_conv)

        img_cs = convert_and_normalize(img, space)
        with torch.no_grad():
            logits = model(img_cs)
        pred_class = logits.argmax(dim=1)

        cam = gradcam.generate(img_cs, class_idx=pred_class)[0]
        H, W = img.shape[2], img.shape[3]
        cam_resized = cv2.resize(cam, (W, H))

        cams_out[space] = {
            "cam": cam_resized,
            "pred": int(pred_class.item()),
            "true": true_label
        }

    return cams_out


def run_gradcam_iou_against_rgb(models_by_space, loader, color_spaces,
                                n_batches=5, thresh="mean"):
    """
    Compute IoU between CAMs of RGB model and others.

    Returns:
      dict[space] -> IoU float  (space != "rgb")
    """
    def gradcam_map_for_batch(model, space, data):
        model.eval()
        last_conv = model.get_last_conv_layer()
        gc = GradCAM(model, last_conv)
        data = data.to(DEVICE)
        data_cs = convert_and_normalize(data, space)
        logits = model(data_cs)
        preds = logits.argmax(dim=1)
        cams = gc.generate(data_cs, class_idx=preds)
        return cams

    ious = {s: [] for s in color_spaces if s != "rgb"}
    it = iter(loader)

    for _ in range(n_batches):
        try:
            data, _ = next(it)
        except StopIteration:
            break

        cams_rgb = gradcam_map_for_batch(models_by_space["rgb"], "rgb", data)
        for space in color_spaces:
            if space == "rgb":
                continue
            cams_sp = gradcam_map_for_batch(models_by_space[space], space, data)
            for cr, cs in zip(cams_rgb, cams_sp):
                if thresh == "mean":
                    t1, t2 = cr.mean(), cs.mean()
                    m1 = cr > t1
                    m2 = cs > t2
                else:
                    m1 = cr > 0.5
                    m2 = cs > 0.5
                inter = np.logical_and(m1, m2).sum()
                union = np.logical_or(m1, m2).sum() + 1e-8
                ious[space].append(float(inter / union))

    return {s: float(np.mean(v)) if v else 0.0 for s, v in ious.items()}


def run_attention_entropy(models_by_space, loader, color_spaces, n_batches=5):
    """
    Computes entropy of Grad-CAM maps per color space.

    Returns:
      dict[space] -> float (mean entropy)
    """
    def gradcam_map_for_batch(model, space, data):
        model.eval()
        last_conv = model.get_last_conv_layer()
        gc = GradCAM(model, last_conv)
        data = data.to(DEVICE)
        data_cs = convert_and_normalize(data, space)
        logits = model(data_cs)
        preds = logits.argmax(dim=1)
        cams = gc.generate(data_cs, class_idx=preds)
        return cams

    ent = {s: [] for s in color_spaces}
    it = iter(loader)

    for _ in range(n_batches):
        try:
            data, _ = next(it)
        except StopIteration:
            break

        for space in color_spaces:
            cams = gradcam_map_for_batch(models_by_space[space], space, data)
            for cam in cams:
                p = cam.flatten()
                p = p / (p.sum() + 1e-8)
                h = -np.sum(p * np.log(p + 1e-8))
                ent[space].append(float(h))

    return {s: float(np.mean(v)) if v else 0.0 for s, v in ent.items()}

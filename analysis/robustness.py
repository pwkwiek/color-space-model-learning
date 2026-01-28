# analysis/robustness.py
import torch
import numpy as np

from utils import DEVICE
from colorspace import rgb_to_color_batch_cpu, color_to_rgb_batch_cpu, convert_and_normalize


def _robustness_shift_single(model, loader, color_space,
                             shift_type="hue", amount=0.1, max_batches=20):
    model.eval()
    correct_base = 0
    correct_shift = 0
    total = 0
    it = iter(loader)

    for _ in range(max_batches):
        try:
            data, target = next(it)
        except StopIteration:
            break

        data, target = data.to(DEVICE), target.to(DEVICE)

        base_cs = convert_and_normalize(data, color_space)
        with torch.no_grad():
            logits = model(base_cs)
        preds = logits.argmax(1)
        correct_base += (preds == target).sum().item()

        rgb = data.clone().detach().cpu()
        if shift_type in ["hue", "sat"]:
            hsv = rgb_to_color_batch_cpu(rgb, "hsv")
            if shift_type == "hue":
                hsv[:, 0] = (hsv[:, 0] + amount) % 1.0
            else:
                hsv[:, 1] = np.clip(hsv[:, 1] + amount, 0, 1)
            rgb_shift = color_to_rgb_batch_cpu(hsv, "hsv").to(DEVICE)
        else:  # brightness
            rgb_shift = torch.clamp(rgb + amount, 0, 1).to(DEVICE)

        shift_cs = convert_and_normalize(rgb_shift, color_space)
        with torch.no_grad():
            logits_shift = model(shift_cs)
        preds_shift = logits_shift.argmax(1)
        correct_shift += (preds_shift == target).sum().item()

        total += target.size(0)

    acc_base = correct_base / max(total, 1)
    acc_shift = correct_shift / max(total, 1)
    return acc_shift / (acc_base + 1e-8)


def run_color_robustness(models_by_space, loader, color_spaces,
                         amount=0.1, max_batches=20):
    """
    For each color space, returns accuracy ratio after perturbations:
      - hue
      - saturation
      - brightness

    Output:
      dict[space] -> {"hue": float, "sat": float, "bright": float}
    """
    out = {}
    for space in color_spaces:
        model = models_by_space[space]
        out[space] = {
            "hue": _robustness_shift_single(model, loader, space, "hue", amount, max_batches),
            "sat": _robustness_shift_single(model, loader, space, "sat", amount, max_batches),
            "bright": _robustness_shift_single(model, loader, space, "brightness", amount, max_batches),
        }
    return out

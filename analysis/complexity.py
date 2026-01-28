# analysis/complexity.py
import numpy as np
import torch
import torch.nn as nn

from utils import DEVICE
from colorspace import convert_and_normalize


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_activation_count(model, loader, color_space="rgb"):
    """
    Rough estimate of total activation count for one forward pass.
    """
    model.eval()
    acts = []

    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            acts.append(out.numel())

    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
                          nn.AdaptiveAvgPool2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook_fn))

    example = next(iter(loader))[0]
    H, W = example.shape[2], example.shape[3]
    x = torch.randn(1, 3, H, W).to(DEVICE)
    x = convert_and_normalize(x, color_space)

    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    return int(np.sum(acts))


def average_feature_variance(first_layer_act):
    """
    first_layer_act: [B,C,H,W] tensor
    """
    f = first_layer_act.contiguous().reshape(first_layer_act.size(0), -1).cpu().numpy()
    return float(np.var(f, axis=1).mean())


def run_complexity_metrics(models_by_space, results, train_loader, color_spaces):
    """
    Uses:
      - count_params
      - estimate_activation_count
      - average_feature_variance
      - epoch_time_mean from results
      - convergence_rate from loss[0] - loss[-1]

    Returns:
      dict[space] -> { "params", "activations", "feat_var",
                       "epoch_time_mean", "convergence_rate" }
    """
    out = {}
    for space in color_spaces:
        model = models_by_space[space]
        res = results[space]

        feat_var = 0.0
        if res["first_layer_activation"] is not None:
            feat_var = average_feature_variance(res["first_layer_activation"])

        metrics = {
            "params": count_params(model),
            "activations": estimate_activation_count(model, train_loader, color_space=space),
            "feat_var": feat_var,
            "epoch_time_mean": float(np.mean(res["epoch_times"])),
            "convergence_rate": (res["loss"][0] - res["loss"][-1]) / max(len(res["loss"]), 1),
        }
        out[space] = metrics

    return out

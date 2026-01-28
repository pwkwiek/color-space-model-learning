# analysis/pruning.py
from copy import deepcopy

from training import evaluate_model


def evaluate_with_pruned_first_conv(model, loader, color_space, prune_ratio=0.5):
    """
    Prunes a fraction of output channels of first conv and evaluates accuracy.
    Returns pruned_acc.
    """
    # work on a copy to avoid side effects
    model = deepcopy(model)

    conv = model.get_first_conv()
    w_orig = conv.weight.data.clone()

    out_ch = w_orig.size(0)
    k = int(out_ch * prune_ratio)

    import numpy as np
    idx = np.random.choice(out_ch, k, replace=False)
    conv.weight.data[idx] = 0.0

    acc = evaluate_model(model, loader, color_space)
    return acc


def run_pruning_sensitivity(models_by_space, results, test_loader, color_spaces,
                            prune_ratios=(0.25, 0.5)):
    """
    For each color space, compute accuracy drop for prune_ratios.
    Uses results[space]["test_acc_last"] as baseline.

    Returns:
      dict[space] -> {f"drop{int(r*100)}": float}
    """
    pruning_sensitivity = {space: {} for space in color_spaces}

    for space in color_spaces:
        base_acc = results[space]["test_acc"][-1]
        for ratio in prune_ratios:
            pruned_acc = evaluate_with_pruned_first_conv(
                models_by_space[space], test_loader, space, prune_ratio=ratio)
            pruning_sensitivity[space][f"drop{int(ratio * 100)}"] = base_acc - pruned_acc

    return pruning_sensitivity

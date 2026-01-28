# analysis/precompute.py

import torch
import numpy as np

from analysis.gradcam import (
    run_gradcam_iou_against_rgb,
    run_attention_entropy,
)
from analysis.embeddings import (
    run_embedding_stability,
    run_class_color_texture_complexity,
)
from analysis.kernels import run_kernel_stats
from analysis.robustness import run_color_robustness
from analysis.complexity import run_complexity_metrics
from analysis.pruning import run_pruning_sensitivity
from analysis.perceptual import run_deltaE, run_ssim_roundtrip


# -------------------------------------------------------------------------
# Confusion Matrix + Per-Class Accuracy
# -------------------------------------------------------------------------
def compute_confusions_and_per_class_acc(
    models_by_space, loader, class_names, color_spaces, convert_fn
):
    confusions = {}
    per_class_acc = {}

    num_classes = len(class_names)

    for space in color_spaces:
        print(f" → Confusion matrix for {space.upper()}")

        model = models_by_space[space]
        model.eval()

        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        correct_per_class = np.zeros(num_classes)
        total_per_class = np.zeros(num_classes)

        with torch.no_grad():
            device = next(model.parameters()).device

            for x, y in loader:
                x, y = x.to(device), y.to(device)
                x_cs = convert_fn(x, space)

                logits = model(x_cs)
                preds = logits.argmax(dim=1)

                for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
                    cm[t, p] += 1
                    total_per_class[t] += 1
                    if t == p:
                        correct_per_class[t] += 1

        acc = np.divide(correct_per_class, np.maximum(total_per_class, 1))

        confusions[space] = cm
        per_class_acc[space] = acc

    return confusions, per_class_acc


# -------------------------------------------------------------------------
# Feature Redundancy From Stored Activations
# -------------------------------------------------------------------------
def compute_feature_redundancy_from_activations(results, color_spaces, thresh=0.8):
    redundancy = {}

    for space in color_spaces:
        acts = results[space].get("first_layer_activation", None)
        if acts is None:
            redundancy[space] = float("nan")
            continue

        feats = acts.detach().cpu()        # (B, C, H, W)
        B, C, H, W = feats.shape

        feats_flat = feats.view(B, C, -1).mean(dim=2).numpy()
        feats_flat = (feats_flat - feats_flat.mean(axis=0)) / (
            feats_flat.std(axis=0) + 1e-8
        )

        corr = np.corrcoef(feats_flat.T)
        off_diag = corr[np.triu_indices(C, k=1)]

        redundancy[space] = float((np.abs(off_diag) > thresh).mean())

    return redundancy


# -------------------------------------------------------------------------
# FLOPs
# -------------------------------------------------------------------------
def _compute_flops_first_conv_for_space(model, train_loader):
    """
    FLOPs = H * W * C_in * C_out * K_h * K_w
    """
    conv = model.get_first_conv()
    weight = conv.weight
    C_out = weight.size(0)
    C_in = weight.size(1)
    k_h, k_w = weight.size(2), weight.size(3)

    example, _ = next(iter(train_loader))
    _, _, H, W = example.shape

    flops = H * W * C_in * C_out * (k_h * k_w)
    return float(flops)


# -------------------------------------------------------------------------
# Master Function: Compute All Analysis Outputs
# -------------------------------------------------------------------------
def compute_all_analysis_outputs(
    models_by_space,
    results,
    train_loader,
    test_loader,
    class_names,
    color_spaces,
    convert_fn,
    prune_ratios=(0.25, 0.5),
):
    analysis_outputs = {}

    # 1. Confusions
    print("\n===  Computing Confusion Matrices & Per-Class Accuracy ===")
    confusions, per_class_acc = compute_confusions_and_per_class_acc(
        models_by_space, test_loader, class_names, color_spaces, convert_fn
    )
    analysis_outputs["confusions"] = confusions
    analysis_outputs["per_class_acc"] = per_class_acc

    # 2. Kernel stats
    print("\n===  Kernel Statistics ===")
    kernel_stats_by_space, kernel_similarity_to_rgb = run_kernel_stats(
        models_by_space, color_spaces
    )
    analysis_outputs["kernel_stats"] = kernel_stats_by_space
    analysis_outputs["kernel_similarity"] = kernel_similarity_to_rgb

    # 3. Feature redundancy
    print("\n===  Feature Redundancy ===")
    feature_redundancy = compute_feature_redundancy_from_activations(
        results, color_spaces
    )
    analysis_outputs["feature_redundancy"] = feature_redundancy

    # 4. Grad-CAM + entropy
    print("\n===  Grad-CAM IoU vs RGB ===")
    gcam_iou = run_gradcam_iou_against_rgb(
        models_by_space, test_loader, color_spaces
    )
    analysis_outputs["gcam_iou"] = gcam_iou

    print("\n===  Attention Entropy ===")
    attention_entropy = run_attention_entropy(
        models_by_space, test_loader, color_spaces
    )
    analysis_outputs["attention_entropy"] = attention_entropy

    # 5. Embedding Stability
    print("\n===  Embedding Stability ===")
    embedding_stability = run_embedding_stability(
        models_by_space, test_loader, class_names, color_spaces
    )
    analysis_outputs["embedding_stability"] = embedding_stability

    # 6. Color–Texture Complexity
    print("\n===  Color–Texture Complexity ===")
    color_cx, texture_cx = run_class_color_texture_complexity(
        test_loader, class_names
    )

    color_complexity_all = {space: dict(color_cx) for space in color_spaces}
    texture_complexity_all = {space: dict(texture_cx) for space in color_spaces}

    analysis_outputs["color_complexity"] = color_complexity_all
    analysis_outputs["texture_complexity"] = texture_complexity_all

    # 7. Robustness
    print("\n===  Color Perturbation Robustness ===")
    color_sensitivity = run_color_robustness(
        models_by_space, test_loader, color_spaces
    )
    analysis_outputs["color_sensitivity"] = color_sensitivity

    # 8. Complexity metrics
    print("\n===  Network Complexity Metrics ===")
    complexity_metrics = run_complexity_metrics(
        models_by_space, results, train_loader, color_spaces
    )
    analysis_outputs["complexity_metrics"] = complexity_metrics

    # 9. Pruning
    print("\n===  Pruning Sensitivity ===")
    pruning_sensitivity = run_pruning_sensitivity(
        models_by_space, results, test_loader, color_spaces,
        prune_ratios=prune_ratios
    )
    analysis_outputs["pruning_sensitivity"] = pruning_sensitivity

    # 10. FLOPs
    print("\n===  FLOPs of First Conv ===")
    flops_by_space = {}
    for space in color_spaces:
        model = models_by_space[space]
        flops_val = _compute_flops_first_conv_for_space(model, train_loader)
        flops_by_space[space] = flops_val
        results[space]["flops"] = flops_val

    analysis_outputs["flops"] = flops_by_space

    # 11. ΔE + SSIM
    print("\n===  ΔE + SSIM Round-Trip Metrics ===")

    deltaE_stats_all = {}
    ssim_stats_all = {}

    for space in color_spaces:
        print(f" → {space.upper()}")

        dE = run_deltaE(
            model=models_by_space[space],
            loader=test_loader,
            class_names=class_names,
            color_space=space,
        )
        ssim = run_ssim_roundtrip(
            color_space=space,
            loader=test_loader,
            sample_size=64,
        )

        deltaE_stats_all[space] = dE
        ssim_stats_all[space] = ssim

        results[space]["deltaE_mean"] = dE["mean"]
        results[space]["deltaE_quantiles"] = dE["quantiles"]
        results[space]["deltaE_class_stats"] = dE["per_class"]
        results[space]["deltaE_samples"] = dE["samples"]
        results[space]["deltaE_rep_map"] = dE["rep_map"]


        results[space]["ssim"] = ssim["mean"]
        results[space]["ssim_samples"] = ssim["samples"]

    analysis_outputs["deltaE_stats"] = deltaE_stats_all
    analysis_outputs["ssim_stats"] = ssim_stats_all

    return analysis_outputs

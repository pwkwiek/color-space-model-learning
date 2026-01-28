# analysis/__init__.py

from .perceptual import run_deltaE, run_ssim_roundtrip
from .gradcam import (
    run_gradcam_maps_for_sample,
    run_gradcam_iou_against_rgb,
    run_attention_entropy,
)
from .embeddings import (
    run_embedding_stability,
    run_class_color_texture_complexity,
)
from .kernels import (
    run_kernel_stats,
    run_feature_redundancy_from_corr,
)
from .robustness import run_color_robustness
from .complexity import run_complexity_metrics
from .pruning import run_pruning_sensitivity
from .statistics import (
    run_anova_and_wilcoxon,
    run_correlations,
    bootstrap_ci,
    run_accuracy_ci,
)
from .summary import build_summary_table

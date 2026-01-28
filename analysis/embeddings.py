# analysis/embeddings.py

import numpy as np
import torch
import numpy.linalg as LA

from utils import DEVICE
from colorspace import convert_and_normalize


# -------------------------------------------------------------
# EMBEDDING STABILITY
# -------------------------------------------------------------
def run_embedding_stability(models_by_space, loader, class_names,
                            color_spaces, max_per_class=30):
    """
    Computes cosine similarity between embeddings of each color space
    and RGB embeddings, per class.

    Returns:
      dict[space != "rgb"] -> {class_name: mean_cosine}
    """

    # Reference: RGB
    rgb_model = models_by_space["rgb"]
    rgb_model.eval()
    rgb_embeds = {c: [] for c in class_names}

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE)
            embeds = rgb_model.extract_embeddings(data, color_space="rgb").cpu().numpy()
            for vec, lab in zip(embeds, targets.numpy()):
                cls = class_names[int(lab)]
                if len(rgb_embeds[cls]) < max_per_class:
                    rgb_embeds[cls].append(vec)

            # Stop early when enough samples collected
            if all(len(rgb_embeds[c]) >= max_per_class for c in class_names):
                break

    # Compare each space to RGB
    stability = {
        space: {c: [] for c in class_names}
        for space in color_spaces
        if space != "rgb"
    }

    for space in color_spaces:
        if space == "rgb":
            continue

        model = models_by_space[space]
        model.eval()

        with torch.no_grad():
            for data, targets in loader:
                data = data.to(DEVICE)
                embeds = model.extract_embeddings(data, color_space=space).cpu().numpy()

                for vec, lab in zip(embeds, targets.numpy()):
                    cls = class_names[int(lab)]

                    if len(stability[space][cls]) >= max_per_class:
                        continue
                    if len(rgb_embeds[cls]) == 0:
                        continue

                    rgb_mean = np.mean(rgb_embeds[cls], axis=0)
                    cos_sim = float(
                        np.dot(vec, rgb_mean)
                        / ((LA.norm(vec) * LA.norm(rgb_mean)) + 1e-8)
                    )
                    stability[space][cls].append(cos_sim)

                if all(len(stability[space][c]) >= max_per_class for c in class_names):
                    break

    # Convert lists → means
    stability_mean = {
        space: {c: float(np.mean(v)) if len(v) > 0 else 0.0
                for c, v in cls_dict.items()}
        for space, cls_dict in stability.items()
    }

    return stability_mean


# -------------------------------------------------------------
# COLOR–TEXTURE COMPLEXITY (PER CLASS)
# -------------------------------------------------------------
def run_class_color_texture_complexity(loader, class_names, max_batches=50):
    """
    Computes per-class:
      - color complexity:   mean std(channel) across image
      - texture complexity: gradient magnitude on grayscale image

    Returns TWO dictionaries:
      color_complexity:  {class: float}
      texture_complexity: {class: float}
    """

    color_complexity = {c: [] for c in class_names}
    texture_complexity = {c: [] for c in class_names}

    for bi, (data, targets) in enumerate(loader):
        if bi >= max_batches:
            break

        imgs = data.numpy()  # shape: (B,3,H,W)

        for img, label in zip(imgs, targets.numpy()):
            cls = class_names[int(label)]

            # ---- Color complexity ----
            col_std = img.reshape(3, -1).std(axis=1).mean()

            # ---- Texture complexity ----
            gray = np.mean(img, axis=0)
            gx = np.gradient(gray, axis=0)
            gy = np.gradient(gray, axis=1)
            tex_mag = np.sqrt(gx**2 + gy**2).mean()

            color_complexity[cls].append(col_std)
            texture_complexity[cls].append(tex_mag)

    # Convert lists to scalar means
    color_complexity = {
        c: float(np.mean(v)) if len(v) else float("nan")
        for c, v in color_complexity.items()
    }
    texture_complexity = {
        c: float(np.mean(v)) if len(v) else float("nan")
        for c, v in texture_complexity.items()
    }

    return color_complexity, texture_complexity

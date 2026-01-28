# Color Matters
Color-Space Effects on CNN Learning and Robustness

RGB is everywhere — but rarely questioned.

This project investigates how **color-space selection** influences learning dynamics, internal representations, and robustness in convolutional neural networks, with a particular focus on **deep architectures** and **medical imaging**.

---

## What This Study Explores

Color representation is a fundamental yet often underexamined engineering choice in CNNs.  
While RGB is the default, alternative color spaces separate luminance and chromatic information in ways that can significantly affect optimization and feature extraction.

This work systematically evaluates **six color spaces** across **two image domains** and **two CNN architectures**.

---

## Experimental Setup

### Datasets
- **CIFAR-10** — natural images
- **ISBI 2018** — dermoscopic medical images

### Color Spaces
- RGB  
- LAB  
- HSV  
- YUV  
- YCrCb  
- XYZ  

### Architectures
- Shallow CNN (lightweight)
- Deep **ResNet**

### Training Regimes
- Short training (early learning dynamics)
- Extended training (converged representations)

---

## Evaluation Criteria

- Classification accuracy
- Convergence speed and stability
- Feature-map and kernel statistics
- Embedding redundancy and stability
- Robustness to color perturbations
- Perceptual color-difference measures (ΔE)
- Grad-CAM–based spatial attention analysis

---

## Key Findings

- **Shallow CNNs** are largely insensitive to color-space choice  
  → stable convergence and similar representations across encodings.
- **Deep networks are highly sensitive** to color representation.
- Perceptually nonlinear spaces (especially **LAB** and **YCrCb**):
  - slow optimization
  - destabilize embeddings
  - increase representational redundancy
  - reduce robustness to color perturbations
- These effects are **stronger in medical imaging**, where subtle color variations carry diagnostic meaning.
- Color-space choice directly affects **where and how networks attend spatially** (Grad-CAM).

---

## Takeaway

Color-space selection is **not a neutral preprocessing step**.

It is a **non-trivial engineering decision**, particularly for:
- deep CNNs
- medical imaging tasks
- settings where robustness and interpretability matter

---

## Notes

- Focused on **analysis and insight**, not color normalization tricks
- No data augmentation beyond color-space conversion
- Designed to expose *why* models behave differently — not just *how well*

# colorspace.py
import numpy as np
import torch
import cv2
from skimage import color
from skimage.metrics import structural_similarity as ssim

# ImageNet normalization (shared for ALL color spaces)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ------------------------------------------------------------
# Basic CPU converters
# ------------------------------------------------------------
def rgb_to_color_batch_cpu(batch_rgb, color_space="lab"):
    """
    RGB batch on CPU -> chosen color space (no normalization).
    batch_rgb: [B,3,H,W] on CPU in [0,1]
    """
    assert batch_rgb.device.type == "cpu"
    batch = batch_rgb.permute(0, 2, 3, 1).numpy()

    converted = []
    for img in batch:
        if color_space == "lab":
            img_safe = np.clip(img, 1e-6, 1 - 1e-6)
            out = color.rgb2lab(img_safe)
        elif color_space == "hsv":
            out = color.rgb2hsv(img)
        elif color_space == "yuv":
            out = color.rgb2yuv(img)
        elif color_space == "ycrcb":
            out = cv2.cvtColor((img * 255).astype(np.uint8),
                               cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
        elif color_space == "xyz":
            out = color.rgb2xyz(img)
        elif color_space == "rgb":
            out = img
        else:
            raise ValueError(f"Unknown color space: {color_space}")
        converted.append(out.astype(np.float32))

    converted = np.stack(converted, axis=0)
    return torch.tensor(converted, dtype=torch.float32).permute(0, 3, 1, 2)


def color_to_rgb_batch_cpu(batch, color_space="lab"):
    """
    Inverse conversion from color space to RGB.
    batch: [B,C,H,W] on CPU
    """
    assert batch.device.type == "cpu"
    batch_np = batch.permute(0, 2, 3, 1).numpy()
    converted = []

    for img in batch_np:
        if color_space == "lab":
            out = color.lab2rgb(img)
        elif color_space == "hsv":
            out = color.hsv2rgb(img)
        elif color_space == "yuv":
            out = color.yuv2rgb(img)
        elif color_space == "ycrcb":
            out = cv2.cvtColor((img * 255).astype(np.uint8),
                               cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
        elif color_space == "xyz":
            out = color.xyz2rgb(img)
        elif color_space == "rgb":
            out = img
        else:
            raise ValueError(f"Unknown color space: {color_space}")
        converted.append(np.clip(out, 0, 1).astype(np.float32))

    converted = np.stack(converted, axis=0)
    return torch.tensor(converted, dtype=torch.float32).permute(0, 3, 1, 2)


# ------------------------------------------------------------
# N1: unified conversion to other spaces then ImageNet norm
# ------------------------------------------------------------
def convert_rgb_np(img, color_space="rgb"):
    """
    img: H x W x 3, float32 in [0,1]
    → H x W x 3, float32, roughly [0,1] after scaling
    """
    if color_space == "rgb":
        return img

    elif color_space == "lab":
        img_safe = np.clip(img, 1e-6, 1 - 1e-6)
        lab = color.rgb2lab(img_safe).astype(np.float32)
        # L [0,100], a,b [-128,128] → squash to ~[0,1]
        lab[..., 0] /= 100.0
        lab[..., 1] = (lab[..., 1] + 128.0) / 255.0
        lab[..., 2] = (lab[..., 2] + 128.0) / 255.0
        return lab

    elif color_space == "hsv":
        return color.rgb2hsv(img).astype(np.float32)

    elif color_space == "yuv":
        return color.rgb2yuv(img).astype(np.float32)

    elif color_space == "ycrcb":
        out = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        return (out.astype(np.float32) / 255.0)

    elif color_space == "xyz":
        xyz = color.rgb2xyz(img).astype(np.float32)
        xyz[..., 0] /= 0.95047
        xyz[..., 1] /= 1.0
        xyz[..., 2] /= 1.08883
        return xyz

    else:
        raise ValueError(f"Unknown color space: {color_space}")


def rgb_to_colorspace_batch(batch_rgb, color_space="rgb"):
    """
    batch_rgb: [B,3,H,W] float32 in [0,1]
    → [B,3,H,W] float32
    """
    batch = batch_rgb.detach().cpu().permute(0, 2, 3, 1).numpy()
    converted = []
    for img in batch:
        converted.append(convert_rgb_np(img, color_space))
    converted = np.stack(converted, axis=0)
    return torch.tensor(converted, dtype=torch.float32).permute(0, 3, 1, 2)


def apply_imagenet_norm(batch):
    return (batch - IMAGENET_MEAN.to(batch.device)) / IMAGENET_STD.to(batch.device)


def convert_and_normalize(batch_rgb, color_space="rgb"):
    """
    Main entry:
    - input: RGB [B,3,H,W] in [0,1]
    - convert to color_space
    - apply ImageNet normalization (same mean/std for all spaces)
    """
    batch_rgb = batch_rgb.clamp(0, 1)
    cs = rgb_to_colorspace_batch(batch_rgb, color_space)
    cs = cs.to(batch_rgb.device)
    return apply_imagenet_norm(cs)


# ------------------------------------------------------------
# SSIM helper
# ------------------------------------------------------------
def compute_ssim_between_rgb_and_colorspace(color_space, loader,
                                            sample_size=64, return_samples=False):
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
    return (mean_ssim, ssim_vals) if return_samples else mean_ssim

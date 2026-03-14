"""
Utility helpers for NR-IQA with Activation Gated MLP.
Contains loss functions, evaluation metrics, GradCAM overlay, and text constants.
Author: Ankit Yadav
"""
import torch
import numpy as np
import cv2
from scipy.stats import spearmanr
import scipy.stats


# ---------------------------------------------------------------------------
# Text / prompt constants
# ---------------------------------------------------------------------------

scenes = (
    "animal, cityscape, human, indoor, landscape, "
    "night, plant, still life, food, water, other"
)

BAD_QUALITY_PROMPT = (
    "A photograph with severe JPEG2000 compression artifacts, heavy JPEG "
    "blocking, intense motion blur, overwhelming white noise, pronounced "
    "pixelation and color quantization, very low contrast, severe under- "
    "and over-exposure, and overall poor image quality and other poor "
    "image quality properties."
)

Text_Template_baseline = (
    "A high-resolution photo with visible distortions such as "
    "{distortion_type} focus on its visual quality."
)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def margin_loss(y: torch.Tensor, yp: torch.Tensor,
                lambda_: float = 0.25) -> torch.Tensor:
    """Pair-wise margin ranking loss for quality scores.

    For every ordered pair (i, j) where y_i != y_j the loss penalises
    predicted differences that disagree with the ground-truth ordering
    beyond a data-driven margin ``lambda_ * std(y)``.

    Args:
        y:  ground-truth scores  (n,)
        yp: predicted scores     (n,)
        lambda_: margin scaling factor (typically 0.25).

    Returns:
        Scalar loss tensor.
    """
    n = y.size(0)
    sigma_y = torch.std(y, unbiased=False)
    m = lambda_ * sigma_y

    loss = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign = torch.sign(y[i] - y[j])
            if sign != 0:
                diff = -(sign * (yp[i] - yp[j])) + m
                loss += torch.clamp(diff, min=0.0)
                count += 1

    if count == 0:
        return torch.tensor(0.0, device=y.device)
    return (2.0 / (n * (n - 1))) * loss


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

class metric:
    """Compute SRCC and PLCC between ground-truth and predicted scores."""

    def __init__(self):
        self.result: dict = {}

    def calcuate_srcc(self, tensor1, tensor2):
        rank1 = scipy.stats.rankdata(tensor1)
        rank2 = scipy.stats.rankdata(tensor2)
        srcc, _ = spearmanr(rank1, rank2)
        self.result["SRCC"] = float(srcc)
        return srcc

    def calculate_plcc(self, tensor1, tensor2):
        t1 = torch.from_numpy(tensor1)
        t2 = torch.from_numpy(tensor2)
        x_mean, y_mean = t1.mean(), t2.mean()
        numerator = ((t1 - x_mean) * (t2 - y_mean)).sum()
        x_var = ((t1 - x_mean) ** 2).sum()
        y_var = ((t2 - y_mean) ** 2).sum()
        plcc = numerator / torch.sqrt(x_var * y_var)
        self.result["PLCC"] = float(plcc)
        return plcc


# ---------------------------------------------------------------------------
# GradCAM overlay visualisation
# ---------------------------------------------------------------------------

def Overlay(img, heatmap, alpha=(0.6, 0.4)):
    """Overlay a GradCAM heatmap on a (possibly normalised) image tensor.

    Args:
        img: ``(C, H, W)`` tensor (ImageNet-normalised or [0, 1]).
        heatmap: ``(H, W)`` numpy array in [0, 1].
        alpha: ``(image_weight, heatmap_weight)`` for ``cv2.addWeighted``.

    Returns:
        (overlay_bgr, original_bgr) pair of uint8 numpy arrays.
    """
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    unnormed = torch.clamp(img * std + mean, 0.0, 1.0)
    rgb = unnormed.cpu().permute(1, 2, 0).numpy()

    heat_u8 = (heatmap * 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    rgb_u8 = (rgb * 255).astype(np.uint8)
    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(bgr_u8, alpha[0], color_heatmap, alpha[1], 0)
    return overlay, bgr_u8

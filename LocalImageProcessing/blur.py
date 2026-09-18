"""Localized Gaussian-blur fallback — explicit partial/review reconstruction."""

from __future__ import annotations

import numpy as np


def apply_feathered_blur(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    sigma: float = 12.0,
    feather_sigma: float = 5.0,
) -> np.ndarray:
    """Blur only masked pixels with a feathered boundary."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("OpenCV is required for blur_fallback reconstruction")

    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel array")
    if not np.any(mask > 0):
        return image.copy()

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather_sigma)
    alpha = np.clip(alpha / 255.0, 0.0, 1.0)

    result = image.astype(np.float32).copy()
    if image.ndim == 2:
        result = (1.0 - alpha) * result + alpha * blurred.astype(np.float32)
        return np.clip(result, 0, 255).astype(image.dtype)

    for channel in range(image.shape[2]):
        result[:, :, channel] = (
            (1.0 - alpha) * image[:, :, channel]
            + alpha * blurred[:, :, channel]
        )
    return np.clip(result, 0, 255).astype(image.dtype)

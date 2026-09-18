"""OpenCV inpainting helpers."""

from __future__ import annotations

import numpy as np


def inpaint_navier_stokes(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 5,
) -> np.ndarray:
    try:
        import cv2

        return cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)
    except ImportError:
        return image.copy()

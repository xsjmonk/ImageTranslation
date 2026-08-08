"""Inpainting – text removal via OpenCV inpainting or fallback methods."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def inpaint_telea(image: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Remove text using Telea inpainting algorithm.

    Args:
        image: BGR source image.
        mask: uint8 mask where 255 = text to inpaint.
        radius: Inpainting radius.

    Returns:
        Inpainted image.
    """
    try:
        import cv2
        return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
    except ImportError:
        logger.warning("OpenCV not available; returning original image.")
        return image


def inpaint_navier_stokes(
    image: np.ndarray, mask: np.ndarray, radius: int = 5
) -> np.ndarray:
    """Remove text using Navier-Stokes inpainting."""
    try:
        import cv2
        return cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)
    except ImportError:
        logger.warning("OpenCV not available; returning original image.")
        return image

"""Text mask generation – creates binary masks from TextRegion polygons."""

from __future__ import annotations

from typing import List

import numpy as np

from ..models.text_region import TextAction, TextRegion


def create_text_mask(
    image: np.ndarray,
    regions: List[TextRegion],
    expansion_pixels: int = 3,
) -> np.ndarray:
    """Create a binary mask from text region polygons.

    Only regions with action 'translate' or 'remove' are included.

    Args:
        image: Source image for dimension reference.
        regions: Classified text regions.
        expansion_pixels: Extra pixels to expand mask around text.

    Returns:
        uint8 mask (height, width) where 255 = area to remove.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for region in regions:
        if region.action not in (TextAction.translate, TextAction.remove):
            continue

        polygon = np.array(region.polygon, dtype=np.int32)
        # Ensure polygon is within bounds
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        # Fill polygon
        cv2_fill_poly(mask, polygon)

        # Expand mask if requested
        if expansion_pixels > 0:
            _expand_mask_region(mask, polygon, expansion_pixels)

    return mask


def cv2_fill_poly(mask: np.ndarray, polygon: np.ndarray) -> None:
    """Fill a polygon on the mask using OpenCV."""
    try:
        import cv2
        cv2.fillPoly(mask, [polygon], 255)
    except ImportError:
        _fallback_fill_poly(mask, polygon)


def _fallback_fill_poly(mask: np.ndarray, polygon: np.ndarray) -> None:
    """Pure-numpy polygon fill fallback (slower)."""
    from PIL import Image, ImageDraw
    h, w = mask.shape
    pil_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(pil_img)
    draw.polygon([tuple(p) for p in polygon], fill=255)
    mask[:] = np.maximum(mask, np.array(pil_img))


def _expand_mask_region(
    mask: np.ndarray, polygon: np.ndarray, pixels: int
) -> None:
    """Dilate the mask region around a polygon."""
    try:
        import cv2
        # Create a small ROI for dilation
        x, y, pw, ph = cv2.boundingRect(polygon)
        x = max(0, x - pixels)
        y = max(0, y - pixels)
        pw = min(mask.shape[1] - x, pw + 2 * pixels)
        ph = min(mask.shape[0] - y, ph + 2 * pixels)

        if pw > 0 and ph > 0:
            roi = mask[y:y + ph, x:x + pw]
            kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
            dilated = cv2.dilate(roi, kernel, iterations=1)
            mask[y:y + ph, x:x + pw] = np.maximum(roi, dilated)
    except ImportError:
        pass  # Skip dilation without OpenCV

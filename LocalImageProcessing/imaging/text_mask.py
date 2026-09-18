"""Polygon mask generation."""

from __future__ import annotations

from typing import List

import numpy as np

from ..contract import RegionAction
from ..regions import ProcessRegion


def create_text_mask(
    image: np.ndarray,
    regions: List[ProcessRegion],
    expansion_pixels: int = 3,
) -> np.ndarray:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for region in regions:
        if region.action not in (RegionAction.translate, RegionAction.remove):
            continue
        polygon = np.array(region.polygon, dtype=np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)
        _fill_poly(mask, polygon)
        if expansion_pixels > 0:
            _expand_mask_region(mask, polygon, expansion_pixels)
    return mask


def _fill_poly(mask: np.ndarray, polygon: np.ndarray) -> None:
    try:
        import cv2

        cv2.fillPoly(mask, [polygon], 255)
    except ImportError:
        from PIL import Image, ImageDraw

        h, w = mask.shape
        pil_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(pil_img)
        draw.polygon([tuple(p) for p in polygon], fill=255)
        mask[:] = np.maximum(mask, np.array(pil_img))


def _expand_mask_region(mask: np.ndarray, polygon: np.ndarray, pixels: int) -> None:
    try:
        import cv2

        x, y, pw, ph = cv2.boundingRect(polygon)
        x = max(0, x - pixels)
        y = max(0, y - pixels)
        pw = min(mask.shape[1] - x, pw + 2 * pixels)
        ph = min(mask.shape[0] - y, ph + 2 * pixels)
        if pw > 0 and ph > 0:
            roi = mask[y : y + ph, x : x + pw]
            kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
            dilated = cv2.dilate(roi, kernel, iterations=1)
            mask[y : y + ph, x : x + pw] = np.maximum(roi, dilated)
    except ImportError:
        return

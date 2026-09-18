"""Image representation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class ImagePayload:
    pixels: np.ndarray
    has_alpha: bool = False
    color_mode: str = "BGR"
    source_suffix: Optional[str] = None

    def copy(self) -> "ImagePayload":
        return ImagePayload(
            pixels=self.pixels.copy(),
            has_alpha=self.has_alpha,
            color_mode=self.color_mode,
            source_suffix=self.source_suffix,
        )


def supports_alpha(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".webp", ".tif", ".tiff"}


def split_alpha(image: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3].copy(), image[:, :, 3].copy()
    return image.copy(), None


def merge_alpha(bgr: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    if alpha is None:
        return bgr
    if alpha.shape[:2] != bgr.shape[:2]:
        raise ValueError("Alpha and BGR dimensions do not match")
    return np.dstack([bgr, alpha])


def processing_view(payload: ImagePayload) -> tuple[np.ndarray, Optional[np.ndarray]]:
    return split_alpha(payload.pixels)


def finalize_output(
    bgr: np.ndarray,
    alpha: Optional[np.ndarray],
    payload: ImagePayload,
) -> ImagePayload:
    if payload.has_alpha or alpha is not None:
        merged = merge_alpha(bgr, alpha)
        return ImagePayload(merged, True, "BGRA", payload.source_suffix)
    return ImagePayload(bgr, False, "BGR", payload.source_suffix)

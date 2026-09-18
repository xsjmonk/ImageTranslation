"""Image representation helpers — color mode and alpha preservation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class ImagePayload:
    """In-memory image with explicit color mode metadata."""

    pixels: np.ndarray
    has_alpha: bool = False
    color_mode: str = "BGR"
    source_suffix: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.pixels.shape

    def copy(self) -> "ImagePayload":
        return ImagePayload(
            pixels=self.pixels.copy(),
            has_alpha=self.has_alpha,
            color_mode=self.color_mode,
            source_suffix=self.source_suffix,
        )


def _suffix(path: Path) -> str:
    return path.suffix.lower()


def supports_alpha(path: Path) -> bool:
    return _suffix(path) in {".png", ".webp", ".tif", ".tiff"}


def bgr_view(image: np.ndarray) -> np.ndarray:
    """Return a 3-channel BGR view for OCR/imaging operations."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=2)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def split_alpha(image: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Split a BGR/BGRA array into BGR pixels and optional alpha channel."""
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3].copy(), image[:, :, 3].copy()
    return image.copy(), None


def merge_alpha(bgr: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    """Reattach an alpha channel to a BGR image."""
    if alpha is None:
        return bgr
    if alpha.shape[:2] != bgr.shape[:2]:
        raise ValueError("Alpha and BGR dimensions do not match")
    return np.dstack([bgr, alpha])


def processing_view(payload: ImagePayload) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Return BGR pixels and source alpha for pipeline processing."""
    bgr, alpha = split_alpha(payload.pixels)
    return bgr, alpha


def finalize_output(bgr: np.ndarray, alpha: Optional[np.ndarray], payload: ImagePayload) -> ImagePayload:
    """Rebuild the output payload, preserving alpha when the source had it."""
    if payload.has_alpha or alpha is not None:
        merged = merge_alpha(bgr, alpha)
        return ImagePayload(merged, True, "BGRA", payload.source_suffix)
    return ImagePayload(bgr, False, "BGR", payload.source_suffix)

"""Inpaint reconstruction quality metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np


def assess_inpaint_quality(
    source_image: np.ndarray,
    cleaned_image: np.ndarray,
    mask: np.ndarray,
    *,
    flatness_threshold: float = 4.0,
    seam_threshold: float = 18.0,
    spill_threshold: float = 8.0,
) -> Dict[str, object]:
    masked = mask > 0
    if not np.any(masked):
        return {
            "flatness_score": 0.0,
            "seam_score": 0.0,
            "spill_score": 0.0,
            "issues": [],
        }

    rgb_source = _rgb_channels(source_image)
    rgb_clean = _rgb_channels(cleaned_image)
    flatness = float(np.std(rgb_clean[masked].astype(np.float32)))
    seam = _edge_seam_score(rgb_source, rgb_clean, mask)
    spill = _outside_mask_change(rgb_source, rgb_clean, mask)

    issues = []
    if flatness < flatness_threshold:
        issues.append(
            {
                "severity": "warning",
                "code": "inpaint_flat_region",
                "message": f"Inpainted area appears overly flat (score {flatness:.2f})",
            }
        )
    if seam > seam_threshold:
        issues.append(
            {
                "severity": "warning",
                "code": "inpaint_seam",
                "message": f"Inpaint seam detected at mask boundary (score {seam:.2f})",
            }
        )
    if spill > spill_threshold:
        issues.append(
            {
                "severity": "warning",
                "code": "inpaint_spill",
                "message": f"Inpaint change spilled outside mask (score {spill:.2f})",
            }
        )

    return {
        "flatness_score": flatness,
        "seam_score": seam,
        "spill_score": spill,
        "issues": issues,
    }


def _rgb_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=2)
    if image.shape[2] >= 3:
        return image[:, :, :3]
    return image


def _edge_seam_score(source: np.ndarray, cleaned: np.ndarray, mask: np.ndarray) -> float:
    try:
        import cv2
    except ImportError:
        return 0.0

    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.dilate(mask, kernel, iterations=1) - mask
    if not np.any(boundary):
        return 0.0
    diff = np.abs(cleaned.astype(np.float32) - source.astype(np.float32))
    gray = diff.mean(axis=2) if diff.ndim == 3 else diff
    return float(np.mean(gray[boundary > 0]))


def _outside_mask_change(source: np.ndarray, cleaned: np.ndarray, mask: np.ndarray) -> float:
    outside = mask == 0
    if not np.any(outside):
        return 0.0
    diff = np.abs(cleaned.astype(np.float32) - source.astype(np.float32))
    gray = diff.mean(axis=2) if diff.ndim == 3 else diff
    return float(np.mean(gray[outside]))

"""Polygon geometry helpers for QC."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def point_in_polygon(x: float, y: float, polygon: np.ndarray, tolerance: float = 0.0) -> bool:
    if _point_in_poly_raw(x, y, polygon):
        return True
    if tolerance <= 0:
        return False
    dists = np.sqrt((polygon[:, 0] - x) ** 2 + (polygon[:, 1] - y) ** 2)
    return float(dists.min()) <= tolerance


def alpha_outside_polygon_ratio(
    layer_rgba: np.ndarray,
    polygon: Sequence[Sequence[float]],
    *,
    margin: float = 0.0,
    alpha_threshold: int = 8,
) -> Tuple[float, int]:
    if layer_rgba is None or layer_rgba.ndim != 3 or layer_rgba.shape[2] < 4:
        return 1.0, 0
    alpha = layer_rgba[:, :, 3]
    ys, xs = np.where(alpha > alpha_threshold)
    if len(xs) == 0:
        return 1.0, 0
    poly = np.array(polygon, dtype=np.float32)
    outside = 0
    for x, y in zip(xs, ys):
        if not point_in_polygon(float(x), float(y), poly, margin):
            outside += 1
    return outside / float(len(xs)), len(xs)


def oriented_layout_polygon(
    center: Tuple[float, float],
    width: float,
    height: float,
    angle_deg: float,
    margin: float = 0.0,
) -> List[List[float]]:
    half_w = float(width) / 2.0 + margin
    half_h = float(height) / 2.0 + margin
    cx, cy = float(center[0]), float(center[1])
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    rotated: List[List[float]] = []
    for x, y in corners:
        if abs(angle_deg) > 0.5:
            theta = math.radians(angle_deg)
            rx = x * math.cos(theta) - y * math.sin(theta)
            ry = x * math.sin(theta) + y * math.cos(theta)
        else:
            rx, ry = x, y
        rotated.append([cx + rx, cy + ry])
    return rotated


def _point_in_poly_raw(x: float, y: float, polygon: np.ndarray) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
        ):
            inside = not inside
    return inside

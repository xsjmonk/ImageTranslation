"""Polygon geometry helpers for QC and region matching."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    pts = np.array(poly, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def point_in_polygon(x: float, y: float, polygon: np.ndarray, tolerance: float = 0.0) -> bool:
    if _point_in_poly_raw(x, y, polygon):
        return True
    if tolerance <= 0:
        return False
    dists = np.sqrt((polygon[:, 0] - x) ** 2 + (polygon[:, 1] - y) ** 2)
    return float(dists.min()) <= tolerance


def polygon_intersection_area(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Approximate intersection area via grid sampling inside the union bbox."""
    poly_a = np.array(a, dtype=np.float64)
    poly_b = np.array(b, dtype=np.float64)
    xs = np.concatenate([poly_a[:, 0], poly_b[:, 0]])
    ys = np.concatenate([poly_a[:, 1], poly_b[:, 1]])
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    if x2 <= x1 or y2 <= y1:
        return 0.0

    steps = 24
    xs_grid = np.linspace(x1, x2, steps)
    ys_grid = np.linspace(y1, y2, steps)
    count = 0
    for x in xs_grid:
        for y in ys_grid:
            if point_in_polygon(x, y, poly_a) and point_in_polygon(x, y, poly_b):
                count += 1
    cell_area = ((x2 - x1) / steps) * ((y2 - y1) / steps)
    return float(count) * cell_area


def polygon_overlap_coverage(
    region_poly: Sequence[Sequence[float]],
    detection_poly: Sequence[Sequence[float]],
) -> float:
    """Return intersection area divided by the smaller polygon area."""
    inter = polygon_intersection_area(region_poly, detection_poly)
    area_a = max(polygon_area(region_poly), 1.0)
    area_b = max(polygon_area(detection_poly), 1.0)
    return inter / min(area_a, area_b)


def alpha_outside_polygon_ratio(
    layer_rgba: np.ndarray,
    polygon: Sequence[Sequence[float]],
    *,
    margin: float = 0.0,
    alpha_threshold: int = 8,
) -> Tuple[float, int]:
    """Return the fraction of glyph alpha pixels that fall outside the polygon."""
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
    """Build the oriented layout quadrilateral used by renderer and validator."""
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

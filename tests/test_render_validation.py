"""Rendered glyph bounds validation tests."""

from __future__ import annotations

import numpy as np

from image_translation.revision.render_validation import (
    measure_alpha_bounds,
    validate_rendered_layout,
)
from image_translation.revision.text_renderer import TextRenderer
from image_translation.utilities.geometry import oriented_layout_polygon


def test_measure_alpha_bounds_detects_text():
    renderer = TextRenderer()
    layer = renderer.render_text_layer(
        120,
        60,
        {
            "font_size": 18,
            "lines": ["HELLO"],
            "line_metrics": [{"text": "HELLO", "width": 40, "height": 18}],
            "center": (60, 30),
            "width": 100,
            "angle": 0.0,
            "alignment": "center",
        },
    )
    bounds = measure_alpha_bounds(layer)
    assert bounds is not None
    assert bounds["alpha_coverage"] > 0.005


def test_rotated_glyph_escape_fails_polygon_qc():
    layer = np.zeros((100, 100, 4), dtype=np.uint8)
    layer[20:80, 10:90, 3] = 255
    layout = {
        "region_id": "r1",
        "center": (50, 50),
        "width": 20,
        "height": 60,
        "angle": 45.0,
        "render_layer": layer,
        "render_bounds": measure_alpha_bounds(layer),
        "stroke_margin": 0,
    }
    polygon = oriented_layout_polygon((50, 50), 20, 60, 45.0)
    issues = validate_rendered_layout(
        layout,
        polygon,
        clip_tolerance_pixels=1.0,
        max_outside_polygon_ratio=0.05,
    )
    assert any(i["code"] == "text_clipped" for i in issues)


def test_aabb_looks_ok_but_polygon_qc_catches_escape():
    layer = np.zeros((100, 100, 4), dtype=np.uint8)
    layer[45:55, 5:95, 3] = 255
    layout = {
        "region_id": "r1",
        "center": (50, 50),
        "width": 20,
        "height": 10,
        "angle": 45.0,
        "render_layer": layer,
        "render_bounds": measure_alpha_bounds(layer),
        "stroke_margin": 0,
    }
    polygon = oriented_layout_polygon((50, 50), 20, 10, 45.0)
    issues = validate_rendered_layout(
        layout,
        polygon,
        clip_tolerance_pixels=1.0,
        max_outside_polygon_ratio=0.05,
    )
    assert any(i["code"] == "text_clipped" for i in issues)


def test_missing_rendered_text_fails():
    layout = {"region_id": "r1"}
    issues = validate_rendered_layout(
        layout,
        [[0, 0], [50, 0], [50, 20], [0, 20]],
        min_alpha_coverage=0.005,
    )
    assert any(i["code"] == "missing_rendered_text" for i in issues)

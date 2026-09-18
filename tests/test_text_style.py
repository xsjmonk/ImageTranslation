"""Text style estimation and rendering fixture tests."""

from __future__ import annotations

import numpy as np

from image_translation.revision.render_validation import measure_alpha_bounds
from image_translation.revision.text_renderer import TextRenderer
from image_translation.revision.text_style import TextStyle, estimate_style_from_crop


def _solid_crop(color_bgr: tuple[int, int, int]) -> np.ndarray:
    crop = np.zeros((30, 80, 3), dtype=np.uint8)
    crop[:, :] = color_bgr
    return crop


def test_estimate_solid_color_style():
    crop = _solid_crop((20, 20, 20))
    style = estimate_style_from_crop(crop, [0, 0, 80, 30])
    assert style.color_rgb[0] < 40
    assert "foreground_color" in style.recovered_effects


def test_gradient_renders_multiple_colors():
    renderer = TextRenderer()
    style = TextStyle(
        gradient_top_rgb=(255, 0, 0),
        gradient_bottom_rgb=(0, 0, 255),
        opacity=1.0,
    )
    layer = renderer.render_text_layer(
        120,
        60,
        {
            "font_size": 24,
            "lines": ["GRAD"],
            "line_metrics": [{"text": "GRAD", "width": 60, "height": 24}],
            "center": (60, 30),
            "width": 100,
            "angle": 0.0,
            "alignment": "center",
        },
        style,
    )
    bounds = measure_alpha_bounds(layer)
    assert bounds is not None
    top_row = int(bounds["y1"])
    bottom_row = int(bounds["y2"] - 1)
    top_color = layer[top_row, int(bounds["x1"]), :3]
    bottom_color = layer[bottom_row, int(bounds["x1"]), :3]
    assert not np.array_equal(top_color, bottom_color)


def test_shadow_blur_does_not_blur_main_glyph():
    renderer = TextRenderer()
    style = TextStyle(
        color_rgb=(255, 255, 255),
        shadow_offset=(4, 4),
        shadow_blur=4,
        shadow_opacity=0.8,
        shadow_color_rgb=(0, 0, 0),
    )
    layer = renderer.render_text_layer(
        140,
        80,
        {
            "font_size": 28,
            "lines": ["SHADOW"],
            "line_metrics": [{"text": "SHADOW", "width": 80, "height": 28}],
            "center": (70, 40),
            "width": 120,
            "angle": 0.0,
            "alignment": "center",
        },
        style,
    )
    bounds = measure_alpha_bounds(layer)
    assert bounds is not None
    ys, xs = np.where(layer[:, :, 3] > 200)
    assert len(xs) > 0
    bright = layer[ys, xs]
    assert np.any(bright[:, :3] > 200)

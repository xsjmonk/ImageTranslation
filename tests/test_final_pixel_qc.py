"""Final-pixel quality control tests with synthetic images."""

import numpy as np

from image_translation.config.models import QcConfig
from image_translation.models.text_region import TextAction, TextRegion  # noqa: F401
from image_translation.quality_control import validate_final_image


def _region(action=TextAction.preserve, text="LOGO"):
    return TextRegion(
        id="r1",
        source_text=text,
        confidence=0.9,
        polygon=[[0, 0], [20, 0], [20, 10], [0, 10]],
        action=action,
    )


def test_dimension_mismatch_fails():
    source = np.zeros((50, 50, 3), dtype=np.uint8)
    final = np.zeros((40, 50, 3), dtype=np.uint8)
    result = validate_final_image(source, final, [], [], QcConfig())
    assert not result.passed
    assert any(i.code == "dimension_mismatch" for i in result.issues)


def test_preserved_region_change_fails():
    source = np.full((50, 50, 3), 200, dtype=np.uint8)
    final = source.copy()
    final[0:10, 0:20] = 0
    result = validate_final_image(
        source, final, [_region()], [], QcConfig(preserved_region_tolerance=5.0)
    )
    assert not result.passed
    assert any(i.code == "preserved_region_changed" for i in result.issues)


def test_clipped_layout_fails():
    source = final = np.zeros((50, 50, 3), dtype=np.uint8)
    region = TextRegion(
        id="r1",
        source_text="TEST",
        confidence=0.9,
        polygon=[[5, 5], [25, 5], [25, 20], [5, 20]],
        action=TextAction.translate,
        translation={"translated_text": "TEST"},
    )
    layer = np.zeros((50, 50, 4), dtype=np.uint8)
    layer[0:30, 0:40, 3] = 255
    layouts = [
        {
            "region_id": "r1",
            "center": (15, 12),
            "width": 20,
            "height": 15,
            "angle": 0.0,
            "stroke_margin": 0,
            "render_layer": layer,
            "render_bounds": {
                "x1": 0.0,
                "y1": 0.0,
                "x2": 40.0,
                "y2": 30.0,
                "alpha_coverage": 0.05,
            },
        }
    ]
    result = validate_final_image(
        source,
        final,
        [region],
        layouts,
        QcConfig(enable_final_ocr=False),
    )
    assert not result.passed
    assert any(i.code == "text_clipped" for i in result.issues)

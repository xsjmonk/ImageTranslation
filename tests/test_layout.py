"""Layout engine tests."""

from image_translation.revision.layout import LayoutEngine


def test_long_unspaced_english_wraps():
    engine = LayoutEngine(minimum_font_size=10, maximum_font_size=24, allow_multiline=True)
    polygon = [[0, 0], [120, 0], [120, 40], [0, 40]]
    layout = engine.compute_layout(polygon, "SUPERCALIFRAGILISTICEXPIALIDOCIOUS", region_id="r1")
    assert len(layout["lines"]) >= 2
    assert layout["font_size"] >= 10


def test_alignment_left():
    engine = LayoutEngine()
    polygon = [[0, 0], [5, 0], [20, 30], [0, 30]]
    layout = engine.compute_layout(polygon, "LEFT", region_id="r1")
    assert layout["alignment"] == "left"

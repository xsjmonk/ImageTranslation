"""RGBA image preservation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from image_translation.utilities.image_format import ImagePayload
from image_translation.utilities.images import load_image, read_dimensions, save_image_atomic


def test_rgba_round_trip_preserves_alpha(tmp_path: Path):
    path = tmp_path / "alpha.png"
    rgba = np.zeros((40, 60, 4), dtype=np.uint8)
    rgba[:, :, :3] = 100
    rgba[:, :, 3] = 128
    payload = ImagePayload(rgba, True, "BGRA", ".png")
    save_image_atomic(path, payload)

    loaded = load_image(path)
    assert loaded.has_alpha
    assert loaded.pixels.shape[2] == 4
    assert loaded.pixels[10, 10, 3] == 128
    w, h = read_dimensions(path)
    assert (w, h) == (60, 40)

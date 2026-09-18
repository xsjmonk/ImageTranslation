"""Inpaint reconstruction quality metric tests."""

from __future__ import annotations

import numpy as np

from image_translation.imaging.inpaint_quality import assess_inpaint_quality


def test_flat_inpaint_region_warns():
    source = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    cleaned = source.copy()
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    cleaned[10:30, 10:30] = 180
    report = assess_inpaint_quality(
        source,
        cleaned,
        mask,
        flatness_threshold=4.0,
        seam_threshold=100.0,
        spill_threshold=100.0,
    )
    assert any(i["code"] == "inpaint_flat_region" for i in report["issues"])


def test_rendered_english_change_not_assessed_here():
    source = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    cleaned = source.copy()
    final = cleaned.copy()
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:40] = 255
    cleaned[10:20, 10:40] = 127
    final[10:20, 10:40] = 0
    report = assess_inpaint_quality(source, cleaned, mask, spill_threshold=1000.0)
    assert report["spill_score"] < 1000.0


def test_spill_outside_mask_warns():
    source = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    cleaned = source.copy()
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    cleaned[:, :] = source
    cleaned[5:25, 5:25] = 0
    report = assess_inpaint_quality(
        source,
        cleaned,
        mask,
        flatness_threshold=0.0,
        seam_threshold=1000.0,
        spill_threshold=0.5,
    )
    assert any(i["code"] == "inpaint_spill" for i in report["issues"])

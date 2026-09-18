"""Fixture-based visual localization smoke test with injected components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from image_translation.config import AppConfig
from image_translation.input.models import AppInput, InputType
from image_translation.pipeline import run_pipeline
from image_translation.pipeline_context import PipelineContext
from image_translation.revision.image_reviser import ImageReviser
from image_translation.utilities.images import load_image
from tests.test_pipeline_e2e import FakeOcr, FakeProcessor, FakeTranslator, _write_test_image


def test_visual_localization_fixture(tmp_path: Path):
    src = tmp_path / "fixture_in"
    src.mkdir()
    image = src / "product.jpg"
    _write_test_image(image)

    out_root = tmp_path / "fixture_out"
    config = AppConfig()
    config.output.overwrite_existing = True
    config.quality_control.enable_final_ocr = False
    config.quality_control.inpaint_flatness_threshold = 0.0
    config.quality_control.inpaint_spill_threshold = 1000.0
    ctx = PipelineContext(
        config,
        ocr_engine=FakeOcr(),
        translator=FakeTranslator(),
        image_processor=FakeProcessor(),
        image_reviser=ImageReviser(config.revision),
    )
    app_input = AppInput(
        input_path=image,
        input_type=InputType.SINGLE_IMAGE,
        input_folder=None,
        single_image_path=image,
        output_folder=out_root,
        config_path=None,
    )
    result = run_pipeline(app_input, config, context=ctx)
    assert result.succeeded == 1

    localized = out_root / "product.jpg"
    preserved = out_root / "original_product.jpg"
    summary = out_root / "summary.json"
    meta = out_root / "metadata" / "product.json"
    assert localized.exists()
    assert preserved.exists()
    assert preserved.read_bytes() == image.read_bytes()
    assert summary.exists()
    assert meta.exists()

    loaded = load_image(localized)
    assert loaded.pixels.shape[0] == 80
    assert loaded.pixels.shape[1] == 120

    diagnostic = json.loads(meta.read_text(encoding="utf-8"))
    assert diagnostic["preserved_original_path"].endswith("original_product.jpg")
    assert diagnostic["localized_output_path"].endswith("product.jpg")
    assert diagnostic["status"] == "completed"

    # Visual evidence: localized output differs from preserved original in at least one pixel.
    original = load_image(preserved).pixels
    localized_pixels = loaded.pixels[:, :, :3]
    original_bgr = original[:, :, :3] if original.ndim == 3 else original
    assert not np.array_equal(localized_pixels, original_bgr)

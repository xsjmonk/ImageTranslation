"""End-to-end RGBA pipeline tests with injected components."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from image_translation.config import AppConfig
from image_translation.input.models import AppInput, InputType
from image_translation.pipeline import run_pipeline
from image_translation.pipeline_context import PipelineContext
from image_translation.revision.image_reviser import ImageReviser
from image_translation.utilities.image_format import ImagePayload
from image_translation.utilities.images import load_image, save_image_atomic
from tests.test_pipeline_e2e import FakeOcr, FakeProcessor, FakeTranslator


def test_rgba_pipeline_preserves_alpha_outside_text_region(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    image = src / "alpha.png"
    rgba = np.zeros((80, 120, 4), dtype=np.uint8)
    rgba[:, :, :3] = 120
    rgba[:, :, 3] = 200
    save_image_atomic(image, ImagePayload(rgba, True, "BGRA", ".png"))

    out_root = tmp_path / "out"
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

    output = out_root / "alpha.png"
    preserved = out_root / "original_alpha.png"
    assert output.exists()
    assert preserved.exists()
    assert preserved.read_bytes() == image.read_bytes()

    loaded = load_image(output)
    assert loaded.has_alpha
    assert loaded.pixels.shape[2] == 4
    assert loaded.pixels[5, 5, 3] == 200
    assert loaded.pixels[70, 110, 3] == 200

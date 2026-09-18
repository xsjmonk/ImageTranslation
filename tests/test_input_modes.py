"""Regression tests for folder vs single-photo input enumeration."""

from __future__ import annotations

from pathlib import Path

from image_translation.config import AppConfig
from image_translation.input.models import AppInput, InputType
from image_translation.pipeline import run_pipeline
from image_translation.pipeline_context import PipelineContext
from image_translation.revision.image_reviser import ImageReviser
from tests.test_pipeline_e2e import FakeOcr, FakeProcessor, FakeTranslator, _write_test_image


def _make_context(config: AppConfig) -> PipelineContext:
    return PipelineContext(
        config,
        ocr_engine=FakeOcr(),
        translator=FakeTranslator(),
        image_processor=FakeProcessor(),
        image_reviser=ImageReviser(config.revision),
    )


def test_folder_processes_all_images(tmp_path: Path):
    src = tmp_path / "photos"
    src.mkdir()
    for name in ("a.jpg", "b.jpg", "c.png"):
        _write_test_image(src / name)

    out_root = tmp_path / "out"
    app_input = AppInput(
        input_path=src,
        input_type=InputType.FOLDER,
        input_folder=src,
        single_image_path=None,
        output_folder=out_root,
        config_path=None,
    )
    config = AppConfig()
    config.output.overwrite_existing = True
    config.quality_control.enable_final_ocr = False
    config.quality_control.inpaint_flatness_threshold = 0.0
    config.quality_control.inpaint_spill_threshold = 1000.0
    result = run_pipeline(app_input, config, context=_make_context(config))

    assert result.total == 3
    assert result.succeeded == 3
    assert (out_root / "original_a.jpg").exists()
    assert (out_root / "original_b.jpg").exists()
    assert (out_root / "original_c.png").exists()


def test_single_photo_processes_only_that_file(tmp_path: Path):
    src = tmp_path / "photos"
    src.mkdir()
    target = src / "pick.jpg"
    sibling = src / "skip.jpg"
    _write_test_image(target)
    _write_test_image(sibling)

    out_root = tmp_path / "out"
    app_input = AppInput(
        input_path=target,
        input_type=InputType.SINGLE_IMAGE,
        input_folder=None,
        single_image_path=target,
        output_folder=out_root,
        config_path=None,
    )
    config = AppConfig()
    config.output.overwrite_existing = True
    config.quality_control.enable_final_ocr = False
    config.quality_control.inpaint_flatness_threshold = 0.0
    config.quality_control.inpaint_spill_threshold = 1000.0
    result = run_pipeline(app_input, config, context=_make_context(config))

    assert result.total == 1
    assert result.succeeded == 1
    assert (out_root / "pick.jpg").exists()
    assert (out_root / "original_pick.jpg").exists()
    assert not (out_root / "skip.jpg").exists()
    assert not (out_root / "original_skip.jpg").exists()

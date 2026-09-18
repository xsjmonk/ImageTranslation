"""Transactional skip, retry, and conflict behavior tests."""

from __future__ import annotations

from pathlib import Path

from image_translation.config import AppConfig
from image_translation.input.models import AppInput, InputType
from image_translation.models.image_job import JobStatus
from image_translation.pipeline import run_pipeline
from image_translation.pipeline_context import PipelineContext
from image_translation.revision.image_reviser import ImageReviser
from image_translation.translation.translator import NoopTranslator
from tests.test_pipeline_e2e import FakeOcr, FakeProcessor, FakeTranslator, _write_test_image


def _app_input(image: Path, out_root: Path) -> AppInput:
    return AppInput(
        input_path=image,
        input_type=InputType.SINGLE_IMAGE,
        input_folder=None,
        single_image_path=image,
        output_folder=out_root,
        config_path=None,
    )


def _ctx(config: AppConfig) -> PipelineContext:
    return PipelineContext(
        config,
        ocr_engine=FakeOcr(),
        translator=FakeTranslator(),
        image_processor=FakeProcessor(),
        image_reviser=ImageReviser(config.revision),
    )


def test_retry_after_failed_run_reuses_preserved_original(tmp_path: Path):
    config = AppConfig()
    config.quality_control.enable_final_ocr = False
    config.quality_control.inpaint_flatness_threshold = 0.0
    config.quality_control.inpaint_spill_threshold = 1000.0

    src = tmp_path / "in"
    src.mkdir()
    image = src / "one.jpg"
    _write_test_image(image)
    out_root = tmp_path / "out"

    fail_ctx = PipelineContext(config, ocr_engine=FakeOcr(), translator=NoopTranslator())
    result1 = run_pipeline(_app_input(image, out_root), config, context=fail_ctx)
    assert result1.failed == 1
    preserved = out_root / "original_one.jpg"
    assert preserved.exists()
    assert preserved.read_bytes() == image.read_bytes()
    assert not (out_root / "one.jpg").exists()

    result2 = run_pipeline(_app_input(image, out_root), config, context=_ctx(config))
    assert result2.succeeded == 1
    assert preserved.read_bytes() == image.read_bytes()
    assert (out_root / "one.jpg").exists()


def test_mismatched_preserved_copy_reports_conflict(tmp_path: Path):
    config = AppConfig()
    config.quality_control.enable_final_ocr = False
    config.quality_control.inpaint_flatness_threshold = 0.0
    config.quality_control.inpaint_spill_threshold = 1000.0

    src = tmp_path / "in"
    src.mkdir()
    image = src / "one.jpg"
    _write_test_image(image)
    out_root = tmp_path / "out"

    result1 = run_pipeline(_app_input(image, out_root), config, context=_ctx(config))
    assert result1.succeeded == 1

    preserved = out_root / "original_one.jpg"
    preserved.write_bytes(b"tampered preserved copy")
    (out_root / "one.jpg").unlink()

    result2 = run_pipeline(_app_input(image, out_root), config, context=_ctx(config))
    assert result2.conflicts == 1
    assert result2.jobs[-1].status == JobStatus.conflict
    assert preserved.read_bytes() == b"tampered preserved copy"


def test_skip_when_localized_output_exists(tmp_path: Path):
    config = AppConfig()
    config.quality_control.enable_final_ocr = False
    config.quality_control.inpaint_flatness_threshold = 0.0
    config.quality_control.inpaint_spill_threshold = 1000.0

    src = tmp_path / "in"
    src.mkdir()
    image = src / "one.jpg"
    _write_test_image(image)
    out_root = tmp_path / "out"

    result1 = run_pipeline(_app_input(image, out_root), config, context=_ctx(config))
    assert result1.succeeded == 1

    result2 = run_pipeline(_app_input(image, out_root), config, context=_ctx(config))
    assert result2.skipped == 1
    assert "localized output already exists" in result2.jobs[-1].error

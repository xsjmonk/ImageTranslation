"""Tests for original_<filename> preservation and naming rules."""

from __future__ import annotations

from pathlib import Path

from image_translation.config import AppConfig
from image_translation.input.models import AppInput, InputType
from image_translation.pipeline import run_pipeline
from image_translation.pipeline_context import PipelineContext
from image_translation.revision.image_reviser import ImageReviser
from image_translation.translation.translator import NoopTranslator
from image_translation.utilities.output_paths import (
    preserved_original_basename,
    resolve_output_file_path,
    resolve_preserved_original_path,
)
from tests.test_pipeline_e2e import FakeOcr, FakeProcessor, FakeTranslator, _write_test_image


class TestPreservedNaming:
    def test_prefix_before_filename(self):
        assert preserved_original_basename("01.jpg") == "original_01.jpg"
        assert preserved_original_basename("product.png") == "original_product.png"

    def test_idempotent_no_double_prefix(self):
        assert preserved_original_basename("original_01.jpg") == "original_01.jpg"

    def test_extension_preserved(self, tmp_path: Path):
        source = tmp_path / "in" / "sub" / "shot.webp"
        source.parent.mkdir(parents=True)
        out_root = tmp_path / "out"
        preserved = resolve_preserved_original_path(source, out_root, tmp_path / "in")
        assert preserved == out_root / "sub" / "original_shot.webp"

    def test_localized_vs_preserved_paths(self, tmp_path: Path):
        source = tmp_path / "01.jpg"
        out_root = tmp_path / "out"
        localized = resolve_output_file_path(source, out_root, None, AppConfig().output)
        preserved = resolve_preserved_original_path(source, out_root, None)
        assert localized.name == "01.jpg"
        assert preserved.name == "original_01.jpg"

    def test_source_already_original_prefix(self, tmp_path: Path):
        source = tmp_path / "original_01.jpg"
        out_root = tmp_path / "out"
        localized = resolve_output_file_path(source, out_root, None, AppConfig().output)
        preserved = resolve_preserved_original_path(source, out_root, None)
        assert preserved.name == "original_01.jpg"
        assert localized.name == "01.jpg"


class TestPreservedCopyPipeline:
    def _run_one(self, tmp_path: Path, source_name: str = "one.jpg"):
        src_dir = tmp_path / "in"
        src_dir.mkdir()
        image = src_dir / source_name
        _write_test_image(image)
        source_bytes = image.read_bytes()

        out_root = tmp_path / "out"
        app_input = AppInput(
            input_path=image,
            input_type=InputType.SINGLE_IMAGE,
            input_folder=None,
            single_image_path=image,
            output_folder=out_root,
            config_path=None,
        )
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
        result = run_pipeline(app_input, config, context=ctx)
        return result, image, source_bytes, out_root

    def test_writes_preserved_copy_with_prefix(self, tmp_path: Path):
        result, image, source_bytes, out_root = self._run_one(tmp_path)
        assert result.succeeded == 1
        preserved = out_root / "original_one.jpg"
        assert preserved.exists()
        assert preserved.read_bytes() == source_bytes
        assert image.read_bytes() == source_bytes

    def test_diagnostic_paths(self, tmp_path: Path):
        self._run_one(tmp_path)
        import json

        meta = json.loads((tmp_path / "out" / "metadata" / "one.json").read_text(encoding="utf-8"))
        assert meta["preserved_original_path"].endswith("original_one.jpg")
        assert meta["localized_output_path"].endswith("one.jpg")
        summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
        entry = summary["images"][0]
        assert entry["preserved_original_path"].endswith("original_one.jpg")
        assert entry["localized_output_path"].endswith("one.jpg")

    def test_idempotent_source_name(self, tmp_path: Path):
        result, image, source_bytes, out_root = self._run_one(tmp_path, "original_01.jpg")
        assert result.succeeded == 1
        preserved = out_root / "original_01.jpg"
        localized = out_root / "01.jpg"
        assert preserved.exists()
        assert localized.exists()
        assert preserved.read_bytes() == source_bytes
        assert not (out_root / "original_original_01.jpg").exists()

    def test_mismatched_preserved_reports_conflict(self, tmp_path: Path):
        src_dir = tmp_path / "in"
        src_dir.mkdir()
        image = src_dir / "one.jpg"
        _write_test_image(image)
        out_root = tmp_path / "out"
        out_root.mkdir()
        preserved = out_root / "original_one.jpg"
        preserved.write_bytes(b"existing preserved")

        app_input = AppInput(
            input_path=image,
            input_type=InputType.SINGLE_IMAGE,
            input_folder=None,
            single_image_path=image,
            output_folder=out_root,
            config_path=None,
        )
        config = AppConfig()
        result = run_pipeline(
            app_input,
            config,
            context=PipelineContext(config, ocr_engine=FakeOcr(), translator=NoopTranslator()),
        )
        assert result.conflicts == 1
        assert "differs from source" in result.jobs[0].error
        assert preserved.read_bytes() == b"existing preserved"

    def test_recursive_output_tree(self, tmp_path: Path):
        src_root = tmp_path / "in"
        sub = src_root / "batch"
        sub.mkdir(parents=True)
        image = sub / "a.jpg"
        _write_test_image(image)
        source_bytes = image.read_bytes()

        out_root = tmp_path / "out"
        app_input = AppInput(
            input_path=src_root,
            input_type=InputType.FOLDER,
            input_folder=src_root,
            single_image_path=None,
            output_folder=out_root,
            config_path=None,
        )
        config = AppConfig()
        config.input.recursive = True
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
        run_pipeline(app_input, config, context=ctx)

        preserved = out_root / "batch" / "original_a.jpg"
        localized = out_root / "batch" / "a.jpg"
        assert preserved.exists()
        assert localized.exists()
        assert preserved.read_bytes() == source_bytes

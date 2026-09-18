"""Model-free end-to-end pipeline tests with injected components."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from image_translation.config import AppConfig
from image_translation.imaging.base import ImageProcessor
from image_translation.input.models import AppInput, InputType
from image_translation.models.text_region import TextAction, TextRegion
from image_translation.ocr.base import OcrEngine
from image_translation.pipeline import run_pipeline
from image_translation.pipeline_context import PipelineContext
from image_translation.revision.image_reviser import ImageReviser
from image_translation.translation.base import Translator
from image_translation.translation.translator import NoopTranslator
from image_translation.utilities.output_paths import (
    resolve_output_file_path,
    resolve_preserved_original_path,
)


class FakeOcr(OcrEngine):
    @property
    def name(self) -> str:
        return "fake"

    def detect(self, image: np.ndarray, *, min_confidence=None) -> List[TextRegion]:
        return [
            TextRegion(
                id="r1",
                source_text="正面展示说明",
                confidence=0.95,
                polygon=[[10, 10], [100, 10], [100, 40], [10, 40]],
                source_crop_bbox=[10, 10, 100, 40],
            )
        ]


class FakeTranslator(Translator):
    @property
    def name(self) -> str:
        return "fake"

    @property
    def runtime_info(self):
        from image_translation.translation.models import TranslationRuntimeInfo
        return TranslationRuntimeInfo(backend="fake", model_name="fake")

    def translate_text(self, text, source_lang="zh", target_lang="en", style=None):
        from image_translation.translation.models import TranslationResult
        return TranslationResult(
            source_text=text,
            translated_text="GUNMETAL",
            compact_text="GUNMETAL",
            literal_text="GUNMETAL",
        )

    def translate_batch_texts(self, texts, source_lang="zh", target_lang="en", max_new_tokens=None, style=None):
        return [self.translate_text(t) for t in texts]


class FakeProcessor(ImageProcessor):
    @property
    def name(self) -> str:
        return "fake"

    def generate_mask(self, image, regions):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[10:40, 10:100] = 255
        return mask

    def remove_text(self, image, mask):
        cleaned = image.copy()
        cleaned[mask > 0] = 255
        return cleaned


def _write_test_image(path: Path) -> None:
    import cv2
    img = np.full((80, 120, 3), 255, dtype=np.uint8)
    cv2.putText(img, "X", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(str(path), img)


def _make_context(config: AppConfig) -> PipelineContext:
    return PipelineContext(
        config,
        ocr_engine=FakeOcr(),
        translator=FakeTranslator(),
        image_processor=FakeProcessor(),
        image_reviser=ImageReviser(config.revision),
    )


def test_pipeline_success_with_diagnostics(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    image = src / "one.jpg"
    _write_test_image(image)

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
    result = run_pipeline(app_input, config, context=_make_context(config))

    assert result.succeeded == 1
    assert result.failed == 0
    output_file = resolve_output_file_path(image, out_root, src, config.output)
    assert output_file.exists()
    preserved = resolve_preserved_original_path(image, out_root, src)
    assert preserved.exists()
    assert preserved.read_bytes() == image.read_bytes()
    assert (out_root / "summary.json").exists()
    assert (out_root / "metadata" / "one.json").exists()


def test_pipeline_skip_existing_output(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    image = src / "one.jpg"
    _write_test_image(image)
    out_root = tmp_path / "out"
    out_root.mkdir()
    existing = resolve_output_file_path(image, out_root, src, AppConfig().output)
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"existing")

    app_input = AppInput(
        input_path=src,
        input_type=InputType.FOLDER,
        input_folder=src,
        single_image_path=None,
        output_folder=out_root,
        config_path=None,
    )
    config = AppConfig()
    result = run_pipeline(app_input, config, context=_make_context(config))
    assert result.skipped == 1
    assert (out_root / "metadata" / "one.json").exists()


def test_pipeline_continue_on_error(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    good = src / "good.jpg"
    bad = src / "bad.jpg"
    _write_test_image(good)
    _write_test_image(bad)

    out_root = tmp_path / "out"
    config = AppConfig()
    config.output.overwrite_existing = True
    config.quality_control.enable_final_ocr = False
    config.quality_control.inpaint_flatness_threshold = 0.0
    config.general.continue_on_error = True

    class FirstEmptyOcr(FakeOcr):
        def __init__(self):
            self._calls = 0

        def detect(self, image):
            self._calls += 1
            if self._calls == 1:
                return []
            return super().detect(image)

    app_input = AppInput(
        input_path=src,
        input_type=InputType.FOLDER,
        input_folder=src,
        single_image_path=None,
        output_folder=out_root,
        config_path=None,
    )
    ctx = PipelineContext(
        config,
        ocr_engine=FirstEmptyOcr(),
        translator=NoopTranslator(),
        image_processor=FakeProcessor(),
        image_reviser=ImageReviser(config.revision),
    )
    result = run_pipeline(app_input, config, context=ctx)
    assert result.succeeded == 1
    assert result.failed == 1


def test_pipeline_fails_on_noop_translation(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    image = src / "one.jpg"
    _write_test_image(image)

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
    config.quality_control.enable_final_ocr = False
    config.quality_control.inpaint_flatness_threshold = 0.0
    ctx = PipelineContext(config, ocr_engine=FakeOcr(), translator=NoopTranslator())
    result = run_pipeline(app_input, config, context=ctx)
    assert result.failed == 1
    assert not resolve_output_file_path(image, out_root, src, config.output).exists()

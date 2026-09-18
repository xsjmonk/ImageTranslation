"""Run-scoped pipeline services — no module-global singletons."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import AppConfig
from .imaging.base import ImageProcessor
from .imaging.factory import create_image_processor
from .ocr import OcrEngine, PaddleOcrEngine
from .revision import ImageReviser
from .translation import Translator
from .translator_factory import create_image_pipeline_translator


class PipelineContext:
    """Owns heavy services for a single pipeline run."""

    def __init__(
        self,
        config: AppConfig,
        repo_root: Optional[Path] = None,
        ocr_engine: Optional[OcrEngine] = None,
        translator: Optional[Translator] = None,
        image_processor: Optional[ImageProcessor] = None,
        image_reviser: Optional[ImageReviser] = None,
    ) -> None:
        self.config = config
        self.repo_root = repo_root or _default_repo_root()
        self._ocr = ocr_engine
        self._translator = translator
        self._image_processor = image_processor
        self._image_reviser = image_reviser

    def ocr(self) -> Optional[OcrEngine]:
        if not self.config.ocr.enabled:
            return None
        if self._ocr is None:
            if self.config.ocr.engine.lower() != "paddleocr":
                raise ValueError(
                    f"Unsupported OCR engine '{self.config.ocr.engine}'. "
                    "Only 'paddleocr' is available."
                )
            self._ocr = PaddleOcrEngine(self.config.ocr)
        return self._ocr

    def translator(self) -> Optional[Translator]:
        if not self.config.translation.enabled:
            return None
        if self._translator is None:
            self._translator = create_image_pipeline_translator(
                self.config.translation,
                repo_root=self.repo_root,
            )
        return self._translator

    def image_processor(self) -> Optional[ImageProcessor]:
        if not self.config.imaging.enabled:
            return None
        if self._image_processor is None:
            self._image_processor = create_image_processor(self.config.imaging)
        return self._image_processor

    def image_reviser(self) -> Optional[ImageReviser]:
        if not self.config.revision.enabled:
            return None
        if self._image_reviser is None:
            self._image_reviser = ImageReviser(self.config.revision)
        return self._image_reviser

    def final_ocr_engine(self) -> Optional[OcrEngine]:
        return self.ocr()

    def backend_names(self) -> dict[str, str]:
        names = {}
        if self.config.ocr.enabled:
            names["ocr"] = self.ocr().name if self._ocr else "pending"
        if self.config.quality_control.enable_final_ocr and self.config.ocr.enabled:
            names["final_ocr"] = (
                f"{self.ocr().name}@min={self.config.quality_control.final_ocr_min_confidence}"
                if self._ocr
                else "pending"
            )
        if self.config.translation.enabled:
            names["translation"] = self.translator().name if self._translator else "pending"
        if self.config.imaging.enabled:
            names["imaging"] = self.image_processor().name if self._image_processor else "pending"
        if self.config.revision.enabled:
            names["revision"] = "image_reviser"
        return names


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

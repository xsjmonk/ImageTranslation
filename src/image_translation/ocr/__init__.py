"""OCR package."""

from .base import OcrEngine
from .models import OcrResult
from .paddle_ocr import PaddleOcrEngine

__all__ = [
    "OcrEngine",
    "OcrResult",
    "PaddleOcrEngine",
]

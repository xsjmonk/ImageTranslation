"""OCR domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OcrResult:
    """Raw OCR output before classification."""
    text: str
    confidence: float
    polygon: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    language: Optional[str] = None

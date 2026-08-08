"""OCR base interface – protocol that all OCR engines must satisfy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..models.text_region import TextRegion


class OcrEngine(ABC):
    """Abstract OCR engine interface.

    Implementations detect and recognize text in images,
    returning TextRegion objects with polygon geometry.
    """

    @abstractmethod
    def detect(self, image) -> List[TextRegion]:
        """Detect and recognize text regions in an image.

        Args:
            image: Image as numpy array (BGR format from OpenCV).

        Returns:
            List of TextRegion objects with text, confidence, and polygon.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...

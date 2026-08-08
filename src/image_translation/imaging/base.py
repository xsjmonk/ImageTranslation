"""Imaging base interface – protocol for image processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..models.text_region import TextRegion


class ImageProcessor(ABC):
    """Abstract image processor for text removal and background reconstruction."""

    @abstractmethod
    def generate_mask(
        self, image: np.ndarray, regions: List[TextRegion]
    ) -> np.ndarray:
        """Create a binary mask covering text regions to remove.

        Only 'translate' and 'remove' regions are eligible.

        Args:
            image: Source image (BGR).
            regions: Classified text regions.

        Returns:
            Binary mask (uint8) where 255 = text to remove.
        """
        ...

    @abstractmethod
    def remove_text(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Remove text from image using the mask via inpainting.

        Args:
            image: Source image (BGR).
            mask: Binary mask of text regions.

        Returns:
            Cleaned image with text removed.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable processor name."""
        ...

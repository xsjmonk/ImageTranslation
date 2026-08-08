"""Hybrid image processor – combines mask generation with inpainting."""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..config import ImagingConfig
from ..models.text_region import TextRegion
from .base import ImageProcessor
from .inpainting import inpaint_telea
from .text_mask import create_text_mask

logger = logging.getLogger(__name__)


class HybridImageProcessor(ImageProcessor):
    """Combines text mask generation with Telea inpainting for text removal."""

    def __init__(self, config: ImagingConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "hybrid"

    def generate_mask(
        self, image: np.ndarray, regions: List[TextRegion]
    ) -> np.ndarray:
        return create_text_mask(
            image,
            regions,
            expansion_pixels=self._config.mask_expansion_pixels,
        )

    def remove_text(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        return inpaint_telea(image, mask)

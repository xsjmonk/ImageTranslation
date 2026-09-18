"""Image processor factory — selects inpainting backend from configuration."""

from __future__ import annotations

import logging

from ..config import ImagingConfig
from .base import ImageProcessor
from .image_processor import HybridImageProcessor
from .inpainting import inpaint_navier_stokes, inpaint_telea
from .text_mask import create_text_mask

logger = logging.getLogger(__name__)


class OpenCvImageProcessor(HybridImageProcessor):
    """OpenCV Navier-Stokes inpainting backend."""

    @property
    def name(self) -> str:
        return "opencv"

    def remove_text(self, image, mask):
        return inpaint_navier_stokes(image, mask)


class EnhancedOpenCvImageProcessor(HybridImageProcessor):
    """Stronger OpenCV Navier-Stokes inpainting — not a neural DL model."""

    @property
    def name(self) -> str:
        return "enhanced_opencv"

    def remove_text(self, image, mask):
        return inpaint_navier_stokes(image, mask, radius=9)


# Backward-compatible alias; `neural` maps to enhanced OpenCV inpainting.
NeuralImageProcessor = EnhancedOpenCvImageProcessor


def create_image_processor(config: ImagingConfig) -> ImageProcessor:
    """Create an image processor for the configured backend."""
    processor = config.processor.lower()
    if processor == "opencv":
        return OpenCvImageProcessor(config)
    if processor in {"neural", "enhanced_opencv"}:
        logger.info(
            "Using enhanced OpenCV inpainting for processor='%s' (not a neural DL model)",
            processor,
        )
        return EnhancedOpenCvImageProcessor(config)
    return HybridImageProcessor(config)

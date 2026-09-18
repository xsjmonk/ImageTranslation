"""Imaging processor factory tests."""

from image_translation.config.models import ImagingConfig
from image_translation.imaging.factory import (
    HybridImageProcessor,
    NeuralImageProcessor,
    OpenCvImageProcessor,
    create_image_processor,
)


def test_hybrid_processor():
    proc = create_image_processor(ImagingConfig(processor="hybrid"))
    assert isinstance(proc, HybridImageProcessor)
    assert proc.name == "hybrid"


def test_opencv_processor():
    proc = create_image_processor(ImagingConfig(processor="opencv"))
    assert isinstance(proc, OpenCvImageProcessor)
    assert proc.name == "opencv"


def test_neural_processor_alias():
    proc = create_image_processor(ImagingConfig(processor="neural"))
    assert isinstance(proc, NeuralImageProcessor)
    assert proc.name == "enhanced_opencv"


def test_enhanced_opencv_processor():
    proc = create_image_processor(ImagingConfig(processor="enhanced_opencv"))
    assert proc.name == "enhanced_opencv"

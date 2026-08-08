"""Shared domain models for the image translation pipeline."""

from .text_region import TextRegion
from .image_job import ImageJob, JobStatus
from .processing_result import ProcessingResult

__all__ = [
    "TextRegion",
    "ImageJob",
    "JobStatus",
    "ProcessingResult",
]

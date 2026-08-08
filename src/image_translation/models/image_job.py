"""ImageJob – per-image processing unit that tracks state through the pipeline."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .text_region import TextRegion


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class ImageJob:
    """Carries one image through the pipeline from source to output."""

    __slots__ = (
        "source_path",
        "output_path",
        "metadata_path",
        "status",
        "text_regions",
        "error",
        "image_width",
        "image_height",
    )

    def __init__(
        self,
        source_path: Path,
        output_path: Path,
        metadata_path: Optional[Path] = None,
        status: JobStatus = JobStatus.pending,
        text_regions: Optional[List[TextRegion]] = None,
        error: Optional[str] = None,
        image_width: int = 0,
        image_height: int = 0,
    ) -> None:
        self.source_path = source_path
        self.output_path = output_path
        self.metadata_path = metadata_path
        self.status = status
        self.text_regions: List[TextRegion] = text_regions or []
        self.error = error
        self.image_width = image_width
        self.image_height = image_height

    @property
    def source_name(self) -> str:
        return self.source_path.name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "output_path": str(self.output_path),
            "width": self.image_width,
            "height": self.image_height,
            "status": self.status.value if isinstance(self.status, JobStatus) else self.status,
            "regions": [r.to_dict() for r in self.text_regions],
        }

    def mark_failed(self, error: str) -> None:
        self.status = JobStatus.failed
        self.error = error

    def mark_completed(self) -> None:
        self.status = JobStatus.completed
        self.error = None

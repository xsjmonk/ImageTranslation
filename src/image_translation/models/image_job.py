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
    partial = "partial"
    failed = "failed"
    skipped = "skipped"
    conflict = "conflict"


class ImageJob:
    """Carries one image through the pipeline from source to output."""

    __slots__ = (
        "source_path",
        "output_path",
        "preserved_original_path",
        "metadata_path",
        "status",
        "text_regions",
        "error",
        "image_width",
        "image_height",
        "metadata_qc",
        "final_qc",
        "inpaint_qc",
        "layouts",
        "qc_warnings",
    )

    def __init__(
        self,
        source_path: Path,
        output_path: Path,
        preserved_original_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        status: JobStatus = JobStatus.pending,
        text_regions: Optional[List[TextRegion]] = None,
        error: Optional[str] = None,
        image_width: int = 0,
        image_height: int = 0,
    ) -> None:
        self.source_path = source_path
        self.output_path = output_path
        self.preserved_original_path = preserved_original_path
        self.metadata_path = metadata_path
        self.status = status
        self.text_regions: List[TextRegion] = text_regions or []
        self.error = error
        self.image_width = image_width
        self.image_height = image_height
        self.metadata_qc: Optional[Dict[str, Any]] = None
        self.final_qc: Optional[Dict[str, Any]] = None
        self.inpaint_qc: Optional[Dict[str, Any]] = None
        self.layouts: Optional[List[Dict[str, Any]]] = None
        self.qc_warnings: List[str] = []

    @property
    def source_name(self) -> str:
        return self.source_path.name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "output_path": str(self.output_path),
            "localized_output_path": str(self.output_path),
            "preserved_original_path": (
                str(self.preserved_original_path) if self.preserved_original_path else None
            ),
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

    def mark_partial(self) -> None:
        self.status = JobStatus.partial
        self.error = None

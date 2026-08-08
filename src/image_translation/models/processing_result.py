"""ProcessingResult – aggregated result for a batch of images."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .image_job import ImageJob


@dataclass
class ProcessingResult:
    """Holds the outcome of processing one or more images."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    jobs: List[ImageJob] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return self.failed > 0 or len(self.errors) > 0

    def add_job(self, job: ImageJob) -> None:
        self.jobs.append(job)
        self.total += 1

    def summary(self) -> str:
        return (
            f"Processed: {self.total}  "
            f"Succeeded: {self.succeeded}  "
            f"Failed: {self.failed}  "
            f"Skipped: {self.skipped}"
        )

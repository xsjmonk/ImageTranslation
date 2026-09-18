"""Per-image and batch diagnostics for the image localization pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AppConfig
from .models.image_job import ImageJob, JobStatus
from .models.processing_result import ProcessingResult
from .utilities import json_utils


def config_fingerprint(config: AppConfig) -> str:
    """Stable short hash of effective configuration."""
    payload = config.model_dump(mode="json")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def build_image_diagnostic(
    job: ImageJob,
    config: AppConfig,
    *,
    started_at: float,
    backend_names: Dict[str, str],
    metadata_qc: Optional[dict] = None,
    final_qc: Optional[dict] = None,
    inpaint_qc: Optional[dict] = None,
    layouts: Optional[List[dict]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a machine-readable diagnostic record for one image."""
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    status = job.status.value if isinstance(job.status, JobStatus) else str(job.status)

    diagnostic: Dict[str, Any] = {
        "schema_version": "1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_path": str(job.source_path),
        "localized_output_path": str(job.output_path) if job.output_path else None,
        "preserved_original_path": (
            str(job.preserved_original_path) if job.preserved_original_path else None
        ),
        "output_path": str(job.output_path) if job.output_path else None,
        "dimensions": {"width": job.image_width, "height": job.image_height},
        "config_fingerprint": config_fingerprint(config),
        "elapsed_ms": elapsed_ms,
        "backends": backend_names,
        "regions": [r.to_dict() for r in job.text_regions],
        "quality_control": {
            "metadata": metadata_qc,
            "inpaint": inpaint_qc,
            "final": final_qc,
        },
        "layouts": layouts or [],
        "error": error,
    }
    if job.qc_warnings:
        diagnostic["warnings"] = list(job.qc_warnings)
    return diagnostic


def write_image_diagnostic(
    output_root: Path,
    job: ImageJob,
    diagnostic: Dict[str, Any],
    config: AppConfig,
) -> Path:
    """Write per-image diagnostic JSON (always when metadata enabled)."""
    meta_dir = output_root / "metadata"
    stem = job.output_path.stem if job.output_path else job.source_path.stem
    path = meta_dir / f"{stem}.json"
    json_utils.save_json_atomic(path, diagnostic)
    return path


def write_batch_summary(
    output_root: Path,
    result: ProcessingResult,
    config: AppConfig,
) -> Path:
    """Write batch summary.json at the output root."""
    summary = {
        "schema_version": "1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": config_fingerprint(config),
        "counts": {
            "total": result.total,
            "succeeded": result.succeeded,
            "partial": result.partial,
            "failed": result.failed,
            "skipped": result.skipped,
            "conflicts": result.conflicts,
        },
        "images": [
            {
                "source_path": str(job.source_path),
                "localized_output_path": str(job.output_path) if job.output_path else None,
                "preserved_original_path": (
                    str(job.preserved_original_path) if job.preserved_original_path else None
                ),
                "source": str(job.source_path),
                "output": str(job.output_path) if job.output_path else None,
                "status": job.status.value,
                "error": job.error,
            }
            for job in result.jobs
        ],
        "errors": result.errors,
        "warnings": result.warnings,
    }
    path = output_root / "summary.json"
    json_utils.save_json_atomic(path, summary)
    return path

"""Per-image and batch diagnostics for LocalImageProcessing."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contract import CONTRACT_VERSION


def build_image_diagnostic(
    *,
    status: str,
    source_path: Path,
    candidate_output_path: Optional[Path],
    preserved_original_path: Optional[Path],
    promoted_output_path: Optional[Path],
    regions: List[Dict[str, Any]],
    methods: List[Dict[str, Any]],
    validation_issues: List[Dict[str, Any]],
    warnings: List[str],
    error: Optional[str],
    started_at: float,
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    return {
        "schema_version": "1",
        "contract_version": CONTRACT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_path": str(source_path),
        "candidate_output_path": str(candidate_output_path) if candidate_output_path else None,
        "preserved_original_path": (
            str(preserved_original_path) if preserved_original_path else None
        ),
        "promoted_output_path": str(promoted_output_path) if promoted_output_path else None,
        "dimensions": {"width": image_width, "height": image_height},
        "elapsed_ms": elapsed_ms,
        "regions": regions,
        "processing_methods": methods,
        "validation_issues": validation_issues,
        "warnings": warnings,
        "error": error,
    }


def build_batch_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {
        "total": len(results),
        "success": 0,
        "partial": 0,
        "review": 0,
        "failure": 0,
        "skipped": 0,
    }
    for item in results:
        status = item.get("status", "failure")
        if status in counts:
            counts[status] += 1
    return {
        "schema_version": "1",
        "contract_version": CONTRACT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "images": results,
    }

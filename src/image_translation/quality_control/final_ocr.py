"""Final-image OCR validation — detects unwanted Chinese in translated regions."""

from __future__ import annotations

import re
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

from ..models.text_region import TextAction, TextRegion
from ..utilities.geometry import polygon_area, polygon_overlap_coverage
from ..utilities.image_format import bgr_view
from .types import QcIssue, QcResult, QcSeverity

if TYPE_CHECKING:
    from ..ocr.base import OcrEngine

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def validate_final_image_ocr(
    final_image: np.ndarray,
    regions: List[TextRegion],
    ocr_engine: "OcrEngine",
    *,
    min_confidence: float = 0.5,
    coverage_threshold: float = 0.12,
    overlap_threshold: float = 0.15,
) -> tuple[QcResult, Dict[str, Any]]:
    """Run OCR on the rendered image and reject residual Chinese in translate regions."""
    issues: List[QcIssue] = []
    evidence: Dict[str, Any] = {
        "backend": ocr_engine.name,
        "effective_min_confidence": min_confidence,
        "coverage_threshold": coverage_threshold,
        "overlap_threshold": overlap_threshold,
        "detections": [],
        "region_matches": [],
    }

    translate_regions = [r for r in regions if r.action == TextAction.translate]
    if not translate_regions:
        return QcResult(passed=True, issues=issues, evidence=evidence), evidence

    detections = ocr_engine.detect(bgr_view(final_image), min_confidence=min_confidence)
    evidence["detections"] = [
        {
            "text": det.source_text,
            "confidence": det.confidence,
            "polygon": det.polygon,
            "has_cjk": bool(_CJK_RE.search(det.source_text)),
        }
        for det in detections
    ]

    for region in translate_regions:
        matched: List[dict] = []
        cjk_area = 0.0
        region_area = max(polygon_area(region.polygon), 1.0)

        for det in detections:
            if not _CJK_RE.search(det.source_text):
                continue
            overlap = polygon_overlap_coverage(region.polygon, det.polygon)
            if overlap < overlap_threshold:
                continue
            matched.append(
                {
                    "text": det.source_text,
                    "confidence": det.confidence,
                    "overlap": overlap,
                }
            )
            cjk_area += polygon_area(det.polygon) * overlap

        coverage = min(1.0, cjk_area / region_area)
        evidence["region_matches"].append(
            {
                "region_id": region.id,
                "source_text": region.source_text,
                "matched_detections": matched,
                "cjk_coverage": coverage,
            }
        )

        if matched and coverage >= coverage_threshold:
            sample = matched[0]["text"][:40]
            issues.append(
                QcIssue(
                    severity=QcSeverity.error,
                    code="residual_chinese_in_output",
                    message=(
                        f"Final-image OCR detected Chinese ({sample}) in translated region "
                        f"for: {region.source_text[:40]}"
                    ),
                    region_id=region.id,
                )
            )

    passed = not any(i.severity == QcSeverity.error for i in issues)
    return QcResult(passed=passed, issues=issues, evidence=evidence), evidence

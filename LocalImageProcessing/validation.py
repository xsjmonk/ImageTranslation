"""Manifest and geometry validation without OCR or translation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .contract import ImageManifest, RegionAction, RegionManifest


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    region_id: str = ""


@dataclass
class ValidationResult:
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)


def validate_image_manifest(
    manifest: ImageManifest,
    image_width: int,
    image_height: int,
) -> ValidationResult:
    """Validate region geometry and required fields."""
    issues: List[ValidationIssue] = []

    for region in manifest.regions:
        issues.extend(_validate_region_geometry(region, image_width, image_height))
        if region.action == RegionAction.translate:
            text = (region.translated_text or "").strip()
            if not text:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="missing_translated_text",
                        message=f"translate region '{region.id}' is missing translated_text",
                        region_id=region.id,
                    )
                )

    passed = not any(issue.severity == "error" for issue in issues)
    return ValidationResult(passed=passed, issues=issues)


def _validate_region_geometry(
    region: RegionManifest,
    width: int,
    height: int,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    xs = [point[0] for point in region.polygon]
    ys = [point[1] for point in region.polygon]

    if any(x < 0 or x > width for x in xs):
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_geometry",
                message=f"region '{region.id}' polygon x coordinates exceed image bounds",
                region_id=region.id,
            )
        )
    if any(y < 0 or y > height for y in ys):
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_geometry",
                message=f"region '{region.id}' polygon y coordinates exceed image bounds",
                region_id=region.id,
            )
        )

    if max(xs) - min(xs) < 1 or max(ys) - min(ys) < 1:
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_geometry",
                message=f"region '{region.id}' polygon is too small",
                region_id=region.id,
            )
        )
    return issues

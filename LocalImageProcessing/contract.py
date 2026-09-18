"""Versioned manifest contract for agent-supplied image operations."""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

CONTRACT_VERSION = "1.0"
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
}


class RegionAction(str, Enum):
    translate = "translate"
    remove = "remove"
    preserve = "preserve"
    review = "review"


class ReconstructionMethod(str, Enum):
    inpaint = "inpaint"
    blur_fallback = "blur_fallback"


class RegionStyle(BaseModel):
    alignment: Optional[str] = None
    font_path: Optional[str] = None
    font_size_hint: Optional[int] = None
    color_rgb: Optional[tuple[int, int, int]] = None
    opacity: Optional[float] = None
    bold: Optional[bool] = None
    stroke_width: Optional[int] = None
    stroke_color_rgb: Optional[tuple[int, int, int]] = None
    shadow_offset: Optional[tuple[int, int]] = None
    shadow_blur: Optional[int] = None
    shadow_color_rgb: Optional[tuple[int, int, int]] = None
    shadow_opacity: Optional[float] = None
    gradient_top_rgb: Optional[tuple[int, int, int]] = None
    gradient_bottom_rgb: Optional[tuple[int, int, int]] = None

    model_config = {"extra": "allow"}


class RegionReconstruction(BaseModel):
    method: ReconstructionMethod = ReconstructionMethod.inpaint
    blur_sigma: float = Field(default=12.0, gt=0)

    model_config = {"extra": "allow"}


class RegionManifest(BaseModel):
    id: str
    polygon: List[List[float]]
    orientation_degrees: float = 0.0
    action: RegionAction
    source_text: str = ""
    translated_text: Optional[str] = None
    style: Optional[RegionStyle] = None
    reconstruction: Optional[RegionReconstruction] = None

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value: List[List[float]]) -> List[List[float]]:
        if len(value) < 3:
            raise ValueError("polygon must contain at least 3 points")
        for point in value:
            if len(point) != 2:
                raise ValueError("each polygon point must be [x, y]")
            float(point[0])
            float(point[1])
        return value

    @model_validator(mode="after")
    def validate_action_requirements(self) -> "RegionManifest":
        if self.action == RegionAction.translate:
            text = (self.translated_text or "").strip()
            if not text:
                raise ValueError(
                    f"region '{self.id}': translated_text is required for translate action"
                )
        return self


class ImageManifest(BaseModel):
    source_path: Path
    source_hash: Optional[str] = None
    regions: List[RegionManifest] = Field(default_factory=list)

    @field_validator("source_path", mode="before")
    @classmethod
    def expand_source(cls, value: Union[str, Path]) -> Path:
        return Path(value).expanduser()

    @model_validator(mode="after")
    def validate_source_extension(self) -> "ImageManifest":
        if self.source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"unsupported image extension for {self.source_path.name}"
            )
        return self


class BatchManifest(BaseModel):
    contract_version: str = CONTRACT_VERSION
    images: List[ImageManifest]

    @field_validator("contract_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != CONTRACT_VERSION:
            raise ValueError(
                f"unsupported contract_version '{value}'; expected {CONTRACT_VERSION}"
            )
        return value

    @model_validator(mode="after")
    def validate_non_empty(self) -> "BatchManifest":
        if not self.images:
            raise ValueError("manifest must contain at least one image")
        return self


class SingleImageManifest(BaseModel):
    """Convenience wrapper for one-image manifests."""

    contract_version: str = CONTRACT_VERSION
    source_path: Path
    source_hash: Optional[str] = None
    regions: List[RegionManifest] = Field(default_factory=list)

    def to_batch(self) -> BatchManifest:
        return BatchManifest(
            contract_version=self.contract_version,
            images=[
                ImageManifest(
                    source_path=self.source_path,
                    source_hash=self.source_hash,
                    regions=self.regions,
                )
            ],
        )


def load_manifest(data: Dict[str, Any]) -> BatchManifest:
    """Parse a manifest dict as single-image or batch form."""
    version = data.get("contract_version", CONTRACT_VERSION)
    if "images" in data:
        return BatchManifest.model_validate(data)
    return SingleImageManifest.model_validate(data).to_batch()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_hash(path: Path, expected: Optional[str]) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"source hash mismatch for {path.name}: expected {expected}, got {actual}"
        )

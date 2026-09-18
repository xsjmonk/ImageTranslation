"""Internal region model for pixel processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .contract import RegionAction, RegionManifest


@dataclass
class ProcessRegion:
    id: str
    source_text: str
    polygon: List[List[float]]
    action: RegionAction
    translated_text: str = ""
    orientation: float = 0.0
    style: Dict[str, Any] = field(default_factory=dict)

    @property
    def translation(self) -> Dict[str, str]:
        if not self.translated_text:
            return {}
        return {"translated_text": self.translated_text}


def from_manifest(region: RegionManifest) -> ProcessRegion:
    style = region.style.model_dump(exclude_none=True) if region.style else {}
    return ProcessRegion(
        id=region.id,
        source_text=region.source_text,
        polygon=region.polygon,
        action=region.action,
        translated_text=(region.translated_text or "").strip(),
        orientation=region.orientation_degrees,
        style=style,
    )

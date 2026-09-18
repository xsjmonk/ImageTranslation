"""Text style attributes for rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TextStyle:
    color_rgb: Tuple[int, int, int] = (255, 255, 255)
    opacity: float = 1.0
    bold: bool = False
    italic_oblique: float = 0.0
    stroke_width: int = 0
    stroke_color_rgb: Tuple[int, int, int] = (0, 0, 0)
    shadow_offset: Tuple[int, int] = (0, 0)
    shadow_blur: int = 0
    shadow_color_rgb: Tuple[int, int, int] = (0, 0, 0)
    shadow_opacity: float = 0.0
    gradient_top_rgb: Optional[Tuple[int, int, int]] = None
    gradient_bottom_rgb: Optional[Tuple[int, int, int]] = None
    alignment: str = "center"
    line_spacing: float = 1.2
    recovered_effects: List[str] = field(default_factory=list)
    review_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

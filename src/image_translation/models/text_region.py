"""TextRegion – shared geometry + classification model used across modules."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional


class TextAction(str, Enum):
    translate = "translate"
    preserve = "preserve"
    remove = "remove"
    review = "review"


class TextRegion:
    """A single detected text block with OCR polygon, classification, and translation."""

    __slots__ = (
        "id",
        "source_text",
        "confidence",
        "polygon",
        "language",
        "action",
        "action_reason",
        "translation",
    )

    def __init__(
        self,
        id: str,
        source_text: str,
        confidence: float,
        polygon: List[List[float]],
        language: Optional[str] = None,
        action: TextAction = TextAction.review,
        action_reason: str = "",
        translation: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.id = id
        self.source_text = source_text
        self.confidence = float(confidence)
        self.polygon = polygon  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4], ...]
        self.language = language
        self.action = action
        self.action_reason = action_reason
        self.translation = translation or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "polygon": self.polygon,
            "language": self.language,
            "action": self.action.value if isinstance(self.action, TextAction) else self.action,
            "action_reason": self.action_reason,
            "translation": self.translation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextRegion":
        return cls(
            id=data["id"],
            source_text=data["source_text"],
            confidence=data["confidence"],
            polygon=data["polygon"],
            language=data.get("language"),
            action=TextAction(data.get("action", "review")),
            action_reason=data.get("action_reason", ""),
            translation=data.get("translation", {}),
        )

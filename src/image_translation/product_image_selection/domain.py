from dataclasses import dataclass
from enum import Enum

class ProductImageLabel(str, Enum):
    TAKEN = "taken"
    NOT_TAKEN = "not_taken"

@dataclass(frozen=True)
class LabeledImageRecord:
    image_path: str
    label: ProductImageLabel
    product_group_id: str
    image_id: str

@dataclass(frozen=True)
class ImagePrediction:
    image_id: str
    taken_probability: float
    not_taken_probability: float
    label: ProductImageLabel
    requires_review: bool
    model_version: str
    image_path: str = ""

@dataclass(frozen=True)
class ImageFailure:
    image_id: str
    image_path: str
    error: str

@dataclass(frozen=True)
class CategorizationResult:
    predictions: tuple[ImagePrediction, ...]
    failures: tuple[ImageFailure, ...]

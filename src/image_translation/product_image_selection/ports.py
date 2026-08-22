from typing import Protocol, Sequence
import numpy as np
from .domain import CategorizationResult, LabeledImageRecord

class RasterReader(Protocol):
    def read_rgb_pixels(self, image_id: str) -> np.ndarray: ...
class LabelManifestReader(Protocol):
    def read(self) -> Sequence[LabeledImageRecord]: ...
class PredictionWriter(Protocol):
    def write(self, result: CategorizationResult) -> None: ...

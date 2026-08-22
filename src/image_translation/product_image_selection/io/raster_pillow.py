from pathlib import Path
import numpy as np
from PIL import Image
class PillowRasterReader:
    def __init__(self, max_decoded_pixels: int): self.max_decoded_pixels = max_decoded_pixels
    def read_rgb_pixels(self, image_id: str) -> np.ndarray:
        with Image.open(Path(image_id)) as image:
            if image.width * image.height > self.max_decoded_pixels: raise ValueError("decoded pixel limit exceeded")
            image.load()
            return np.asarray(image.convert("RGB")).copy()

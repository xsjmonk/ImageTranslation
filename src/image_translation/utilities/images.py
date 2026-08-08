"""Image I/O utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


def load_image(path: Path):
    """Load an image and return a numpy array (BGR if OpenCV, RGB if Pillow).

    Uses OpenCV for speed; falls back to Pillow.
    """
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img
    except ImportError:
        from PIL import Image
        import numpy as np
        pil_img = Image.open(path).convert("RGB")
        return np.array(pil_img)[:, :, ::-1]  # RGB -> BGR for consistency


def save_image(path: Path, image, lossless: bool = False) -> None:
    """Save an image (numpy array BGR) to disk.

    Uses PNG for lossless intermediate artifacts, JPEG otherwise.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        cv2.imwrite(str(p), image)
    except ImportError:
        from PIL import Image
        import numpy as np
        if image.shape[2] == 3:
            rgb = image[:, :, ::-1]  # BGR -> RGB
        else:
            rgb = image
        pil_img = Image.fromarray(rgb)
        pil_img.save(str(p))


def read_dimensions(path: Path) -> Tuple[int, int]:
    """Return (width, height) of an image without loading full pixels."""
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        h, w = img.shape[:2]
        return w, h
    except ImportError:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (width, height)


def validate_readable_image(path: Path) -> bool:
    """Check that a file can be opened as a readable image."""
    try:
        read_dimensions(path)
        return True
    except Exception:
        return False


def preserve_dimensions(original_path: Path, output_path: Path) -> bool:
    """Verify output image has the same dimensions as the original.

    Returns True if dimensions match.
    """
    ow, oh = read_dimensions(original_path)
    nw, nh = read_dimensions(output_path)
    return ow == nw and oh == nh

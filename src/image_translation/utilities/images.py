"""Image I/O utilities with alpha and color-mode preservation."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

from .image_format import ImagePayload, supports_alpha


def load_image(path: Path) -> ImagePayload:
    """Load an image preserving alpha when present."""
    suffix = path.suffix.lower()
    try:
        import cv2
        flag = cv2.IMREAD_UNCHANGED if supports_alpha(path) else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), flag)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        has_alpha = img.ndim == 3 and img.shape[2] == 4
        return ImagePayload(
            pixels=img,
            has_alpha=has_alpha,
            color_mode="BGRA" if has_alpha else "BGR",
            source_suffix=suffix,
        )
    except ImportError:
        from PIL import Image
        import numpy as np

        pil_img = Image.open(path)
        mode = pil_img.mode
        if mode in ("RGBA", "LA"):
            rgba = np.array(pil_img.convert("RGBA"))
            bgra = rgba[:, :, [2, 1, 0, 3]]
            return ImagePayload(bgra, True, "BGRA", suffix)
        rgb = np.array(pil_img.convert("RGB"))
        return ImagePayload(rgb[:, :, ::-1], False, "BGR", suffix)


def image_array(payload_or_array: Union[ImagePayload, object]):
    if isinstance(payload_or_array, ImagePayload):
        return payload_or_array.pixels
    return payload_or_array


def save_image_atomic(
    path: Path,
    image: Union[ImagePayload, object],
    *,
    has_alpha: bool = False,
    lossless: bool = False,
) -> None:
    """Save an image atomically via a temporary sibling file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.stem}.tmp{p.suffix}")
    save_image(tmp, image, has_alpha=has_alpha, lossless=lossless)
    tmp.replace(p)


def save_image(
    path: Path,
    image: Union[ImagePayload, object],
    *,
    has_alpha: bool = False,
    lossless: bool = False,
) -> None:
    """Save an image to disk, preserving alpha for PNG/WebP when requested."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = image if isinstance(image, ImagePayload) else ImagePayload(image, has_alpha, "BGRA" if has_alpha else "BGR")
    arr = payload.pixels
    write_alpha = payload.has_alpha or has_alpha
    suffix = p.suffix.lower()
    force_lossless = lossless or suffix in {".png", ".webp", ".tif", ".tiff"}

    try:
        import cv2
        if write_alpha and arr.ndim == 3 and arr.shape[2] == 4:
            cv2.imwrite(str(p), arr)
            return
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if force_lossless and suffix == ".png":
            cv2.imwrite(str(p), arr)
            return
        cv2.imwrite(str(p), arr)
        return
    except ImportError:
        pass

    from PIL import Image
    import numpy as np

    if write_alpha and arr.ndim == 3 and arr.shape[2] == 4:
        rgba = arr[:, :, [2, 1, 0, 3]]
        Image.fromarray(rgba, mode="RGBA").save(str(p))
        return
    rgb = arr[:, :, ::-1] if arr.ndim == 3 else arr
    Image.fromarray(rgb).save(str(p))


def read_dimensions(path: Path) -> Tuple[int, int]:
    """Return (width, height) of an image without loading full pixels."""
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        h, w = img.shape[:2]
        return w, h
    except ImportError:
        from PIL import Image
        with Image.open(path) as im:
            return im.size


def validate_readable_image(path: Path) -> bool:
    """Check that a file can be opened as a readable image."""
    try:
        read_dimensions(path)
        return True
    except Exception:
        return False


def preserve_dimensions(original_path: Path, output_path: Path) -> bool:
    """Verify output image has the same dimensions as the original."""
    ow, oh = read_dimensions(original_path)
    nw, nh = read_dimensions(output_path)
    return ow == nw and oh == nh

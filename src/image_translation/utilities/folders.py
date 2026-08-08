"""Folder utilities – enumeration, output derivation, recursive support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Set


def folder_exists(path: Path) -> bool:
    """Check if a path exists and is a directory."""
    return Path(path).is_dir()


def create_folder(path: Path) -> None:
    """Create a folder (and parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def enumerate_images(
    folder: Path,
    extensions: Set[str],
    recursive: bool = False,
) -> List[Path]:
    """List image files in a folder, optionally recursive.

    Args:
        folder: Root folder to scan.
        extensions: Lowercase extensions including dot (e.g. {'.jpg', '.png'}).
        recursive: If True, scan subdirectories (excluding _processed dirs).

    Returns:
        Sorted list of image Paths.
    """
    images: List[Path] = []

    if recursive:
        for root, dirs, files in os.walk(str(folder)):
            # Exclude _processed directories
            dirs[:] = [d for d in dirs if not d.endswith("_processed")]
            root_path = Path(root)
            for fname in files:
                if Path(fname).suffix.lower() in extensions:
                    images.append(root_path / fname)
    else:
        for entry in sorted(folder.iterdir()):
            if entry.is_file() and entry.suffix.lower() in extensions:
                images.append(entry)

    # Natural sort
    images.sort(key=lambda p: _natural_key(p.name))
    return images


def _natural_key(name: str) -> list:
    """Key function for natural string sorting (e.g. '2' before '10')."""
    import re
    parts = re.split(r"(\d+)", name)
    result: list = []
    for part in parts:
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part.lower())
    return result


def derive_output_folder(input_folder: Path) -> Path:
    """Derive <folder>_processed in the same parent."""
    return input_folder.parent / f"{input_folder.name}_processed"


def derive_output_path(source: Path, output_folder: Path, input_folder: Optional[Path] = None) -> Path:
    """Map a source image to its output path, preserving relative structure.

    Args:
        source: Source image path.
        output_folder: Target output root folder.
        input_folder: If provided and recursive, preserve relative subfolder.

    Returns:
        Output path with same filename.
    """
    if input_folder is not None:
        try:
            relative = source.relative_to(input_folder)
            return output_folder / relative
        except ValueError:
            pass
    return output_folder / source.name

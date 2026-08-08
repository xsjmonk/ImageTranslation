"""Input resolver – validates paths, determines input type, derives output folder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Set

from .models import AppInput, InputType

# Supported image extensions (lowercase, with dot)
SUPPORTED_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
}


class InputError(Exception):
    """Raised when input validation fails."""


def resolve_input(
    parsed: argparse.Namespace,
    extensions: Optional[Set[str]] = None,
) -> AppInput:
    """Validate and resolve user-provided input into a typed AppInput.

    Args:
        parsed: Parsed argument namespace from arguments.py.
        extensions: Optional override for supported image extensions.

    Returns:
        A fully resolved AppInput.

    Raises:
        InputError: If the path does not exist or is invalid.
    """
    exts = extensions or SUPPORTED_EXTENSIONS
    input_path = parsed.input_path.resolve()

    if not input_path.exists():
        raise InputError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        # Single image mode
        ext = input_path.suffix.lower()
        if ext not in exts:
            raise InputError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(exts))}"
            )
        output_folder = _derive_output_for_file(input_path)
        return AppInput(
            input_path=input_path,
            input_type=InputType.SINGLE_IMAGE,
            input_folder=None,
            single_image_path=input_path,
            output_folder=output_folder,
            config_path=parsed.config_path,
        )

    # Folder mode
    output_folder = _derive_output_for_folder(input_path)
    return AppInput(
        input_path=input_path,
        input_type=InputType.FOLDER,
        input_folder=input_path,
        single_image_path=None,
        output_folder=output_folder,
        config_path=parsed.config_path,
    )


def _derive_output_for_folder(folder: Path) -> Path:
    """Derive <folder>_processed in the same parent directory."""
    parent = folder.parent
    name = folder.name
    return parent / f"{name}_processed"


def _derive_output_for_file(file_path: Path) -> Path:
    """Derive <parent>_processed/<filename>."""
    parent = file_path.parent
    parent_name = parent.name
    grandparent = parent.parent
    output_folder = grandparent / f"{parent_name}_processed"
    return output_folder / file_path.name

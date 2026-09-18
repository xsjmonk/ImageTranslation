"""Input resolver – validates paths, determines input type, derives output folder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Set

from ..config import AppConfig
from ..utilities.output_paths import (
    OutputPathError,
    resolve_output_file_path,
    resolve_output_root,
    validate_input_output_paths,
)
from .models import AppInput, InputType

SUPPORTED_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
}


class InputError(Exception):
    """Raised when input validation fails."""


def resolve_input(
    parsed: argparse.Namespace,
    config: Optional[AppConfig] = None,
    extensions: Optional[Set[str]] = None,
) -> AppInput:
    """Validate and resolve user-provided input into a typed AppInput."""
    exts = extensions or SUPPORTED_EXTENSIONS
    input_path = parsed.input_path.resolve()
    config = config or AppConfig()

    if not input_path.exists():
        raise InputError(f"Input path does not exist: {input_path}")

    cli_output = getattr(parsed, "output_folder", None)

    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext not in exts:
            raise InputError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(exts))}"
            )
        output_root = resolve_output_root(
            input_path, InputType.SINGLE_IMAGE, config, cli_output
        )
        validate_input_output_paths(input_path, output_root, config)
        return AppInput(
            input_path=input_path,
            input_type=InputType.SINGLE_IMAGE,
            input_folder=None,
            single_image_path=input_path,
            output_folder=output_root,
            config_path=parsed.config_path,
        )

    output_root = resolve_output_root(
        input_path, InputType.FOLDER, config, cli_output
    )
    validate_input_output_paths(input_path, output_root, config)
    return AppInput(
        input_path=input_path,
        input_type=InputType.FOLDER,
        input_folder=input_path,
        single_image_path=None,
        output_folder=output_root,
        config_path=parsed.config_path,
    )

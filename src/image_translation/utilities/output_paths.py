"""Output path resolution with CLI/config precedence."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config.models import AppConfig, OutputConfig
from ..input.models import InputType


class OutputPathError(Exception):
    """Raised when input/output paths are invalid or unsafe."""


def resolve_output_root(
    input_path: Path,
    input_type: InputType,
    config: AppConfig,
    cli_output_folder: Optional[Path] = None,
) -> Path:
    """Resolve the output root directory.

    Precedence: explicit CLI output > config.output.directory > derived sibling.
    """
    if cli_output_folder is not None:
        return cli_output_folder.resolve()

    if config.output.directory:
        configured = Path(config.output.directory)
        if not configured.is_absolute():
            base = input_path.parent if input_type == InputType.SINGLE_IMAGE else input_path
            return (base / configured).resolve()
        return configured.resolve()

    if input_type == InputType.SINGLE_IMAGE:
        parent = input_path.parent
        return (parent.parent / f"{parent.name}{config.output.suffix}").resolve()

    return (input_path.parent / f"{input_path.name}{config.output.suffix}").resolve()


ORIGINAL_PREFIX = "original_"


def preserved_original_basename(source_filename: str) -> str:
    """Return the preserved-original filename with idempotent prefix rules."""
    if source_filename.startswith(ORIGINAL_PREFIX):
        return source_filename
    return f"{ORIGINAL_PREFIX}{source_filename}"


def localized_output_basename(source_filename: str) -> str:
    """Return the localized output filename.

    When the source already uses the ``original_`` prefix, the localized
    product image is written without that prefix to avoid colliding with the
    preserved byte-identical copy (``original_01.jpg`` -> localized ``01.jpg``).
    """
    if source_filename.startswith(ORIGINAL_PREFIX):
        return source_filename[len(ORIGINAL_PREFIX) :]
    return source_filename


def _output_dir_for_source(
    source: Path,
    output_root: Path,
    input_root: Optional[Path],
) -> Path:
    if input_root is not None:
        try:
            return output_root / source.relative_to(input_root).parent
        except ValueError:
            pass
    return output_root


def resolve_output_file_path(
    source: Path,
    output_root: Path,
    input_root: Optional[Path],
    config: OutputConfig,
) -> Path:
    """Map a source image to its localized output file path."""
    out_dir = _output_dir_for_source(source, output_root, input_root)
    basename = localized_output_basename(source.name)
    target = out_dir / basename

    if not config.preserve_filename:
        target = target.with_name(f"{target.stem}_en{target.suffix}")
    return target


def resolve_preserved_original_path(
    source: Path,
    output_root: Path,
    input_root: Optional[Path],
) -> Path:
    """Map a source image to its preserved original copy path."""
    out_dir = _output_dir_for_source(source, output_root, input_root)
    return out_dir / preserved_original_basename(source.name)


def validate_input_output_paths(
    input_path: Path,
    output_root: Path,
    config: AppConfig,
) -> None:
    """Ensure input and output do not collide unless explicitly allowed."""
    input_resolved = input_path.resolve()
    output_resolved = output_root.resolve()

    same = input_resolved == output_resolved
    nested_input_in_output = input_resolved.is_relative_to(output_resolved)
    nested_output_in_input = output_resolved.is_relative_to(input_resolved)

    if (same or nested_input_in_output or nested_output_in_input) and not config.output.allow_same_as_input:
        raise OutputPathError(
            "Input and output resolve to the same location. "
            "Set output.allow_same_as_input=true to permit in-place processing."
        )

"""Output path helpers owned by LocalImageProcessing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

ORIGINAL_PREFIX = "original_"


def preserved_original_basename(source_filename: str) -> str:
    if source_filename.startswith(ORIGINAL_PREFIX):
        return source_filename
    return f"{ORIGINAL_PREFIX}{source_filename}"


def localized_output_basename(source_filename: str) -> str:
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
    input_root: Optional[Path] = None,
    *,
    preserve_filename: bool = True,
) -> Path:
    out_dir = _output_dir_for_source(source, output_root, input_root)
    target = out_dir / localized_output_basename(source.name)
    if not preserve_filename:
        target = target.with_name(f"{target.stem}_en{target.suffix}")
    return target


def resolve_preserved_original_path(
    source: Path,
    output_root: Path,
    input_root: Optional[Path] = None,
) -> Path:
    out_dir = _output_dir_for_source(source, output_root, input_root)
    return out_dir / preserved_original_basename(source.name)

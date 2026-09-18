"""Atomic file operations."""

from __future__ import annotations

import shutil
from pathlib import Path


def ensure_parent_folder(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def bytes_equal(path_a: Path, path_b: Path) -> bool:
    return Path(path_a).read_bytes() == Path(path_b).read_bytes()


def copy_file_atomic(source: Path, destination: Path) -> None:
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.stem}.tmp{dest.suffix}")
    shutil.copy2(source, tmp)
    tmp.replace(dest)

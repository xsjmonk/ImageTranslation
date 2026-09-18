"""File I/O utilities with UTF-8 defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    """Read a text file as UTF-8."""
    return Path(path).read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    """Write text as UTF-8, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def read_bytes(path: Path) -> bytes:
    """Read a file as raw bytes."""
    return Path(path).read_bytes()


def write_bytes(path: Path, content: bytes) -> None:
    """Write raw bytes, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def file_exists(path: Path) -> bool:
    """Check if a path exists and is a file."""
    return Path(path).is_file()


def ensure_parent_folder(path: Path) -> None:
    """Create the parent directory of a path if it does not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def bytes_equal(path_a: Path, path_b: Path) -> bool:
    """Return True when two files contain identical bytes."""
    return Path(path_a).read_bytes() == Path(path_b).read_bytes()


def copy_file_atomic(source: Path, destination: Path) -> None:
    """Copy a file byte-for-byte via a temporary sibling file."""
    import shutil

    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.stem}.tmp{dest.suffix}")
    shutil.copy2(source, tmp)
    tmp.replace(dest)

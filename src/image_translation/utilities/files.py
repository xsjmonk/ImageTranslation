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

"""JSON utilities – load/save with UTF-8 and readable Chinese text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    """Load JSON from a file, UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any, pretty: bool = True) -> None:
    """Save data as JSON, UTF-8 with readable non-ASCII (ensure_ascii=False).

    Args:
        path: Output file path.
        data: JSON-serializable data.
        pretty: If True, indent for readability.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2 if pretty else None,
            default=str,
        )


def save_json_atomic(path: Path, data: Any, pretty: bool = True) -> None:
    """Write JSON atomically via a temporary sibling file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2 if pretty else None,
            default=str,
        )
    tmp.replace(p)

"""JSON load/save helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json_atomic(path: Path, data: Any, pretty: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2 if pretty else None,
            default=str,
        )
    tmp.replace(p)

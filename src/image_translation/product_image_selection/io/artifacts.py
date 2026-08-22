from __future__ import annotations
import hashlib, json, os
from pathlib import Path
def config_sha256(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def atomic_json(path: str | Path, value) -> None:
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp,path)
def run_directory(output_directory, run_name, overwrite=False):
    path=Path(output_directory)/run_name
    if path.exists() and not overwrite: raise FileExistsError(f"run already exists: {path}")
    path.mkdir(parents=True, exist_ok=overwrite); return path

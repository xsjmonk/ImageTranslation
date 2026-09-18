"""CLI entry point for LocalImageProcessing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError

from .contract import load_manifest
from .env_check import format_environment_errors, run_environment_check
from .exit_codes import exit_code_from_summary
from .paths import ensure_import_paths, resolve_repo_root
from .processor import ProcessingOptions, process_batch

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _default_output_root(manifest_path: Path) -> Path:
    source = manifest_path.resolve().parent
    return source.parent / f"{source.name}_processed"


@app.command("process")
def process_command(
    manifest: Path = typer.Option(..., "--manifest", "-m", help="Path to manifest JSON"),
    output_folder: Optional[Path] = typer.Option(
        None, "--output-folder", "-o", help="Output/archive root"
    ),
    promote: bool = typer.Option(
        False,
        "--promote",
        help="Replace source images after successful processing",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing localized outputs",
    ),
) -> None:
    ensure_import_paths()
    manifest_path = manifest.expanduser().resolve()
    if not manifest_path.is_file():
        typer.echo(f"[ERROR] manifest not found: {manifest_path}", err=True)
        raise typer.Exit(code=1)

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        typer.echo(f"[ERROR] malformed manifest JSON: {exc}", err=True)
        raise typer.Exit(code=1)

    try:
        batch = load_manifest(data)
    except (ValidationError, ValueError) as exc:
        typer.echo(f"[ERROR] invalid manifest: {exc}", err=True)
        raise typer.Exit(code=1)

    output_root = (output_folder or _default_output_root(manifest_path)).expanduser().resolve()

    summary = process_batch(
        batch,
        ProcessingOptions(
            output_root=output_root,
            promote=promote,
            overwrite_existing=overwrite,
        ),
    )
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
    raise typer.Exit(code=exit_code_from_summary(summary))


@app.command("check-env")
def check_env_command() -> None:
    ensure_import_paths()
    issues = run_environment_check()
    if issues:
        typer.echo(format_environment_errors(issues), err=True)
        raise typer.Exit(code=1)
    typer.echo(f"[OK] LocalImageProcessing environment ready ({resolve_repo_root()})")


def main(argv: Optional[list[str]] = None) -> None:
    app(argv or sys.argv[1:])

"""CLI entry point: python -m image_translation.local_manifest"""

from __future__ import annotations

import sys

import typer

from .env_check import check_local_image_environment, format_environment_errors

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local manifest image processor (deterministic pixel operations only).",
)


def _require_environment() -> None:
    issues = check_local_image_environment()
    if issues:
        typer.echo(format_environment_errors(issues), err=True)
        raise typer.Exit(code=1)


@app.command("check-env")
def check_env() -> None:
    """Verify the dp conda environment has local image-processing packages."""
    issues = check_local_image_environment()
    if issues:
        typer.echo(format_environment_errors(issues), err=True)
        raise typer.Exit(code=1)
    typer.echo("Local image-processing environment OK.")


@app.command("process-image-manifest")
def process_image_manifest(
    manifest: str = typer.Option(..., "--manifest", help="Path to image or batch manifest JSON."),
    output: str | None = typer.Option(None, "--output", help="Output folder override."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate manifest only; do not modify images."),
) -> None:
    """Process one image manifest or a batch manifest."""
    _require_environment()
    typer.echo(
        "Manifest processing is not implemented yet. "
        "Environment setup is complete; use --dry-run after validation is added.",
        err=True,
    )
    typer.echo(f"manifest={manifest}", err=True)
    if output:
        typer.echo(f"output={output}", err=True)
    if dry_run:
        typer.echo("dry_run=true", err=True)
    raise typer.Exit(code=2)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["--check-env"]:
        check_env()
        return
    app(argv)


if __name__ == "__main__":
    main()

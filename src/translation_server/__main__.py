"""Entry point — python -m translation_server."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="translation_server",
        description="Standalone M2M100 GPU translation API server",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to translation-server.config.json",
    )
    parser.add_argument(
        "--check-cache",
        action="store_true",
        help="Validate the configured model cache (resolution + snapshot "
             "completeness + offline policy) without loading the model, "
             "print the resolved snapshot, and exit 0/1.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else None

    # Load config + build runtime (validates config before importing heavy deps)
    from .config import load_server_config
    try:
        server_config = load_server_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] Config error: {e}", file=sys.stderr)
        return 1

    # Setup logging
    log_level = getattr(logging, server_config.server.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Config-derived cache diagnostics (no resolution, no model load)
    from .runtime import TranslationRuntime
    diag = TranslationRuntime(server_config).cache_diagnostics()
    logging.getLogger("translation_server").info(
        "[INFO] Model cache: %s (offline=%s revision=%s)",
        diag["cache_dir"] or "HF default", diag["offline"], diag["revision"],
    )

    if args.check_cache:
        return _check_cache(server_config)

    # Build runtime + app
    from .app import create_app

    runtime = TranslationRuntime(server_config)
    app = create_app(runtime)

    # Start uvicorn
    import uvicorn
    sc = server_config.server
    uvicorn.run(
        app,
        host=sc.host,
        port=sc.port,
        workers=sc.workers,
        log_level=sc.log_level,
        access_log=(sc.log_level == "debug"),
    )
    return 0


def _check_cache(server_config) -> int:
    """Run the --check-cache preflight and exit.

    Reuses the shared translator's authoritative cache resolution; no
    tokenizer/model is loaded and no GPU is required.
    """
    from image_translation.translation import create_translator
    from image_translation.translation.exceptions import TranslationModelLoadError

    translator = create_translator(server_config.translation)
    try:
        resolved = translator.check_cache()
    except TranslationModelLoadError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    print(f"[OK] model: {resolved.model_name} revision: {resolved.revision}")
    print(f"[OK] cache: {resolved.cache_dir or 'HF default'}")
    print(f"[OK] snapshot: {resolved.snapshot_path}")
    print(f"[OK] status: {resolved.cache_status} | offline: {resolved.offline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

    # Build runtime + app
    from .runtime import TranslationRuntime
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


if __name__ == "__main__":
    raise SystemExit(main())

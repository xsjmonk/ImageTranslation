"""CLI entry point – wires input, config, and pipeline together."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from .config import ConfigLoadError, load_config
from .input import InputError, parse_arguments, resolve_input
from .pipeline import run_pipeline

logger = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    # 1. Parse CLI arguments
    try:
        parsed = parse_arguments(argv)
    except SystemExit:
        return 1

    # 2. Load configuration (before input validation, so config errors surface first)
    try:
        config = load_config(parsed.config_path)
    except ConfigLoadError as e:
        _log_error(f"Config error: {e}")
        return 1

    _setup_logging(config.logging.level)

    # 3. Resolve and validate input
    try:
        app_input = resolve_input(parsed)
    except InputError as e:
        _log_error(f"Input error: {e}")
        return 1

    # 4. Log summary
    logger.info("[INFO] Input: %s", app_input.input_path)
    logger.info("[INFO] Output: %s", app_input.output_folder)
    if parsed.config_path:
        logger.info("[INFO] Config: %s", parsed.config_path)

    # 5. Run pipeline
    result = run_pipeline(app_input, config)

    # 6. Print summary
    print()
    print(result.summary())

    if result.has_failures:
        return 1
    return 0


def _setup_logging(level: str) -> None:
    """Configure the root logger with a simple format."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(message)s",
        stream=sys.stderr,
    )


def _log_error(message: str) -> None:
    """Log an error message to stderr."""
    print(f"[ERROR] {message}", file=sys.stderr)

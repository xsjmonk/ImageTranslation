"""CLI argument parsing using argparse. Returns a clean namespace, never raw argparse."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def parse_arguments(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments and return a simple namespace.

    Args:
        argv: Optional argument list (for testing). Uses sys.argv[1:] if None.
    """
    parser = argparse.ArgumentParser(
        prog="image_translation",
        description="Translate Chinese text in images for Amazon listings.",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an image file or a folder of images.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        dest="config_path",
        help="Path to config.json. Falls back to <repo-root>/config.json or defaults.",
    )

    parsed = parser.parse_args(argv)
    # Normalize to Path
    parsed.input_path = Path(parsed.input_path)
    if parsed.config_path:
        parsed.config_path = Path(parsed.config_path)
    return parsed

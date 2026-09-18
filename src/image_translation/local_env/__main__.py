"""Read-only environment validation entry point."""

from __future__ import annotations

import sys

from .env_check import format_environment_errors, run_environment_check


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if args not in (["--check-env"], ["check-env"]):
        print(
            "Usage: python -m image_translation.local_env --check-env",
            file=sys.stderr,
        )
        raise SystemExit(2)

    issues = run_environment_check()
    if issues:
        print(format_environment_errors(issues), file=sys.stderr)
        raise SystemExit(1)

    print("Local image-processing environment OK.")


if __name__ == "__main__":
    main()

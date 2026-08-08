"""Allow `python -m image_translation` to work."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

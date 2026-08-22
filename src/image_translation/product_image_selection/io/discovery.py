from pathlib import Path
def discover_images(roots, recursive=True, max_decoded_pixels=None):
    for root in sorted((Path(x) for x in roots), key=lambda x: str(x).lower()):
        if not root.is_dir():
            raise ValueError(f"configured image root does not exist: {root}")
        paths = root.rglob("*") if recursive else root.glob("*")
        for path in sorted(paths, key=lambda x: str(x).lower()):
            if path.is_file():
                yield path.resolve()

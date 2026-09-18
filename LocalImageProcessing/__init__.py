"""Repository-owned deterministic local image operations."""

from .contract import (
    CONTRACT_VERSION,
    BatchManifest,
    ImageManifest,
    RegionManifest,
    load_manifest,
)
from .paths import ensure_import_paths, module_root, resolve_repo_root
from .processor import ProcessingOptions, process_batch

__all__ = [
    "CONTRACT_VERSION",
    "BatchManifest",
    "ImageManifest",
    "RegionManifest",
    "ProcessingOptions",
    "ensure_import_paths",
    "load_manifest",
    "module_root",
    "process_batch",
    "resolve_repo_root",
]

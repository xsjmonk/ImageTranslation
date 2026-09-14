"""Shared Hugging Face model snapshot resolution for translation backends."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from .exceptions import TranslationModelLoadError
from .models import ResolvedModel

logger = logging.getLogger(__name__)

VerifySnapshot = Callable[[str], None]


def resolve_model_snapshot(
    *,
    model_name: str,
    model_revision: str,
    model_cache_dir: Optional[str],
    offline: bool,
    model_family: str,
    verify_snapshot: VerifySnapshot,
) -> ResolvedModel:
    """Resolve a model snapshot using the configured cache policy.

    Args:
        model_name: Hugging Face model id.
        model_revision: Revision/tag/commit.
        model_cache_dir: Server-owned cache root (``None`` uses HF default).
        offline: When true, network access is forbidden on cache miss.
        model_family: Diagnostic label stored on ``ResolvedModel``.
        verify_snapshot: Backend-specific completeness check.

    Raises:
        TranslationModelLoadError: Cache miss, download failure, or incomplete
            snapshot.
    """
    from huggingface_hub import snapshot_download

    cache_root: Optional[str] = None
    if model_cache_dir:
        cache_root = os.path.expandvars(model_cache_dir)
        root = Path(cache_root).expanduser().resolve()
        cache_root = str(root)
        if offline and not root.is_dir():
            raise TranslationModelLoadError(
                f"offline mode: configured model cache does not exist: "
                f"{cache_root}; pre-download the model (see README) or "
                f"fix model_cache_dir"
            )
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TranslationModelLoadError(
                f"cannot create configured model cache {cache_root}: {exc}"
            ) from exc

    logger.info(
        "[INFO] Model cache: %s (model=%s revision=%s offline=%s)",
        cache_root or "HF default",
        model_name,
        model_revision,
        offline,
    )

    def _snapshot(local_only: bool) -> str:
        kwargs = {}
        if cache_root:
            kwargs["cache_dir"] = cache_root
        return snapshot_download(
            repo_id=model_name,
            revision=model_revision,
            local_files_only=local_only,
            **kwargs,
        )

    try:
        snapshot_path = _snapshot(local_only=True)
        cache_status = "cache_hit"
        logger.info("[INFO] Model cache HIT (reused): %s", snapshot_path)
    except Exception as exc:
        if offline:
            raise TranslationModelLoadError(
                f"offline model cache miss: {model_name} revision "
                f"{model_revision} not found in cache "
                f"{cache_root or 'HF default'}; pre-download the model "
                f"or set allow_model_download=true"
            ) from exc
        logger.info(
            "[INFO] Model cache MISS; downloading %s revision %s into %s ...",
            model_name,
            model_revision,
            cache_root or "HF default",
        )
        try:
            snapshot_path = _snapshot(local_only=False)
        except Exception as exc2:
            raise TranslationModelLoadError(
                f"model download failed for {model_name} revision "
                f"{model_revision} into {cache_root or 'HF default'}: {exc2}"
            ) from exc2
        cache_status = "download"
        logger.info("[INFO] Model download COMPLETE: %s", snapshot_path)

    verify_snapshot(snapshot_path)
    return ResolvedModel(
        snapshot_path=snapshot_path,
        model_name=model_name,
        revision=model_revision,
        cache_dir=cache_root or "",
        cache_status=cache_status,
        offline=offline,
        model_family=model_family,
    )


def verify_seq2seq_snapshot(snapshot_path: str) -> None:
    """Fail before ready if a seq2seq snapshot misses required files."""
    root = Path(snapshot_path)
    required = [
        ("config.json", ["config.json"]),
        ("model weights", ["model.safetensors", "pytorch_model.bin"]),
        ("tokenizer files", ["tokenizer.json", "sentencepiece.bpe.model"]),
    ]
    missing = [
        label
        for label, candidates in required
        if not any((root / candidate).is_file() for candidate in candidates)
    ]
    if missing:
        raise TranslationModelLoadError(
            f"incomplete model snapshot at {snapshot_path}: missing "
            f"{', '.join(missing)}; re-download the model or repair the cache"
        )


def verify_causal_lm_snapshot(snapshot_path: str) -> None:
    """Fail before ready if a causal-LM snapshot misses required files."""
    root = Path(snapshot_path)
    required = [
        ("config.json", ["config.json"]),
        ("model weights", ["model.safetensors", "pytorch_model.bin"]),
        (
            "tokenizer files",
            ["tokenizer.json", "tokenizer_config.json", "vocab.json"],
        ),
    ]
    missing = [
        label
        for label, candidates in required
        if not any((root / candidate).is_file() for candidate in candidates)
    ]
    if missing:
        raise TranslationModelLoadError(
            f"incomplete model snapshot at {snapshot_path}: missing "
            f"{', '.join(missing)}; re-download the model or repair the cache"
        )

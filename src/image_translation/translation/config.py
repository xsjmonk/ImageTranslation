"""Translation configuration — reusable, independent of FastAPI/ImageTranslation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationConfig:
    """M2M100 generation parameter defaults."""
    max_new_tokens: int = 256
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = True


@dataclass
class TranslationConfig:
    """Reusable translation configuration.

    Used by both ImageTranslation pipeline and the standalone translation server.
    """
    model_name: str = "facebook/m2m100_418M"
    source_language: str = "zh"
    target_language: str = "en"
    device: str = "cuda"
    cuda_device: int = 0
    allow_cpu_fallback: bool = False
    precision: str = "auto"  # auto | float16 | float32
    batch_size: int = 8
    max_input_characters: int = 4000
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    model_cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.precision not in ("auto", "float16", "float32"):
            raise ValueError(
                f"precision must be 'auto', 'float16', or 'float32', got '{self.precision}'"
            )
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got '{self.device}'")
        if self.cuda_device < 0:
            raise ValueError("cuda_device must be >= 0")
        if self.max_input_characters < 1:
            raise ValueError("max_input_characters must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def effective_device(self) -> str:
        """Return the device string to use, e.g. 'cuda:0'."""
        if self.device == "cuda":
            return f"cuda:{self.cuda_device}"
        return "cpu"

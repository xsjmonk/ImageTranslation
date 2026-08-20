"""Model-family adapters for local sequence-to-sequence translation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .config import TranslationConfig
from .exceptions import TranslationConfigurationError


class ModelFamilyAdapter(ABC):
    """Owns model-family-specific tokenizer and generation semantics."""

    family: str

    @abstractmethod
    def validate_configuration(self, config: TranslationConfig) -> None:
        ...

    @abstractmethod
    def load_tokenizer(self, snapshot_path: str, local_files_only: bool) -> Any:
        ...

    @abstractmethod
    def load_model(self, snapshot_path: str, local_files_only: bool) -> Any:
        ...

    @abstractmethod
    def configure_source_language(self, tokenizer: Any, source_language: str) -> str:
        ...

    @abstractmethod
    def resolve_target_language_id(
        self, tokenizer: Any, target_language: str
    ) -> int | None:
        ...

    @abstractmethod
    def build_generation_kwargs(
        self,
        tokenizer: Any,
        source_language: str,
        target_language: str,
        *,
        max_new_tokens: int,
        num_beams: int,
        do_sample: bool,
        no_repeat_ngram_size: int | None,
    ) -> dict:
        ...


class NllbAdapter(ModelFamilyAdapter):
    family = "nllb"

    _LANGUAGE_ALIASES = {
        "zh": "zho_Hans",
        "zh-cn": "zho_Hans",
        "zh_cn": "zho_Hans",
        "en": "eng_Latn",
        "en-us": "eng_Latn",
        "en_us": "eng_Latn",
    }

    def validate_configuration(self, config: TranslationConfig) -> None:
        if config.commercial_use:
            raise TranslationConfigurationError(
                "facebook/nllb-200-distilled-600M is CC-BY-NC-4.0 and cannot "
                "be used in commercial_use mode; configure "
                "Helsinki-NLP/opus-mt-zh-en with model_family='helsinki' "
                "explicitly instead"
            )
        if "nllb" not in config.model_name.lower():
            raise TranslationConfigurationError(
                f"model_family='nllb' requires an NLLB model, got "
                f"{config.model_name!r}"
            )

    def load_tokenizer(self, snapshot_path: str, local_files_only: bool):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            snapshot_path, local_files_only=local_files_only
        )

    def load_model(self, snapshot_path: str, local_files_only: bool):
        from transformers import AutoModelForSeq2SeqLM

        return AutoModelForSeq2SeqLM.from_pretrained(
            snapshot_path, local_files_only=local_files_only
        )

    def configure_source_language(self, tokenizer, source_language: str) -> str:
        source = self._normalize(source_language)
        tokenizer.src_lang = source
        return source

    def resolve_target_language_id(self, tokenizer, target_language: str) -> int:
        target = self._normalize(target_language)
        token_id = tokenizer.convert_tokens_to_ids(target)
        if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
            raise TranslationConfigurationError(
                f"NLLB target language token is unavailable: {target}"
            )
        return int(token_id)

    def build_generation_kwargs(
        self,
        tokenizer,
        source_language,
        target_language,
        *,
        max_new_tokens,
        num_beams,
        do_sample,
        no_repeat_ngram_size,
    ) -> dict:
        return {
            "forced_bos_token_id": self.resolve_target_language_id(
                tokenizer, target_language
            ),
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            **(
                {"no_repeat_ngram_size": no_repeat_ngram_size}
                if no_repeat_ngram_size is not None
                else {}
            ),
        }

    @classmethod
    def _normalize(cls, language: str) -> str:
        return cls._LANGUAGE_ALIASES.get(language.lower(), language)


class HelsinkiAdapter(ModelFamilyAdapter):
    """Explicit commercial alternative using the same adapter contract."""

    family = "helsinki"

    def validate_configuration(self, config: TranslationConfig) -> None:
        if "opus-mt-zh-en" not in config.model_name.lower():
            raise TranslationConfigurationError(
                f"model_family='helsinki' requires Helsinki opus-mt-zh-en, "
                f"got {config.model_name!r}"
            )

    def load_tokenizer(self, snapshot_path: str, local_files_only: bool):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            snapshot_path, local_files_only=local_files_only
        )

    def load_model(self, snapshot_path: str, local_files_only: bool):
        from transformers import AutoModelForSeq2SeqLM

        return AutoModelForSeq2SeqLM.from_pretrained(
            snapshot_path, local_files_only=local_files_only
        )

    def configure_source_language(self, tokenizer, source_language: str) -> str:
        return source_language

    def resolve_target_language_id(self, tokenizer, target_language: str) -> None:
        return None

    def build_generation_kwargs(
        self,
        tokenizer,
        source_language,
        target_language,
        *,
        max_new_tokens,
        num_beams,
        do_sample,
        no_repeat_ngram_size,
    ) -> dict:
        return {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            **(
                {"no_repeat_ngram_size": no_repeat_ngram_size}
                if no_repeat_ngram_size is not None
                else {}
            ),
        }


def create_model_family_adapter(config: TranslationConfig) -> ModelFamilyAdapter:
    adapters = {
        "nllb": NllbAdapter(),
        "helsinki": HelsinkiAdapter(),
    }
    try:
        adapter = adapters[config.model_family]
    except KeyError as exc:
        raise TranslationConfigurationError(
            f"unsupported model_family {config.model_family!r}; choose "
            "'nllb' or 'helsinki'"
        ) from exc
    adapter.validate_configuration(config)
    return adapter

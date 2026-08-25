from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .exceptions import ConfigurationError


class SelectorModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NormalizationConfig(SelectorModel):
    mean: tuple[float, float, float] = Field(...)
    std: tuple[float, float, float] = Field(...)
    padding_rgb: tuple[int, int, int] = Field(...)

    @field_validator("std")
    @classmethod
    def positive_std(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(not math.isfinite(item) or item <= 0 for item in value):
            raise ValueError("normalization std must be finite and positive")
        return value

    @field_validator("padding_rgb")
    @classmethod
    def valid_padding(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(item < 0 or item > 255 for item in value):
            raise ValueError("padding_rgb values must be in [0,255]")
        return value


# Backward-compatible name for callers of the original scaffold.
Normalization = NormalizationConfig


class TrainingData(SelectorModel):
    image_root: Path = Field(...)
    labels_csv: Path = Field(...)
    image_path_column: str = Field(..., min_length=1)
    label_column: str = Field(..., min_length=1)
    product_group_column: str = Field(..., min_length=1)
    image_id_column: str = Field(..., min_length=1)
    allow_absolute_image_paths: bool = Field(...)
    max_decoded_pixels: int = Field(..., gt=0)


class InferenceData(SelectorModel):
    image_roots: tuple[Path, ...] = Field(...)
    recursive: bool = Field(...)
    max_decoded_pixels: int = Field(..., gt=0)
    unreadable_image_policy: Literal["report", "fail"] = Field(...)
    inference_batch_size: int = Field(..., gt=0)


class DataConfig(SelectorModel):
    training: TrainingData = Field(...)
    inference: InferenceData = Field(...)


class ModelConfig(SelectorModel):
    architecture: Literal["efficientnet_v2_s"] = Field(...)
    pretrained_weights: Literal["DEFAULT"] | None = Field(...)
    input_size: int = Field(..., gt=0)
    normalization: NormalizationConfig = Field(...)
    checkpoint_path: Path = Field(...)


class SplitConfig(SelectorModel):
    train_fraction: float = Field(..., gt=0, le=1)
    validation_fraction: float = Field(..., gt=0, le=1)
    test_fraction: float = Field(..., gt=0, le=1)
    minimum_groups_per_class: int = Field(..., gt=0)

    @model_validator(mode="after")
    def fractions(self) -> "SplitConfig":
        values = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("split fractions must be finite")
        if abs(sum(values) - 1) > 1e-6:
            raise ValueError("split fractions must sum to 1")
        return self


class TrainingConfig(SelectorModel):
    device: Literal["cpu", "cuda"] = Field(...)
    allow_cpu_fallback: bool = Field(...)
    seed: int = Field(...)
    deterministic_algorithms: bool = Field(...)
    num_workers: int = Field(..., ge=0)
    batch_size: int = Field(..., gt=0)
    head_warmup_epochs: int = Field(..., ge=0)
    fine_tune_epochs: int = Field(..., ge=0)
    learning_rate_head: float = Field(..., gt=0)
    learning_rate_backbone: float = Field(..., gt=0)
    weight_decay: float = Field(..., ge=0)
    early_stopping_patience: int = Field(..., gt=0)
    mixed_precision: bool = Field(...)
    augmentation: "AugmentationConfig" = Field(...)
    split: SplitConfig = Field(...)

    @field_validator(
        "learning_rate_head",
        "learning_rate_backbone",
        "weight_decay",
    )
    @classmethod
    def finite_training_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("training numeric values must be finite")
        return value


class AugmentationConfig(SelectorModel):
    enabled: bool = Field(...)
    color_jitter: float = Field(..., ge=0, le=1)
    scale_min: float = Field(..., gt=0, le=1)
    translation_fraction: float = Field(..., ge=0, le=1)
    horizontal_flip_probability: float = Field(..., ge=0, le=1)


class DecisionConfig(SelectorModel):
    threshold_candidates: tuple[float, ...] = Field(...)
    selection_objective: Literal[
        "maximize_taken_recall_subject_to_min_taken_precision"
    ] = Field(...)
    min_taken_precision: float = Field(..., ge=0, le=1)
    review_band_half_width: float = Field(..., ge=0, le=1)
    calibrate_temperature: bool = Field(...)

    @model_validator(mode="after")
    def ranges(self) -> "DecisionConfig":
        if not self.threshold_candidates:
            raise ValueError("at least one threshold is required")
        if any(
            not math.isfinite(value) or not 0 <= value <= 1
            for value in self.threshold_candidates
        ):
            raise ValueError("thresholds must be finite and in [0,1]")
        if len(set(self.threshold_candidates)) != len(self.threshold_candidates):
            raise ValueError("thresholds must be unique")
        if tuple(sorted(self.threshold_candidates)) != self.threshold_candidates:
            raise ValueError("thresholds must be sorted")
        return self


class QualityGate(SelectorModel):
    min_test_taken_precision: float = Field(..., ge=0, le=1)
    min_test_taken_recall: float = Field(..., ge=0, le=1)
    max_test_not_taken_false_positive_rate: float = Field(..., ge=0, le=1)
    minimum_test_images_per_class: int = Field(..., gt=0)
    fail_training_if_not_met: bool = Field(...)


class Artifacts(SelectorModel):
    output_directory: Path = Field(...)
    run_name: str = Field(..., min_length=1)
    overwrite_existing_run: bool = Field(...)


class OutputConfig(SelectorModel):
    kind: Literal["csv"] = Field(...)
    csv_path: Path = Field(...)


class SelectorConfig(SelectorModel):
    data: DataConfig = Field(...)
    model: ModelConfig = Field(...)
    training: TrainingConfig = Field(...)
    decision: DecisionConfig = Field(...)
    quality_gate: QualityGate = Field(...)
    artifacts: Artifacts = Field(...)
    output: OutputConfig = Field(...)

    @model_validator(mode="after")
    def validate_paths(self) -> "SelectorConfig":
        if not self.data.inference.image_roots:
            raise ValueError("at least one inference image root is required")
        input_roots = (
            self.data.training.image_root,
            *self.data.inference.image_roots,
        )
        for field_name, output_path in (
            ("artifacts.output_directory", self.artifacts.output_directory),
            ("output.csv_path", self.output.csv_path),
        ):
            if any(_is_within(output_path, root) for root in input_roots):
                raise ValueError(f"{field_name} must not be inside an input image root")
        return self


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _resolve_path(base: Path, value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field_name} must be a non-empty path string")
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((base / candidate).resolve())


def load_config(path: str | Path) -> SelectorConfig:
    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(str(error)) from error

    if not isinstance(raw, dict):
        raise ConfigurationError("selector configuration must be a JSON object")

    resolved = copy.deepcopy(raw)
    try:
        training_data = resolved["data"]["training"]
        inference_data = resolved["data"]["inference"]
        resolved["data"]["training"]["image_root"] = _resolve_path(
            config_path.parent,
            training_data["image_root"],
            "data.training.image_root",
        )
        resolved["data"]["training"]["labels_csv"] = _resolve_path(
            config_path.parent,
            training_data["labels_csv"],
            "data.training.labels_csv",
        )
        image_roots = inference_data["image_roots"]
        if not isinstance(image_roots, list):
            raise ConfigurationError("data.inference.image_roots must be a list")
        resolved["data"]["inference"]["image_roots"] = [
            _resolve_path(
                config_path.parent,
                value,
                "data.inference.image_roots",
            )
            for value in image_roots
        ]
        for section, key in (
            ("model", "checkpoint_path"),
            ("output", "csv_path"),
            ("artifacts", "output_directory"),
        ):
            resolved[section][key] = _resolve_path(
                config_path.parent,
                resolved[section][key],
                f"{section}.{key}",
            )
        return SelectorConfig.model_validate(resolved)
    except ConfigurationError:
        raise
    except (KeyError, TypeError) as error:
        raise ConfigurationError(
            f"configuration structure is incomplete: {error}"
        ) from error
    except Exception as error:
        raise ConfigurationError(str(error)) from error

from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, model_validator, field_validator
from .exceptions import ConfigurationError

class NormalizationConfig(BaseModel):
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    padding_rgb: tuple[int, int, int]
    @field_validator("std")
    @classmethod
    def positive_std(cls, value):
        if any(x <= 0 for x in value): raise ValueError("normalization std must be positive")
        return value

    @field_validator("padding_rgb")
    @classmethod
    def valid_padding(cls, value):
        if any(x < 0 or x > 255 for x in value): raise ValueError("padding_rgb values must be in [0,255]")
        return value

# Backward-compatible name for callers of the original scaffold.
Normalization = NormalizationConfig
class TrainingData(BaseModel):
    image_root: Path
    labels_csv: Path
    image_path_column: str = "image_path"; label_column: str = "label"
    product_group_column: str = "product_group_id"; image_id_column: str = "image_id"
    allow_absolute_image_paths: bool = False; max_decoded_pixels: int = Field(gt=0)
class InferenceData(BaseModel):
    image_roots: tuple[Path, ...]; recursive: bool = True
    max_decoded_pixels: int = Field(gt=0); unreadable_image_policy: str
    inference_batch_size: int = Field(gt=0)
    @field_validator("unreadable_image_policy")
    @classmethod
    def valid_policy(cls, value):
        if value not in {"report", "fail"}: raise ValueError("unreadable_image_policy must be report or fail")
        return value
class DataConfig(BaseModel):
    training: TrainingData; inference: InferenceData
class ModelConfig(BaseModel):
    architecture: Literal["efficientnet_v2_s"]
    pretrained_weights: Literal["DEFAULT"] | None
    input_size: int = Field(gt=0); normalization: NormalizationConfig; checkpoint_path: Path
class SplitConfig(BaseModel):
    train_fraction: float = Field(gt=0); validation_fraction: float = Field(gt=0); test_fraction: float = Field(gt=0)
    minimum_groups_per_class: int = Field(gt=0)
    @model_validator(mode="after")
    def fractions(self):
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1) > 1e-6:
            raise ValueError("split fractions must sum to 1")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("split fractions must be positive")
        return self
class TrainingConfig(BaseModel):
    device: Literal["cpu","cuda"] = "cpu"; allow_cpu_fallback: bool = True; seed: int = 0
    deterministic_algorithms: bool = True; num_workers: int = Field(ge=0); batch_size: int = Field(gt=0)
    head_warmup_epochs: int = Field(ge=0); fine_tune_epochs: int = Field(ge=0)
    learning_rate_head: float = Field(gt=0); learning_rate_backbone: float = Field(gt=0)
    weight_decay: float = Field(ge=0); early_stopping_patience: int = Field(gt=0)
    mixed_precision: bool; augmentation: "AugmentationConfig"; split: SplitConfig

class AugmentationConfig(BaseModel):
    enabled: bool
    color_jitter: float = Field(ge=0, le=1)
    scale_min: float = Field(gt=0, le=1)
    translation_fraction: float = Field(ge=0, le=1)
    horizontal_flip_probability: float = Field(ge=0, le=1)
class DecisionConfig(BaseModel):
    threshold_candidates: tuple[float, ...]; selection_objective: Literal["maximize_taken_recall_subject_to_min_taken_precision"]
    min_taken_precision: float = Field(ge=0, le=1); review_band_half_width: float = Field(ge=0, le=1)
    calibrate_temperature: bool = True
    @model_validator(mode="after")
    def ranges(self):
        if any(not 0 <= x <= 1 for x in self.threshold_candidates): raise ValueError("thresholds must be in [0,1]")
        if not self.threshold_candidates: raise ValueError("at least one threshold is required")
        if len(set(self.threshold_candidates)) != len(self.threshold_candidates):
            raise ValueError("thresholds must be unique")
        if tuple(sorted(self.threshold_candidates)) != self.threshold_candidates:
            raise ValueError("thresholds must be sorted")
        return self
class QualityGate(BaseModel):
    min_test_taken_precision: float = Field(ge=0, le=1); min_test_taken_recall: float = Field(ge=0, le=1)
    max_test_not_taken_false_positive_rate: float = Field(ge=0, le=1)
    minimum_test_images_per_class: int = Field(gt=0); fail_training_if_not_met: bool = True
class Artifacts(BaseModel):
    output_directory: Path; run_name: str; overwrite_existing_run: bool = False
class OutputConfig(BaseModel):
    kind: str = "csv"; csv_path: Path
    @model_validator(mode="after")
    def csv_only(self):
        if self.kind != "csv": raise ValueError("output.kind must be csv")
        return self
class SelectorConfig(BaseModel):
    data: DataConfig; model: ModelConfig; training: TrainingConfig
    decision: DecisionConfig; quality_gate: QualityGate; artifacts: Artifacts; output: OutputConfig

    @model_validator(mode="after")
    def paths_exist_as_values(self):
        if not self.data.inference.image_roots:
            raise ValueError("at least one inference image root is required")
        return self

def load_config(path: str | Path) -> SelectorConfig:
    p = Path(path).resolve()
    try: raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e: raise ConfigurationError(str(e)) from e
    raw.setdefault("data", {}).setdefault("training", {})
    if not isinstance(raw.get("data", {}).get("inference", {}).get("image_roots"), list):
        raise ConfigurationError("data.inference.image_roots must be a list")
    raw["data"]["training"]["image_root"] = str((p.parent / raw["data"]["training"]["image_root"]).resolve()) if not Path(raw["data"]["training"]["image_root"]).is_absolute() else raw["data"]["training"]["image_root"]
    raw["data"]["training"]["labels_csv"] = str((p.parent / raw["data"]["training"]["labels_csv"]).resolve()) if not Path(raw["data"]["training"]["labels_csv"]).is_absolute() else raw["data"]["training"]["labels_csv"]
    raw["data"]["inference"]["image_roots"] = [str((p.parent / x).resolve()) if not Path(x).is_absolute() else x for x in raw["data"]["inference"]["image_roots"]]
    for section, key in (("model","checkpoint_path"),("output","csv_path"),("artifacts","output_directory")):
        value=raw[section][key]
        raw[section][key]=str((p.parent / value).resolve()) if not Path(value).is_absolute() else value
    try: return SelectorConfig.model_validate(raw)
    except Exception as e: raise ConfigurationError(str(e)) from e

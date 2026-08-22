import torch
from datetime import datetime, timezone
from pathlib import Path
import os
import tempfile
import math
from .exceptions import ArtifactError
def build_model(architecture="efficientnet_v2_s", pretrained_weights="DEFAULT"):
    if architecture != "efficientnet_v2_s" or pretrained_weights not in ("DEFAULT", None):
        raise ValueError("unsupported selector architecture or pretrained weights")
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained_weights == "DEFAULT" else None
    model = efficientnet_v2_s(weights=weights); model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    return model
def save_checkpoint(path, model, metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload={**metadata, "state_dict": model.state_dict(),
             "pretrained_weights": metadata.get("pretrained_weights"),
             "config_sha256": metadata.get("config_sha256",""),
             "model_version": metadata.get("model_version","unknown"),
             "torch_version": metadata.get("torch_version",torch.__version__),
             "torchvision_version": metadata.get("torchvision_version","unknown"),
             "cuda_version": metadata.get("cuda_version",torch.version.cuda),
             "quality_gate": metadata.get("quality_gate",{"passed":True}),
             "created_utc": metadata.get("created_utc",datetime.now(timezone.utc).isoformat())}
    target = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
def load_checkpoint(path, model, expected=None):
    try: item=torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e: raise ArtifactError(f"unable to read checkpoint: {e}") from e
    required=("artifact_schema_version","state_dict","architecture","pretrained_weights","labels","input_size","normalization","temperature","taken_threshold","review_band_half_width","config_sha256","model_version","torch_version","torchvision_version","cuda_version","quality_gate","created_utc")
    if not isinstance(item, dict) or any(k not in item for k in required):
        raise ArtifactError("checkpoint missing required schema fields")
    if item["artifact_schema_version"] != 1 or item["labels"] != ["not_taken","taken"]:
        raise ArtifactError("unsupported checkpoint schema or class order")
    if not isinstance(item["temperature"], (int,float)) or not math.isfinite(item["temperature"]) or item["temperature"] <= 0:
        raise ArtifactError("invalid checkpoint temperature")
    if not isinstance(item["taken_threshold"], (int,float)) or not math.isfinite(item["taken_threshold"]) or not 0 <= item["taken_threshold"] <= 1:
        raise ArtifactError("invalid checkpoint threshold")
    if not isinstance(item["review_band_half_width"], (int,float)) or not math.isfinite(item["review_band_half_width"]) or not 0 <= item["review_band_half_width"] <= 1:
        raise ArtifactError("invalid checkpoint review width")
    if not isinstance(item["architecture"], str) or not item["architecture"] or not isinstance(item["input_size"], int) or item["input_size"] <= 0:
        raise ArtifactError("invalid checkpoint model metadata")
    if expected:
        for key, value in expected.items():
            if item.get(key) != value: raise ArtifactError(f"incompatible checkpoint: {key}")
    if not isinstance(item["normalization"], dict) or not all(k in item["normalization"] for k in ("mean","std","padding_rgb")):
        raise ArtifactError("checkpoint preprocessing is incomplete")
    if len(item["normalization"]["mean"]) != 3 or len(item["normalization"]["std"]) != 3 or len(item["normalization"]["padding_rgb"]) != 3 or any(float(x) <= 0 for x in item["normalization"]["std"]):
        raise ArtifactError("checkpoint preprocessing vectors must have length three")
    if not isinstance(item["quality_gate"], dict) or "passed" not in item["quality_gate"]:
        raise ArtifactError("checkpoint quality gate summary is incomplete")
    checks = item["quality_gate"].get("checks")
    if not isinstance(checks, dict) or not checks or any(
        not isinstance(v, dict) or not all(k in v for k in ("actual", "required", "passed"))
        for v in checks.values()
    ):
        raise ArtifactError("checkpoint quality gate checks are incomplete")
    try: model.load_state_dict(item["state_dict"])
    except Exception as e: raise ArtifactError(f"incompatible model state_dict: {e}") from e
    return item

def checkpoint_metadata(config, config_hash, model_version):
    return {
        "artifact_schema_version": 1,
        "architecture": config.model.architecture,
        "pretrained_weights": config.model.pretrained_weights,
        "labels": ["not_taken", "taken"],
        "input_size": config.model.input_size,
        "normalization": config.model.normalization.model_dump(),
        "config_sha256": config_hash,
        "model_version": model_version,
        "temperature": 1.0,
        "review_band_half_width": config.decision.review_band_half_width,
        "taken_threshold": 0.5,
        "quality_gate": {"passed": False, "checks": {}},
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "cuda_version": torch.version.cuda,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

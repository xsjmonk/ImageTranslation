import torch
from .domain import CategorizationResult, ImageFailure, ImagePrediction, ProductImageLabel
from .preprocessing import preprocess_pixels
from .training import logits_to_taken_probability
class ProductImageClassifier:
    def __init__(self, model, config, model_version="unknown", device=None, artifact=None):
        self.model=model.eval(); self.config=config; self.model_version=model_version
        requested = device or config.training.device
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            if not config.training.allow_cpu_fallback: raise RuntimeError("CUDA requested but unavailable")
            requested = "cpu"
        self.device=torch.device(requested); self.model.to(self.device)
        artifact = artifact or {}
        required = ("taken_threshold", "temperature", "review_band_half_width",
                    "labels", "architecture", "input_size", "normalization")
        if any(key not in artifact for key in required):
            raise ValueError("incomplete selector artifact")
        if artifact["labels"] != ["not_taken", "taken"]:
            raise ValueError("invalid artifact class order")
        if artifact["architecture"] != config.model.architecture:
            raise ValueError("artifact architecture does not match active config")
        normalization = (config.model.normalization.model_dump()
                         if hasattr(config.model.normalization, "model_dump")
                         else {"mean": list(config.model.normalization.mean),
                               "std": list(config.model.normalization.std),
                               "padding_rgb": list(config.model.normalization.padding_rgb)})
        normalized_artifact = {key: list(value) for key, value in artifact["normalization"].items()}
        normalized_config = {key: list(value) for key, value in normalization.items()}
        if artifact["input_size"] != config.model.input_size or normalized_artifact != normalized_config:
            raise ValueError("artifact preprocessing does not match active config")
        self.threshold = float(artifact["taken_threshold"])
        self.temperature = float(artifact["temperature"])
        if self.temperature <= 0 or not 0 <= self.threshold <= 1:
            raise ValueError("invalid artifact decision policy")
        self.review_width = float(artifact["review_band_half_width"])
        if self.review_width < 0 or self.review_width > 1:
            raise ValueError("invalid artifact review width")
    def predict(self, image_id, pixels, image_path=None):
        c=self.config.model; x=preprocess_pixels(pixels,c.input_size,c.normalization.mean,c.normalization.std,c.normalization.padding_rgb).unsqueeze(0).to(self.device)
        with torch.inference_mode(): taken=float(logits_to_taken_probability(self.model(x), self.temperature)[0])
        not_taken=1.0-taken; threshold=self.threshold
        label=ProductImageLabel.TAKEN if taken >= threshold else ProductImageLabel.NOT_TAKEN
        return ImagePrediction(image_id,taken,not_taken,label,abs(taken-threshold)<=self.review_width,self.model_version,image_path or image_id)
    def categorize(self, items):
        predictions=[]; failures=[]; data=getattr(self.config,"data",None); inference=getattr(data,"inference",None)
        batch_size=getattr(inference,"inference_batch_size",getattr(self.config.training,"batch_size",16))
        items=list(items)
        for start in range(0,len(items),batch_size):
            batch=items[start:start+batch_size]; tensors=[]; valid=[]
            for item in batch:
                image_id,pixels=item[:2]; image_path=item[2] if len(item)>2 else image_id
                try:
                    c=self.config.model; tensors.append(preprocess_pixels(pixels,c.input_size,c.normalization.mean,c.normalization.std,c.normalization.padding_rgb)); valid.append((image_id,image_path))
                except Exception as e: failures.append(ImageFailure(image_id,image_path,str(e)))
            if not tensors: continue
            try:
                with torch.inference_mode(): taken_probs=logits_to_taken_probability(self.model(torch.stack(tensors).to(self.device)), self.temperature).cpu()
                for (image_id,image_path),taken in zip(valid,taken_probs):
                    taken=float(taken); label=ProductImageLabel.TAKEN if taken>=self.threshold else ProductImageLabel.NOT_TAKEN
                    predictions.append(ImagePrediction(image_id,taken,1.0-taken,label,abs(taken-self.threshold)<=self.review_width,self.model_version,image_path))
            except Exception as e:
                for image_id,image_path in valid: failures.append(ImageFailure(image_id,image_path,str(e)))
        return CategorizationResult(tuple(predictions),tuple(failures))

import torch
from .domain import CategorizationResult, ImageFailure, ImagePrediction, ProductImageLabel
from .preprocessing import preprocess_pixels
class ProductImageClassifier:
    def __init__(self, model, config, model_version="unknown", device=None, artifact=None):
        self.model=model.eval(); self.config=config; self.model_version=model_version
        requested = device or config.training.device
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            if not config.training.allow_cpu_fallback: raise RuntimeError("CUDA requested but unavailable")
            requested = "cpu"
        self.device=torch.device(requested); self.model.to(self.device)
        artifact = artifact or {}
        self.threshold = float(artifact["taken_threshold"])
        self.temperature = float(artifact["temperature"])
        if self.temperature <= 0 or not 0 <= self.threshold <= 1:
            raise ValueError("invalid artifact decision policy")
        self.review_width = float(artifact.get("review_band_half_width", config.decision.review_band_half_width))
    def predict(self, image_id, pixels, image_path=None):
        c=self.config.model; x=preprocess_pixels(pixels,c.input_size,c.normalization.mean,c.normalization.std,c.normalization.padding_rgb).unsqueeze(0).to(self.device)
        with torch.inference_mode(): p=torch.softmax(self.model(x)[0].float() / self.temperature,dim=0)
        taken=float(p[1]); threshold=self.threshold
        label=ProductImageLabel.TAKEN if taken >= threshold else ProductImageLabel.NOT_TAKEN
        return ImagePrediction(image_id,taken,float(p[0]),label,abs(taken-threshold)<=self.review_width,self.model_version,image_path or image_id)
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
                with torch.inference_mode(): probs=torch.softmax(self.model(torch.stack(tensors).to(self.device))/self.temperature,dim=1).cpu()
                for (image_id,image_path),p in zip(valid,probs):
                    taken=float(p[1]); label=ProductImageLabel.TAKEN if taken>=self.threshold else ProductImageLabel.NOT_TAKEN
                    predictions.append(ImagePrediction(image_id,taken,float(p[0]),label,abs(taken-self.threshold)<=self.review_width,self.model_version,image_path))
            except Exception as e:
                for image_id,image_path in valid: failures.append(ImageFailure(image_id,image_path,str(e)))
        return CategorizationResult(tuple(predictions),tuple(failures))

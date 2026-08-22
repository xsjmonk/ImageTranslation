from __future__ import annotations
import copy, random
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from .preprocessing import preprocess_pixels
from .split import grouped_split, validate_pixel_leakage, pixel_hash
from .io.raster_pillow import PillowRasterReader
from .exceptions import ManifestError

def seed_everything(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic: torch.use_deterministic_algorithms(True, warn_only=True)

def resolve_device(cfg):
    requested = cfg.training.device
    if requested == "cpu": return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available(): return torch.device(requested)
    if cfg.training.allow_cpu_fallback: return torch.device("cpu")
    raise RuntimeError("configured CUDA device is unavailable and CPU fallback is disabled")

class _Images(Dataset):
    def __init__(self, records, cfg, augment=False):
        self.records=records; self.cfg=cfg; self.reader=PillowRasterReader(cfg.data.training.max_decoded_pixels); self.augment=augment
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]; pixels=self.reader.read_rgb_pixels(r.image_path)
        if self.augment and self.cfg.training.augmentation.enabled:
            a=self.cfg.training.augmentation
            image=Image.fromarray(pixels,"RGB")
            if a.scale_min < 1:
                scale=random.uniform(a.scale_min,1.0); w,h=image.size
                image=image.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BILINEAR)
                canvas=Image.new("RGB",(w,h)); canvas.paste(image,((w-image.width)//2,(h-image.height)//2)); image=canvas
            if a.translation_fraction:
                dx=random.uniform(-a.translation_fraction,a.translation_fraction)*image.width
                dy=random.uniform(-a.translation_fraction,a.translation_fraction)*image.height
                image=image.transform(image.size,Image.Transform.AFFINE,(1,0,-dx,0,1,-dy),fillcolor=0)
            if a.color_jitter:
                factor=1+random.uniform(-a.color_jitter,a.color_jitter)
                image=ImageEnhance.Color(ImageEnhance.Brightness(image).enhance(factor)).enhance(factor)
            pixels=np.asarray(image)
            if random.random() < a.horizontal_flip_probability: pixels=pixels[:, ::-1].copy()
        c=self.cfg.model
        return preprocess_pixels(pixels,c.input_size,c.normalization.mean,c.normalization.std,c.normalization.padding_rgb), int(r.label.value=="taken")

def logits_to_taken_probability(logits, temperature):
    if temperature <= 0: raise ValueError("temperature must be positive")
    return torch.softmax(torch.as_tensor(logits).float() / temperature, dim=1)[:, 1]

def metrics_from_taken_probability(taken_probability, labels, threshold=.5):
    taken_probability=torch.as_tensor(taken_probability).float(); labels=torch.as_tensor(labels).long()
    pred=(taken_probability>=threshold).long()
    tp=int(((pred==1)&(labels==1)).sum()); fp=int(((pred==1)&(labels==0)).sum()); fn=int(((pred==0)&(labels==1)).sum()); tn=int(((pred==0)&(labels==0)).sum())
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.; nr=tn/(tn+fp) if tn+fp else 0.
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    return {"confusion_matrix":[[tn,fp],[fn,tp]],"taken_precision":precision,"taken_recall":recall,"taken_f1":f1,"not_taken_precision":tn/(tn+fn) if tn+fn else 0.,"not_taken_recall":nr,"not_taken_f1":2*(tn/(tn+fn) if tn+fn else 0.)*nr/((tn/(tn+fn) if tn+fn else 0.)+nr) if (tn+fn and tn+fp) else 0.,"not_taken_false_positive_rate":fp/(fp+tn) if fp+tn else 0.,"support":{"not_taken":tn+fp,"taken":tp+fn}}

def confusion_metrics(logits, labels, threshold=.5, temperature=1.):
    return metrics_from_taken_probability(logits_to_taken_probability(logits, temperature), labels, threshold)

def calibrate_temperature(logits, labels):
    logits=torch.as_tensor(logits).float(); labels=torch.as_tensor(labels).long()
    log_t=torch.zeros(1,requires_grad=True); opt=torch.optim.LBFGS([log_t],lr=.1,max_iter=50)
    def closure():
        opt.zero_grad(); loss=torch.nn.functional.cross_entropy(logits/torch.exp(log_t),labels); loss.backward(); return loss
    opt.step(closure); return float(torch.exp(log_t).clamp(.05,20).item())

def select_threshold(logits, labels, candidates, min_precision, calibrate=True):
    temperature=calibrate_temperature(logits,labels) if calibrate else 1.
    probabilities=logits_to_taken_probability(logits, temperature)
    options=[]
    for threshold in candidates:
        metric=metrics_from_taken_probability(probabilities,labels,float(threshold))
        if metric["taken_precision"] >= min_precision: options.append((metric["taken_recall"],-float(threshold),float(threshold)))
    if not options: raise ManifestError("no threshold candidate satisfies minimum taken precision")
    return max(options)[2],temperature

@dataclass(frozen=True)
class PreflightResult:
    records: tuple
    decoded_pixels: tuple
    pixel_hashes: dict
    splits: dict
    split_counts: dict

def preflight(records,cfg, reader=None):
    reader=reader or PillowRasterReader(cfg.data.training.max_decoded_pixels); pixels=[]; hashes={}
    for record in records:
        data=reader.read_rgb_pixels(record.image_path); pixels.append(data); digest=pixel_hash(data)
        if digest in hashes and (hashes[digest].label != record.label or hashes[digest].product_group_id != record.product_group_id):
            raise ManifestError("identical decoded pixels have conflicting labels or groups")
        hashes[digest]=record
    fractions=(cfg.training.split.train_fraction,cfg.training.split.validation_fraction,cfg.training.split.test_fraction)
    splits=grouped_split(records,cfg.training.seed,fractions,cfg.training.split.minimum_groups_per_class)
    validate_pixel_leakage(records,pixels,splits)
    if any(not s or {r.label.value for r in s}!={"taken","not_taken"} for s in splits.values()):
        raise ManifestError("every split must be non-empty and contain both classes")
    return PreflightResult(tuple(records),tuple(pixels),{r.image_id:pixel_hash(p) for r,p in zip(records,pixels)},splits,
                           {k:{"total":len(v),"taken":sum(r.label.value=="taken" for r in v),"not_taken":sum(r.label.value=="not_taken" for r in v)} for k,v in splits.items()})

def train(model, records, cfg, preflight_result=None):
    seed_everything(cfg.training.seed,cfg.training.deterministic_algorithms); device=resolve_device(cfg)
    preflight_result=preflight_result or preflight(records,cfg)
    splits=preflight_result.splits
    loader=DataLoader(_Images(splits["train"],cfg,True),batch_size=cfg.training.batch_size,shuffle=True,num_workers=cfg.training.num_workers)
    val=DataLoader(_Images(splits["validation"],cfg),batch_size=cfg.training.batch_size,shuffle=False,num_workers=cfg.training.num_workers)
    counts=np.bincount([int(r.label.value=="taken") for r in splits["train"]],minlength=2)
    weights=torch.tensor([len(splits["train"])/max(1,2*n) for n in counts],dtype=torch.float32,device=device)
    model.to(device); head=getattr(model,"classifier",None)
    if head is not None:
        for p in model.parameters(): p.requires_grad=False
        for p in head.parameters(): p.requires_grad=True
    scaler=torch.amp.GradScaler("cuda",enabled=cfg.training.mixed_precision and device.type=="cuda")
    best=None; best_score=(-1.,-1.); stale=0
    stages=[(cfg.training.head_warmup_epochs,cfg.training.learning_rate_head,False),(cfg.training.fine_tune_epochs,cfg.training.learning_rate_head,True)]
    for epochs,lr,unfreeze in stages:
        stale = 0
        if unfreeze:
            for p in model.parameters(): p.requires_grad=True
        if unfreeze:
            head_params=list(head.parameters()) if head is not None else []
            head_ids={id(p) for p in head_params}
            backbone=[p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
            params=[{"params":backbone,"lr":cfg.training.learning_rate_backbone},{"params":head_params,"lr":cfg.training.learning_rate_head}]
        else:
            params=[{"params":[p for p in model.parameters() if p.requires_grad],"lr":cfg.training.learning_rate_head}]
        opt=torch.optim.AdamW(params,weight_decay=cfg.training.weight_decay)
        for _ in range(epochs):
            model.train()
            for x,y in loader:
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()):
                    loss=torch.nn.functional.cross_entropy(model(x.to(device)),y.to(device),weight=weights)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            model.eval(); logits=[]; labels=[]
            with torch.inference_mode():
                for x,y in val: logits.append(model(x.to(device)).cpu()); labels.append(y)
            metrics=confusion_metrics(torch.cat(logits),torch.cat(labels))
            score=(metrics["taken_precision"],metrics["taken_recall"])
            if score>best_score: best_score=score; best=copy.deepcopy(model.state_dict()); stale=0
            else: stale+=1
            if stale>=cfg.training.early_stopping_patience: break
    if best is not None: model.load_state_dict(best)
    return model,splits

def evaluate(model, records, cfg, threshold=.5, temperature=1.):
    if not records: raise ValueError("cannot evaluate an empty record set")
    device=next(model.parameters()).device; loader=DataLoader(_Images(records,cfg),batch_size=cfg.training.batch_size,shuffle=False)
    logits=[]; labels=[]; model.eval()
    with torch.inference_mode():
        for x,y in loader: logits.append(model(x.to(device)).cpu()); labels.append(y)
    raw_logits=torch.cat(logits); raw_labels=torch.cat(labels)
    return metrics_from_taken_probability(logits_to_taken_probability(raw_logits,temperature),raw_labels,threshold), raw_logits, raw_labels

def quality_gate(metrics,cfg):
    support=metrics["support"]
    checks={"minimum_taken_support":{"actual":support["taken"],"required":cfg.quality_gate.minimum_test_images_per_class,"passed":support["taken"] >= cfg.quality_gate.minimum_test_images_per_class},
            "minimum_not_taken_support":{"actual":support["not_taken"],"required":cfg.quality_gate.minimum_test_images_per_class,"passed":support["not_taken"] >= cfg.quality_gate.minimum_test_images_per_class},
            "taken_precision":{"actual":metrics["taken_precision"],"required":cfg.quality_gate.min_test_taken_precision,"passed":metrics["taken_precision"] >= cfg.quality_gate.min_test_taken_precision},
            "taken_recall":{"actual":metrics["taken_recall"],"required":cfg.quality_gate.min_test_taken_recall,"passed":metrics["taken_recall"] >= cfg.quality_gate.min_test_taken_recall},
            "not_taken_false_positive_rate":{"actual":metrics["not_taken_false_positive_rate"],"required":cfg.quality_gate.max_test_not_taken_false_positive_rate,"passed":metrics["not_taken_false_positive_rate"] <= cfg.quality_gate.max_test_not_taken_false_positive_rate}}
    return {"passed":all(x["passed"] for x in checks.values()),"checks":checks,"support":support}


def promote_checkpoint(staged_path, production_path):
    """Atomically promote only a fully written, already-gated checkpoint."""
    import os
    from pathlib import Path
    target = Path(production_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_path, target)


def promote_if_quality_gate_passed(staged_path, production_path, quality_gate_summary,
                                   diagnostics_path=None, diagnostics=None):
    """Persist run diagnostics and promote only after an explicit passing gate."""
    if diagnostics_path is not None:
        from .io.artifacts import atomic_json
        atomic_json(diagnostics_path, diagnostics or {"quality_gate": quality_gate_summary})
    if not quality_gate_summary.get("passed", False):
        raise RuntimeError("quality gate failed; production checkpoint was not promoted")
    promote_checkpoint(staged_path, production_path)

def roc_auc(logits, labels):
    labels=torch.as_tensor(labels).long(); scores=logits_to_taken_probability(logits,1.)
    positives=scores[labels==1]; negatives=scores[labels==0]
    if not len(positives) or not len(negatives): return None
    return float((positives[:,None] > negatives[None,:]).float().mean()+.5*(positives[:,None] == negatives[None,:]).float().mean())

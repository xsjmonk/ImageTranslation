import argparse
import time
import torch
from pathlib import Path
from .config import load_config
from .io.manifest_csv import CsvManifestReader
from .model import build_model, save_checkpoint, checkpoint_metadata
from .training import train, evaluate, select_threshold, quality_gate, roc_auc, preflight, promote_checkpoint, logits_to_taken_probability
from .io.artifacts import config_sha256, atomic_json, atomic_csv, run_directory
def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True)
    args=parser.parse_args(); cfg=load_config(args.config)
    records=CsvManifestReader(cfg.data.training.labels_csv,cfg.data.training.image_root, {
        "image_path":cfg.data.training.image_path_column, "label":cfg.data.training.label_column,
        "product_group_id":cfg.data.training.product_group_column, "image_id":cfg.data.training.image_id_column,
        "allow_absolute_image_paths":cfg.data.training.allow_absolute_image_paths},
        cfg.data.training.max_decoded_pixels).read()
    if not records: raise ValueError("manifest contains no records")
    preflight_result=preflight(records,cfg)
    model=build_model(cfg.model.architecture,cfg.model.pretrained_weights)
    model,splits=train(model,records,cfg,preflight_result=preflight_result)
    version=cfg.artifacts.run_name; digest=config_sha256(cfg.model_dump(mode="json"))
    run=run_directory(cfg.artifacts.output_directory,version,cfg.artifacts.overwrite_existing_run)
    _, val_logits, val_labels=evaluate(model,splits["validation"],cfg,temperature=1.)
    try:
        threshold,temperature=select_threshold(val_logits,val_labels,cfg.decision.threshold_candidates,cfg.decision.min_taken_precision,cfg.decision.calibrate_temperature)
    except Exception as error:
        atomic_json(run/"metrics.json",{"split_counts":{k:len(v) for k,v in splits.items()},"quality_gate":{"passed":False},"threshold_selection_error":str(error)})
        raise
    test_metrics,test_logits,test_labels=evaluate(model,splits["test"],cfg,threshold,temperature)
    quality=quality_gate(test_metrics,cfg)
    passed=quality["passed"]
    metrics={"split_counts":{k:len(v) for k,v in splits.items()},"group_counts":{k:len(set(r.product_group_id for r in v)) for k,v in splits.items()},"selected_threshold":threshold,"temperature":temperature,"test":test_metrics,"roc_auc":roc_auc(test_logits,test_labels),"effective_device":str(next(model.parameters()).device),"duration_seconds":time.monotonic()-started,"quality_gate":quality}
    atomic_json(run/"metrics.json",metrics)
    atomic_json(run/"model_card.json",{"intended_use":"binary product image selection","exclusions":["metadata, EXIF, filenames"],"label_policy":{"taken":"product or suitable usage/instructions","not_taken":"advertisements, coupons, banners, campaign art"},"split_counts":metrics["split_counts"],"threshold":threshold,"temperature":temperature,"metrics":test_metrics,"quality_gate":metrics["quality_gate"],"known_limitations":["requires grouped labelled data"],"no_exif_guarantee":True})
    atomic_json(run/"effective_selector_config.json",cfg.model_dump(mode="json"))
    atomic_csv(run/"split_assignments.csv", ("image_id","image_path","label","product_group_id","split","pixel_hash"),
               ({"image_id":r.image_id,"image_path":r.image_path,"label":r.label.value,"product_group_id":r.product_group_id,
                 "split":split,"pixel_hash":preflight_result.pixel_hashes[r.image_id]}
                for split, rows in splits.items() for r in rows))
    atomic_csv(run/"test_predictions.csv", ("image_id","image_path","label","taken_probability","predicted_label"),
               ({"image_id":r.image_id,"image_path":r.image_path,"label":r.label.value,
                 "taken_probability":float(logits_to_taken_probability(logit.unsqueeze(0),temperature)[0]),
                 "predicted_label":"taken" if float(logits_to_taken_probability(logit.unsqueeze(0),temperature)[0])>=threshold else "not_taken"}
                for r,logit in zip(splits["test"],test_logits)))
    if not passed:
        raise RuntimeError("quality gate failed; production checkpoint was not promoted")
    metadata=checkpoint_metadata(cfg,digest,version); metadata.update({"temperature":temperature,"taken_threshold":threshold,"quality_gate":metrics["quality_gate"]})
    staging=run/"model.checkpoint"; save_checkpoint(staging,model,metadata)
    promote_checkpoint(staging,cfg.model.checkpoint_path)
if __name__=="__main__": main()

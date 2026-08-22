import csv, json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from image_translation.product_image_selection.config import load_config
from image_translation.product_image_selection.domain import LabeledImageRecord, ProductImageLabel
from image_translation.product_image_selection.exceptions import ManifestError
from image_translation.product_image_selection.io.discovery import discover_images
from image_translation.product_image_selection.io.manifest_csv import CsvManifestReader
from image_translation.product_image_selection.io.results_csv import CsvPredictionWriter
from image_translation.product_image_selection.preprocessing import preprocess_pixels
from image_translation.product_image_selection.split import grouped_split, pixel_hash

def _config(tmp):
    raw=json.loads(Path("product-image-selector.config.example.json").read_text())
    raw["data"]["training"]["image_root"]="images"; raw["data"]["training"]["labels_csv"]="labels.csv"
    raw["data"]["inference"]["image_roots"]=["images"]; raw["model"]["checkpoint_path"]="model.pt"
    raw["artifacts"]["output_directory"]="artifacts"; raw["output"]["csv_path"]="out.csv"
    p=tmp/"c.json"; p.write_text(json.dumps(raw)); return p
def test_config_and_manifest(tmp_path):
    (tmp_path/"images").mkdir()
    with (tmp_path/"images"/"a.weird").open("wb") as f: Image.new("RGB",(4,4),(1,2,3)).save(f,format="PNG")
    with (tmp_path/"labels.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["image_path","label","product_group_id","image_id"]); w.writerow(["a.weird","taken","g","id"])
    cfg=load_config(_config(tmp_path)); rows=CsvManifestReader(cfg.data.training.labels_csv,cfg.data.training.image_root).read()
    assert rows[0].label is ProductImageLabel.TAKEN
    (tmp_path/"labels.csv").write_text("image_path,label\nx,bad\n")
    with pytest.raises(ManifestError): CsvManifestReader(tmp_path/"labels.csv",tmp_path/"images").read()
def test_pixel_only_format_agnostic_and_discovery(tmp_path):
    root=tmp_path/"images"; root.mkdir(); pixels=np.zeros((3,5,3),dtype=np.uint8); pixels[...,0]=200
    with (root/"not-an-extension.data").open("wb") as f: Image.fromarray(pixels).save(f,format="PNG")
    (root/"bad.bin").write_text("no")
    found=list(discover_images([root],True,100)); assert [p.name for p in found]==["bad.bin","not-an-extension.data"]
    tensor=preprocess_pixels(pixels,8,(0,0,0),(1,1,1)); assert tensor.shape==(3,8,8); assert str(tensor.dtype)=="torch.float32"
def test_grouped_split_is_deterministic_and_disjoint():
    records=[LabeledImageRecord(str(i),ProductImageLabel.TAKEN if i//2 % 2 else ProductImageLabel.NOT_TAKEN,f"g{i//2}",str(i)) for i in range(12)]
    a=grouped_split(records,7); b=grouped_split(records,7)
    assert [[r.image_id for r in a[x]] for x in a]==[[r.image_id for r in b[x]] for x in b]
    assert not (set(r.product_group_id for r in a["train"]) & set(r.product_group_id for r in a["test"]))
def test_result_writer_stable_columns(tmp_path):
    from image_translation.product_image_selection.domain import CategorizationResult, ImageFailure, ImagePrediction
    p=ImagePrediction("x",.8,.2,ProductImageLabel.TAKEN,False,"v1")
    out=tmp_path/"r.csv"; CsvPredictionWriter(out).write(CategorizationResult((p,),(ImageFailure("b","b","bad"),)))
    with out.open() as f: rows=list(csv.DictReader(f))
    assert list(rows[0])==list(CsvPredictionWriter.fields); assert rows[1]["status"]=="failed"

def test_injected_model_inference_is_deterministic():
    import torch
    from types import SimpleNamespace
    from image_translation.product_image_selection.inference import ProductImageClassifier
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*4*4,2))
    with torch.no_grad(): model[1].weight.zero_(); model[1].bias[:] = torch.tensor([-1.,1.])
    cfg=SimpleNamespace(model=SimpleNamespace(input_size=4,normalization=SimpleNamespace(mean=(0,0,0),std=(1,1,1),padding_rgb=(0,0,0))),training=SimpleNamespace(device="cpu",allow_cpu_fallback=False),decision=SimpleNamespace(threshold_candidates=(.5,),review_band_half_width=.05),data=SimpleNamespace(inference=SimpleNamespace(inference_batch_size=2)))
    service=ProductImageClassifier(model,cfg,"test-v1","cpu",{"taken_threshold":.5,"temperature":1.})
    a=service.predict("image",np.ones((4,4,3),dtype=np.uint8)); b=service.predict("image",np.ones((4,4,3),dtype=np.uint8))
    assert a==b and a.label is ProductImageLabel.TAKEN and a.not_taken_probability < .2

def test_checkpoint_and_cpu_inference(tmp_path):
    import torch
    from image_translation.product_image_selection.inference import ProductImageClassifier
    from image_translation.product_image_selection.model import load_checkpoint, save_checkpoint
    cfg=load_config(_config(tmp_path))
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*cfg.model.input_size*cfg.model.input_size,2))
    save_checkpoint(tmp_path/"model.pt",model,{"artifact_schema_version":1,"architecture":"tiny","pretrained_weights":None,"labels":["not_taken","taken"],"input_size":cfg.model.input_size,"normalization":{"mean":[0,0,0],"std":[1,1,1],"padding_rgb":[0,0,0]},"temperature":1.,"taken_threshold":.5})
    loaded=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*cfg.model.input_size*cfg.model.input_size,2))
    load_checkpoint(tmp_path/"model.pt",loaded,{"artifact_schema_version":1})
    result=ProductImageClassifier(loaded,cfg,model_version="test",device="cpu",artifact={"taken_threshold":.5,"temperature":1.}).predict("fixture",np.zeros((8,8,3),dtype=np.uint8))
    assert result.model_version=="test"; assert result.taken_probability+result.not_taken_probability==pytest.approx(1)

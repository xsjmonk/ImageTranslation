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
from image_translation.product_image_selection.split import grouped_split, pixel_hash, validate_pixel_leakage
from image_translation.product_image_selection.training import (
    PreflightResult, logits_to_taken_probability, metrics_from_taken_probability,
    promote_if_quality_gate_passed,
)

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
    assert list(rows[0])==list(CsvPredictionWriter.fields)
    assert [row["status"] for row in rows].count("failed") == 1

def test_injected_model_inference_is_deterministic():
    import torch
    from types import SimpleNamespace
    from image_translation.product_image_selection.inference import ProductImageClassifier
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*4*4,2))
    with torch.no_grad(): model[1].weight.zero_(); model[1].bias[:] = torch.tensor([-1.,1.])
    normalization=SimpleNamespace(mean=(0,0,0),std=(1,1,1),padding_rgb=(0,0,0))
    cfg=SimpleNamespace(model=SimpleNamespace(architecture="tiny",input_size=4,normalization=normalization),training=SimpleNamespace(device="cpu",allow_cpu_fallback=False),decision=SimpleNamespace(threshold_candidates=(.5,),review_band_half_width=.05),data=SimpleNamespace(inference=SimpleNamespace(inference_batch_size=2)))
    service=ProductImageClassifier(model,cfg,"test-v1","cpu",{"taken_threshold":.5,"temperature":1.,"review_band_half_width":.05,"labels":["not_taken","taken"],"architecture":"tiny","input_size":4,"normalization":{"mean":[0,0,0],"std":[1,1,1],"padding_rgb":[0,0,0]}})
    a=service.predict("image",np.ones((4,4,3),dtype=np.uint8)); b=service.predict("image",np.ones((4,4,3),dtype=np.uint8))
    assert a==b and a.label is ProductImageLabel.TAKEN and a.not_taken_probability < .2

def test_checkpoint_and_cpu_inference(tmp_path):
    import torch
    from image_translation.product_image_selection.inference import ProductImageClassifier
    from image_translation.product_image_selection.model import load_checkpoint, save_checkpoint
    cfg=load_config(_config(tmp_path))
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*cfg.model.input_size*cfg.model.input_size,2))
    save_checkpoint(tmp_path/"model.pt",model,{"artifact_schema_version":1,"architecture":cfg.model.architecture,"pretrained_weights":None,"labels":["not_taken","taken"],"input_size":cfg.model.input_size,"normalization":cfg.model.normalization.model_dump(),"temperature":1.,"taken_threshold":.5,"review_band_half_width":.05,"quality_gate":{"passed":True,"checks":{"fixture":{"actual":1,"required":1,"passed":True}}}})
    loaded=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*cfg.model.input_size*cfg.model.input_size,2))
    metadata=load_checkpoint(tmp_path/"model.pt",loaded,{"artifact_schema_version":1})
    result=ProductImageClassifier(loaded,cfg,model_version="test",device="cpu",artifact=metadata).predict("fixture",np.zeros((8,8,3),dtype=np.uint8))
    assert result.model_version=="test"; assert result.taken_probability+result.not_taken_probability==pytest.approx(1)


def _records(groups):
    return [
        LabeledImageRecord(f"{group}-{index}", label, group, f"{group}-{index}")
        for group, labels in groups.items()
        for index, label in enumerate(labels)
    ]


def test_mixed_label_groups_are_assigned_once_with_diagnostics():
    records = _records({f"g{i}": (ProductImageLabel.TAKEN, ProductImageLabel.NOT_TAKEN)
                        for i in range(6)})
    splits = grouped_split(records, seed=11, minimum_groups_per_class=1)
    owners = {record.product_group_id: split for split, rows in splits.items() for record in rows}
    assert len(owners) == 6
    assert all({row.label.value for row in rows} == {"taken", "not_taken"} for rows in splits.values())
    assert set(splits.diagnostics["group_assignments"]) == set(owners)
    assert splits.diagnostics["actual_totals"] == {name: len(rows) for name, rows in splits.items()}


def test_imbalanced_group_sizes_influence_total_targets():
    records = _records({
        "large-taken": tuple([ProductImageLabel.TAKEN] * 10),
        "small-taken": (ProductImageLabel.TAKEN,),
        "small-taken-2": (ProductImageLabel.TAKEN,),
        "small-not-1": (ProductImageLabel.NOT_TAKEN,),
        "small-not-2": (ProductImageLabel.NOT_TAKEN,),
        "small-not-3": (ProductImageLabel.NOT_TAKEN,),
        "small-not-4": (ProductImageLabel.NOT_TAKEN,),
        "small-not-5": (ProductImageLabel.NOT_TAKEN,),
    })
    splits = grouped_split(records, seed=3, minimum_groups_per_class=1)
    assert sum(len(rows) for rows in splits.values()) == len(records)
    assert splits.diagnostics["actual_totals"]["train"] >= splits.diagnostics["actual_totals"]["test"]


def test_impossible_group_support_is_rejected():
    records = _records({"g1": (ProductImageLabel.TAKEN,), "g2": (ProductImageLabel.NOT_TAKEN,)})
    with pytest.raises(ManifestError, match="groups per class"):
        grouped_split(records, minimum_groups_per_class=1)


def test_pixel_hash_is_shape_aware_and_leakage_rejects_labels_and_groups():
    first = np.zeros((2, 3, 3), dtype=np.uint8)
    second = np.zeros((3, 2, 3), dtype=np.uint8)
    assert pixel_hash(first) != pixel_hash(second)
    records = [
        LabeledImageRecord("a", ProductImageLabel.TAKEN, "g1", "a"),
        LabeledImageRecord("b", ProductImageLabel.NOT_TAKEN, "g1", "b"),
    ]
    with pytest.raises(ManifestError, match="conflicting"):
        validate_pixel_leakage(records, (first, first))
    records[1] = LabeledImageRecord("b", ProductImageLabel.TAKEN, "g2", "b")
    with pytest.raises(ManifestError, match="groups"):
        validate_pixel_leakage(records, (first, first))


def test_temperature_probability_path_matches_emitted_metrics():
    import torch
    logits = torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.5, 0.7], [0.7, 0.5]])
    labels = torch.tensor([1, 0, 1, 0])
    probabilities = logits_to_taken_probability(logits, temperature=2.0)
    metrics = metrics_from_taken_probability(probabilities, labels, threshold=.55)
    assert metrics == metrics_from_taken_probability(probabilities, labels, threshold=.55)
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]


def test_preflight_result_is_reused_without_second_preflight(monkeypatch, tmp_path):
    import torch
    from types import SimpleNamespace
    import image_translation.product_image_selection.training as training
    image = tmp_path / "x.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(image)
    records = _records({"g0": (ProductImageLabel.TAKEN,), "g1": (ProductImageLabel.NOT_TAKEN,)})
    # The composition contract is exercised at the train seam: a supplied
    # result must be used directly, never replaced by a second preflight.
    splits = {"train": records[:1], "validation": records[1:], "test": records[:1]}
    result = PreflightResult(tuple(records), (np.zeros((4, 4, 3), dtype=np.uint8),) * 2,
                             {r.image_id: "hash" for r in records}, splits, {})
    called = []
    monkeypatch.setattr(training, "preflight", lambda *_: called.append(True))
    normalization = SimpleNamespace(mean=(0, 0, 0), std=(1, 1, 1), padding_rgb=(0, 0, 0))
    cfg = SimpleNamespace(
        model=SimpleNamespace(input_size=4, normalization=normalization),
        data=SimpleNamespace(training=SimpleNamespace(max_decoded_pixels=100)),
        training=SimpleNamespace(seed=1, deterministic_algorithms=False, device="cpu",
            allow_cpu_fallback=False, batch_size=1, num_workers=0, mixed_precision=False,
            augmentation=SimpleNamespace(enabled=False), head_warmup_epochs=0,
            fine_tune_epochs=0, learning_rate_head=.01, learning_rate_backbone=.001,
            weight_decay=0., early_stopping_patience=1),
    )
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))
    training.train(model, records, cfg, preflight_result=result)
    assert called == []


def test_incomplete_checkpoint_is_rejected(tmp_path):
    import torch
    from image_translation.product_image_selection.model import load_checkpoint
    path = tmp_path / "bad.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(Exception, match="schema"):
        load_checkpoint(path, torch.nn.Linear(2, 2))


def test_artifact_review_width_is_authoritative():
    import torch
    from types import SimpleNamespace
    from image_translation.product_image_selection.inference import ProductImageClassifier
    normalization = SimpleNamespace(mean=(0, 0, 0), std=(1, 1, 1), padding_rgb=(0, 0, 0))
    cfg = SimpleNamespace(model=SimpleNamespace(architecture="tiny", input_size=4,
        normalization=normalization), training=SimpleNamespace(device="cpu",
        allow_cpu_fallback=False), data=SimpleNamespace(inference=SimpleNamespace(
            inference_batch_size=1)))
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))
    with torch.no_grad():
        model[1].weight.zero_(); model[1].bias.zero_()
    artifact = {"taken_threshold": .5, "temperature": 1., "review_band_half_width": .25,
                "labels": ["not_taken", "taken"], "architecture": "tiny", "input_size": 4,
                "normalization": {"mean": [0, 0, 0], "std": [1, 1, 1], "padding_rgb": [0, 0, 0]}}
    prediction = ProductImageClassifier(model, cfg, artifact=artifact).predict(
        "x", np.zeros((4, 4, 3), dtype=np.uint8))
    assert prediction.requires_review


def test_categorization_report_includes_malformed_nested_and_nonstandard_files(tmp_path):
    from types import SimpleNamespace
    from image_translation.product_image_selection.categorize_cli import categorize_images
    from image_translation.product_image_selection.domain import CategorizationResult, ImagePrediction
    root = tmp_path / "images"; nested = root / "nested"; nested.mkdir(parents=True)
    readable = root / "read.data"; Image.new("RGB", (3, 3), (1, 2, 3)).save(readable, format="PNG")
    Image.new("RGB", (3, 3), (4, 5, 6)).save(nested / "nested.bin", format="PNG")
    malformed = root / "bad.file"; malformed.write_text("not an image")
    output = tmp_path / "report.csv"
    cfg = SimpleNamespace(
        data=SimpleNamespace(inference=SimpleNamespace(image_roots=(root,), recursive=True,
            max_decoded_pixels=100, unreadable_image_policy="report")),
        output=SimpleNamespace(csv_path=output))
    class FakeClassifier:
        def categorize(self, items):
            return CategorizationResult(tuple(
                ImagePrediction(item[0], .8, .2, ProductImageLabel.TAKEN, False, "tiny", item[2])
                for item in items), ())
    result, had_failures = categorize_images(cfg, FakeClassifier())
    assert had_failures and len(result.predictions) == 2 and len(result.failures) == 1
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["image_path"] for row in rows] == sorted(row["image_path"] for row in rows)
    assert [row["status"] for row in rows].count("ok") == 2
    assert [row["status"] for row in rows].count("failed") == 1
    assert next(row for row in rows if row["status"] == "failed")["image_path"].endswith("bad.file")


def test_categorization_fail_policy_does_not_invoke_classifier(tmp_path):
    from types import SimpleNamespace
    from image_translation.product_image_selection.categorize_cli import categorize_images
    root = tmp_path / "images"; root.mkdir(); (root / "bad").write_text("bad")
    cfg = SimpleNamespace(data=SimpleNamespace(inference=SimpleNamespace(
        image_roots=(root,), recursive=False, max_decoded_pixels=100, unreadable_image_policy="fail")),
        output=SimpleNamespace(csv_path=tmp_path / "out.csv"))
    class NeverClassifier:
        def categorize(self, items):
            raise AssertionError("classifier must not run")
    with pytest.raises(RuntimeError, match="unreadable"):
        categorize_images(cfg, NeverClassifier())


def test_missing_inference_root_fails_before_enumeration(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        list(discover_images([tmp_path / "missing"], True))


def test_failed_quality_gate_preserves_checkpoint_and_writes_diagnostics(tmp_path):
    production = tmp_path / "production.pt"
    staged = tmp_path / "staged.pt"
    diagnostics = tmp_path / "run" / "metrics.json"
    production.write_bytes(b"known production checkpoint")
    staged.write_bytes(b"failed candidate")
    summary = {"passed": False, "checks": {
        "taken_precision": {"actual": .1, "required": .9, "passed": False}}}
    with pytest.raises(RuntimeError, match="quality gate failed"):
        promote_if_quality_gate_passed(staged, production, summary, diagnostics,
                                       {"metrics": {"taken_precision": .1},
                                        "quality_gate": summary,
                                        "split_assignments": ["g1"],
                                        "test_predictions": ["a"]})
    assert production.read_bytes() == b"known production checkpoint"
    assert staged.exists()
    assert json.loads(diagnostics.read_text())["quality_gate"]["checks"]["taken_precision"]["passed"] is False


def test_raster_reader_ignores_exif_apis(monkeypatch, tmp_path):
    from image_translation.product_image_selection.io.raster_pillow import PillowRasterReader
    pixels = np.full((3, 4, 3), 90, dtype=np.uint8)
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    Image.fromarray(pixels).save(first, format="PNG")
    Image.fromarray(pixels).save(second, format="PNG")
    monkeypatch.setattr(Image.Image, "getexif", lambda self: (_ for _ in ()).throw(AssertionError("EXIF accessed")))
    reader = PillowRasterReader(100)
    assert np.array_equal(reader.read_rgb_pixels(first), reader.read_rgb_pixels(second))


def test_selector_boundary_and_powerShell_scripts_parse():
    import subprocess
    package = Path(__file__).parents[2] / "src" / "image_translation" / "product_image_selection"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.rglob("*.py"))
    assert "fastapi" not in source.lower()
    assert "translation_server" not in source.lower()
    scripts = ["Initialize-Env.ps1", "Start-TranslationServer.ps1",
               "Train-ProductImageSelector.ps1", "Categorize-ProductImages.ps1"]
    root = Path(__file__).parents[2]
    for script in scripts:
        script_path = str(root / "script" / script).replace("'", "''")
        command = (
            "$tokens=$null;$errors=$null;"
            f"$null=[System.Management.Automation.Language.Parser]::ParseFile('{script_path}',"
            "[ref]$tokens,[ref]$errors);if($errors.Count){exit 1}"
        )
        completed = subprocess.run(["pwsh", "-NoProfile", "-Command", command],
                                   cwd=root, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr

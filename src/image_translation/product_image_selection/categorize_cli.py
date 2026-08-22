import argparse
from .config import load_config
from .model import build_model, load_checkpoint
from .inference import ProductImageClassifier
from .io.discovery import discover_images
from .io.raster_pillow import PillowRasterReader
from .io.results_csv import CsvPredictionWriter
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True)
    cfg=load_config(parser.parse_args().config)
    model=build_model(cfg.model.architecture,None)
    metadata=load_checkpoint(cfg.model.checkpoint_path,model,{"architecture":cfg.model.architecture,"pretrained_weights":cfg.model.pretrained_weights,"labels":["not_taken","taken"],"input_size":cfg.model.input_size,"normalization":cfg.model.normalization.model_dump()})
    classifier=ProductImageClassifier(model,cfg,metadata.get("model_version","unknown"),"cpu" if cfg.training.device=="cpu" else None,metadata)
    reader=PillowRasterReader(cfg.data.inference.max_decoded_pixels); items=[]; failures=[]
    for path in discover_images(cfg.data.inference.image_roots,cfg.data.inference.recursive):
        try: items.append((str(path),reader.read_rgb_pixels(str(path))))
        except Exception as e: failures.append((path.name,str(path),str(e)))
    if failures and cfg.data.inference.unreadable_image_policy == "fail":
        raise RuntimeError(f"{len(failures)} unreadable candidate image(s)")
    result=classifier.categorize(items)
    from .domain import CategorizationResult, ImageFailure
    result=CategorizationResult(result.predictions,result.failures+tuple(ImageFailure(i,p,e) for i,p,e in failures))
    cfg.output.csv_path.parent.mkdir(parents=True,exist_ok=True); CsvPredictionWriter(cfg.output.csv_path).write(result)
    if failures: raise RuntimeError(f"{len(failures)} unreadable candidate image(s) reported")
if __name__=="__main__": main()

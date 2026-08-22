import csv
from pathlib import Path
from ..domain import LabeledImageRecord, ProductImageLabel
from ..exceptions import ManifestError
from .raster_pillow import PillowRasterReader
class CsvManifestReader:
    def __init__(self, path, image_root, columns=None, max_decoded_pixels=80_000_000):
        self.path=Path(path); self.image_root=Path(image_root); self.columns=columns or {}
        self.reader=PillowRasterReader(max_decoded_pixels)
    def read(self):
        names={k: self.columns.get(k,k) for k in ("image_path","label","product_group_id","image_id")}
        required=list(names.values()); out=[]; ids=set()
        with self.path.open(encoding="utf-8-sig", newline="") as f:
            rows=csv.DictReader(f)
            if not rows.fieldnames or any(x not in rows.fieldnames for x in required): raise ManifestError("missing required headers")
            for r in rows:
                try:
                    values={k:(r.get(v) or "").strip() for k,v in names.items()}
                    if any(not v for v in values.values()): raise ValueError("blank required value")
                    image_id=values["image_id"]
                    if image_id in ids: raise ValueError("duplicate image_id")
                    ids.add(image_id); label=ProductImageLabel(values["label"])
                    path=Path(values["image_path"])
                    if path.is_absolute(): path=path if self.columns.get("allow_absolute_image_paths") else (_ for _ in ()).throw(ValueError("absolute path not allowed"))
                    else: path=self.image_root/path
                    if not path.is_file(): raise ValueError("missing image")
                    self.reader.read_rgb_pixels(str(path))
                    out.append(LabeledImageRecord(str(path),label,values["product_group_id"],image_id))
                except Exception as e: raise ManifestError(f"invalid row: {e}") from e
        return out

# LocalImageProcessing

Repository-owned deterministic image operations for the `translate-image` skill
and other callers. This module performs pixel work only. It does **not** OCR,
translate, classify logos, load models, or make remote calls.

## Authoritative contract

- **Version:** `1.0` (`CONTRACT_VERSION` in `contract.py`)
- **Schema:** `RegionManifest`, `ImageManifest`, `BatchManifest`
- **Actions:** `translate`, `remove`, `preserve`, `review`
- **Reconstruction:** `inpaint` (default) or explicit `blur_fallback`

Invalid geometry, missing `translated_text` for `translate` regions, clipping,
or format loss fail with actionable diagnostics. The module never guesses text.

### Single-image manifest

```json
{
  "contract_version": "1.0",
  "source_path": "D:/photos/product01.jpg",
  "source_hash": "optional-sha256",
  "regions": [
    {
      "id": "region-001",
      "polygon": [[10, 20], [200, 20], [200, 60], [10, 60]],
      "orientation_degrees": 0,
      "action": "translate",
      "source_text": "加厚防水",
      "translated_text": "THICK WATERPROOF FABRIC",
      "style": {
        "alignment": "center",
        "color_rgb": [255, 255, 255],
        "opacity": 1.0
      },
      "reconstruction": { "method": "inpaint" }
    }
  ]
}
```

### Batch manifest

```json
{
  "contract_version": "1.0",
  "images": [
    { "source_path": "...", "regions": [] }
  ]
}
```

## Independence

`LocalImageProcessing` is self-contained. It does **not** import
`image_translation`, OCR engines, translation models, or the translation server.
It uses only packages declared in `environment.yml`.

Owned components live under `LocalImageProcessing/`:

- `io/` — image load/save, atomic files, JSON
- `imaging/` — masks, inpainting, inpaint QC
- `revision/` — layout, rendering, compositing
- `qc/` — pixel validation and geometry
- `env_check.py` — dependency validation for this module
- `exit_codes.py` — CLI exit-code policy

## Entry points

```powershell
$env:PYTHONPATH = "D:\Drop\outlook.com\LocalBox\ImageTranslation"
conda run -n dp python -m LocalImageProcessing process --manifest manifest.json -o D:\work\photos_processed
conda run -n dp python -m LocalImageProcessing check-env
```

Exit codes for `process`:

- `0` — all images `success` or `skipped`
- `1` — at least one hard `failure`
- `2` — no failures, but `partial` or `review` outcomes remain

Skill launcher:

```powershell
D:\Drop\outlook.com\LocalBox\Code\Skills\ImageTranslation\scripts\translate-image.ps1 -Manifest manifest.json -OutputFolder D:\work\photos_processed
```

## Outputs

Default archive layout:

```text
<output>_processed/
    original_image01.jpg
    image01.jpg              # localized candidate (not promoted by default)
    metadata/
        image01.json
    summary.json
```

- Preserved originals use the exact `original_<source filename>` prefix.
- Candidates are written separately until the host agent approves promotion.
- Use `--promote` only after visual QA to replace the source in place.

## Environment

Uses the repository `dp` conda environment from `environment.yml`.
Initialize with `script\Initialize-Env.ps1` and validate with
`script\Check-LocalImageEnv.ps1` or `python -m LocalImageProcessing check-env`.

## Responsibility split

| Host agent | LocalImageProcessing |
|------------|----------------------|
| Visual recognition and OCR | Image I/O and format preservation |
| Translation and terminology | Manifest validation |
| Region classification | Masks, inpainting, blur fallback |
| Visual QA and promotion | Supplied-text rendering |
| Manifest authoring | Diagnostics and atomic outputs |

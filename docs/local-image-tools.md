# Local image tools



Repository-owned deterministic image operations for the `translate-image` skill

and other callers.



## Module



- **Path:** `LocalImageProcessing/`

- **Contract:** `LocalImageProcessing/README.md` (version `1.0`)

- **Entry point:** `python -m LocalImageProcessing`



The module validates agent manifests, performs masking/inpainting/rendering, and

writes candidate outputs plus diagnostics. It does **not** OCR, translate,

classify logos, load models, or make remote calls.



## Environment



- **Conda environment:** `dp` (from `environment.yml`)

- **Initialize / update (idempotent):**



```powershell

D:\Drop\outlook.com\LocalBox\ImageTranslation\script\Initialize-Env.ps1

```



Do **not** use ad-hoc `pip install` during setup checks. Change

`environment.yml` and re-run `Initialize-Env.ps1` instead.



## Required local packages



| Purpose | Packages |

|---------|----------|

| Load/save common formats | `pillow`, `opencv`, `imageio` |

| Arrays, filters, inpaint math | `numpy`, `scipy`, `scikit-image` |

| Polygon geometry | `shapely`, `pyclipper` |

| Config/validation utilities | `pydantic`, `jsonschema`, `pyyaml` |



OCR/translation packages remain in the same environment because they are already

declared in `environment.yml`. Local image processing does **not** load them.



## Commands



Repository:



```powershell

$env:PYTHONPATH = "D:\Drop\outlook.com\LocalBox\ImageTranslation"

conda run -n dp python -m LocalImageProcessing check-env

conda run -n dp python -m LocalImageProcessing process --manifest manifest.json -o D:\work\photos_processed

```



Read-only environment check:



```powershell

D:\Drop\outlook.com\LocalBox\ImageTranslation\script\Check-LocalImageEnv.ps1

```



Skill wrappers:



```powershell

D:\Drop\outlook.com\LocalBox\Code\Skills\ImageTranslation\scripts\check-local-image-env.ps1

D:\Drop\outlook.com\LocalBox\Code\Skills\ImageTranslation\scripts\translate-image.ps1 -Manifest manifest.json -OutputFolder D:\work\photos_processed

```



Override repository path with `IMAGE_TRANSLATION_REPO` when needed.

Override environment name with `IMAGE_TRANSLATION_CONDA_ENV` when needed.

`process` exit codes: `0` success/skipped only, `1` hard failure, `2` partial/review without failure.

## Testing

Default repository tests configure a writable temp base at `.pytest_tmp/` when the
system temp directory is inaccessible on Windows.

```powershell
conda run -n dp python -m pytest tests/local_image_processing -v
conda run -n dp python -m pytest tests/ -v
```

## Outputs



```text

<output>_processed/

    original_image01.jpg

    image01.jpg

    metadata/

        image01.json

    summary.json

```



- Preserved originals use the exact `original_<source filename>` prefix.

- Candidates are written separately until the host agent approves promotion.

- Use `--promote` / `-Promote` only after visual QA.



## Failure messages



| Problem | Action |

|---------|--------|

| Conda not found | Install Miniconda/Anaconda |

| `dp` missing | Run `Initialize-Env.ps1` |

| Import/operation check fails | Re-run `Initialize-Env.ps1` after editing `environment.yml` |

| Manifest validation fails | Fix geometry, actions, or missing `translated_text` |

| Skill launcher not found | Set `IMAGE_TRANSLATION_REPO` to the repository root |



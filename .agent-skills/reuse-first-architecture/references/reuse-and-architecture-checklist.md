# Reuse and architecture checklist

Run this checklist before and after any architecture-sensitive change.

## Before editing

- [ ] Read repository agent instructions (`AGENTS.md`).
- [ ] Read the dependency manifest (`environment.yml`) — it is the single
      source of truth; never create another packaging manifest.
- [ ] Read the README sections relevant to the change.
- [ ] List the public entry points and interfaces that could be affected
      (`Translator`, `StructuredTranslator`, `create_translator`,
      `create_app`, `load_server_config`, CLI entry points, HTTP routes).
- [ ] Search by behavior and concept for existing implementations
      (grep for the capability, not only the requested name).
- [ ] Check callers: who calls the component, with which data contracts,
      side effects, error types, configuration paths, and threading
      assumptions.
- [ ] Check whether an existing dependency or stdlib feature already
      solves the problem; add a dependency only with version evidence.
- [ ] Classify the change: reuse / adapt / extend / replace, with
      evidence. Write it down before implementing.
- [ ] If the requirement is ambiguous but a safe reversible assumption
      exists, state the assumption and continue. Ask only when the choice
      materially changes architecture or external behavior.

## Dependency-direction checks (ImageTranslation)

- [ ] No `fastapi`/`uvicorn`/`translation_server` imports under
      `src/image_translation` (verify with grep).
- [ ] Model/tokenizer resolution has one owner; HTML segmentation measures
      through `Translator.measure_source_tokens`, never a second
      `from_pretrained`.
- [ ] Cache, offline, revision, GPU, precision, and language-pair policy
      have one owner (`Seq2SeqTranslator` + model-family adapter + config).
- [ ] No independent Hugging Face access from HTML segmentation or other
      callers after the model is loaded.

## After editing

- [ ] No duplicate logic introduced (model loading, parsing/protection/
      reconstruction, config/path normalization, HTTP/error handling,
      retry/timeout/cache policy, validation, fixtures).
- [ ] No public method or API default changed without authorization.
- [ ] Ordering, cardinality, idempotency, and no-partial-result guarantees
      preserved.
- [ ] Failure/timeout/cancellation/concurrency behavior defined and tested
      where relevant.
- [ ] Configuration validation stays at its loading boundary.
- [ ] Tests cover the changed contract and the most likely regression.
- [ ] Documentation/configuration matches the implementation.
- [ ] Diff contains only intended changes; unrelated files untouched.

## Verification commands (ImageTranslation)

```powershell
# Focused
conda run -n dp python -m pytest tests/translation/test_model_cache.py tests/translation/test_batching.py tests/translation/test_structured_translation.py tests/translation_server/test_api.py tests/translation_server/test_api_html.py -q
# Broadest practical
conda run -n dp python -m pytest tests/ -q
# Real GPU quality (requires NVIDIA GPU + cached model)
conda run -n dp python -m pytest tests/translation/test_quality_regression.py tests/translation/test_html_gpu_quality.py -q -s
```

If any command cannot run, report the exact command, the reason, and the
remaining risk. Never claim a suite passed without running it.

# SOLID and dependency boundaries

Concrete application of SOLID with the repository examples that make the
rules testable.

## Single Responsibility

| Responsibility | Owner (ImageTranslation example) |
| --- | --- |
| HTTP transport | `src/translation_server` (FastAPI app, api models, runtime) |
| Orchestration | `StructuredTranslator` (HTML-aware translation flow) |
| Model inference | `Seq2SeqTranslator` + `ModelFamilyAdapter` |
| Parsing/protection/reconstruction | `html_document`, `html_protection`, `reconstruction` |
| Cache/resolution policy | `Seq2SeqTranslator._resolve_model_snapshot` + `ResolvedModel` |
| Configuration validation | config modules (`TranslationConfig`, server config) |
| Filesystem/JSON/image utilities | `utilities/` |

Rule: if a new responsibility does not map to one of these owners, decide
its owner before writing code. Do not put business logic into routes,
CLI argument parsing, or test helpers.

## Open/Closed

- Variation is real (new translation engines, cache policies, transports):
  extend through the existing interfaces (`Translator`, configuration
  dataclasses, `create_translator` factory).
- Do not add speculative strategies/plugins with a single implementation.

## Liskov Substitution

Implementations of `Translator` must preserve:

- method contracts and return shapes (`TranslationResult`, lists in input
  order);
- cardinality (one result per input; validated batch boundaries);
- error semantics (structured exceptions, fail-closed, no partial output);
- ordering guarantees (source order, strict placeholder sequence);
- timeout/deadline and concurrency behavior (per-call lock, deadline
  checks between batches).

Test fakes must implement the same contracts (e.g. `measure_source_tokens`
with identical count semantics).

## Interface Segregation

- `Translator` stays small and capability-oriented:
  `translate_text`, `translate_batch_texts`, `warmup`, `runtime_info`,
  `measure_source_tokens`.
- Optional capabilities are injected callbacks or narrow interfaces, not
  methods forced onto every implementation.

## Dependency Inversion

- `StructuredTranslator` depends on the `Translator` protocol, not on
  Transformers/GPU classes.
- `translation_server` depends on the shared translation module; the
  shared module never imports FastAPI or server code (static check:
  no `fastapi`/`translation_server` imports under `src/image_translation`).
- Concrete adapters are wired at composition roots: `create_translator`,
  `create_app`, `load_server_config`, CLI/server startup.

## Rejected patterns

- Service locators and module-level mutable service singletons
  (pipeline services are constructed and injected).
- God classes and circular imports.
- Hidden global state except documented caches with clear ownership and
  tests (e.g. the test-only tokenizer cache).
- Silent fallback (default cache when an explicit cache is configured,
  CPU when GPU is required, truncated inputs).
- Catch-all exception swallowing; `except Exception: pass`.
- Duplicate infrastructure: model loading, HTML parsing/protection,
  configuration/path normalization, HTTP/error handling, retry/timeout/
  cache policy, domain validation, test fixtures.

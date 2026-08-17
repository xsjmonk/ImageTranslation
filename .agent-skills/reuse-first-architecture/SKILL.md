---
name: reuse-first-architecture
description: Apply reuse-first architecture, SOLID design, dependency inversion, contract preservation, and evidence-based implementation to every programming change. Use before writing or editing code, tests, scripts, configuration, or infrastructure — including features, bug fixes, refactors, library or dependency selection, and changes to services, parsers, model loaders, clients, configuration, or infrastructure.
---

# Reuse-first architecture

Workflow for modifying an existing codebase without reinventing capabilities.
Follow the steps in order. Do not skip the search, design, or verification
phases. The final report must contain evidence, not assurances.

MANDATORY PRE-CODING GATE

Before creating, editing, or deleting any code, test code, script,
configuration code, or infrastructure definition:

1. Read this skill and the repository agent instructions.
2. Search for existing components, interfaces, factories, helpers,
   libraries, configuration, and tests that may already provide the behavior.
3. State a concise decision: reuse, adapt, extend, or build, naming the
   component inspected and the reason.
4. Identify the affected public contract and dependency direction.
5. Only then write code.

For a trivial change, perform the gate briefly. "No architecture impact"
is acceptable, but skipping the gate is not. Read-only inspection may happen
before the gate; code edits may not.

## 1. Establish constraints

- Read the repository agent instructions (`AGENTS.md` or equivalent) before
  changing anything; they take precedence where more specific.
- Inspect the project layout, dependency manifests, configuration files,
  public entry points, interfaces, implementations, tests, and relevant
  documentation.
- Identify the language/runtime, the dependency source of truth, the test
  commands, and the deployment constraints.
- Preserve unrelated worktree edits. Avoid destructive commands.

## 2. Search before building

- Search by behavior and concept, not only by the requested name.
- Look for existing classes, functions, adapters, factories, helpers,
  libraries, configuration, and tests that already implement or
  approximate the requirement.
- Check project dependencies and the standard library before writing code.
- Decide, in order of preference:
  1. **Reuse** the existing component unchanged;
  2. **Adapt** it behind a narrow adapter/facade;
  3. **Extend** its owning abstraction while preserving callers;
  4. **Replace** only with migration and compatibility tests.
- Never claim a mature solution was reused without naming the component,
  its contract, and the tests that prove the contract.

## 3. Define architecture before editing

Write a compact design containing:

- current-state data flow, ownership, relevant interfaces and callers;
- the reuse/adapt/extend/replace decision and its evidence;
- component responsibilities and dependency direction;
- public-contract impact and backward compatibility;
- lifecycle/concurrency/timeout/cache/failure behavior;
- the test strategy mapped to acceptance criteria.

Use a small text diagram only when it clarifies real modules or boundaries.
Never draw architecture theater: every box must map to a real module,
interface, adapter, or composition root.

## 4. Apply SOLID concretely

- **Single Responsibility**: transport, orchestration, domain rules,
  persistence/cache, parsing, model inference, and presentation have
  distinct owners. No business logic in routes, CLI parsing, or test
  helpers.
- **Open/Closed**: extend through stable seams where variation is real;
  do not create speculative plugin frameworks with one implementation.
- **Liskov Substitution**: implementations preserve method contracts,
  ordering, cardinality, error semantics, and no-partial-result
  guarantees.
- **Interface Segregation**: small capability interfaces; do not force
  every implementation to support unrelated methods.
- **Dependency Inversion**: high-level logic depends on protocols, not
  concrete libraries; concrete adapters are wired only at composition
  roots (factories, runtime builders, startup).

Reject: service locators, god classes, circular imports, hidden global
mutable state, silent fallback, magic strings, duplicate infrastructure,
and abstractions without a real seam.

## 5. Implement one coherent slice

- Make the smallest complete production change.
- Put behavior in the module that owns the responsibility; keep transport
  layers thin.
- Inject filesystem, network, clocks, model loaders, and external services
  where tests need isolation.
- Preserve public methods and defaults unless a breaking change is
  explicitly authorized.
- Keep configuration validation at its loading boundary.
- Do not refactor unrelated files merely for style.

## 6. Verify and report

- Run focused unit tests, then relevant integration/API tests, then the
  broadest practical suite within the environment limit.
- Inspect the final diff for duplicate logic, dependency-direction
  violations, untested public changes, and accidental behavior changes.
- If a suite cannot run, report the exact command, the reason, and the
  remaining risk. Never claim it passed.
- Do not report completion until implementation, tests,
  documentation/configuration, and evidence are all complete.

## Repository-specific rules

See `references/solid-and-boundaries.md` and
`references/reuse-and-architecture-checklist.md` for detailed checklists
and the architecture contracts of the ImageTranslation repository.

These repository rules are labeled so the skill stays reusable elsewhere:
- [ImageTranslation] shared translation must stay transport-independent;
  `src/translation_server` may depend on it, never the reverse.
- [ImageTranslation] model/tokenizer resolution is centralized; HTML
  segmentation must not download or load another tokenizer.
- [ImageTranslation] cache, offline, revision, GPU, precision, and
  language-pair policies have one owner.
- [ImageTranslation] `environment.yml` is the dependency source of truth.

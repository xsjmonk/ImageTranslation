---
name: reuse-first-architecture
description: Apply reuse-first architecture, SOLID design, dependency inversion, contract preservation, evidence-based implementation, and bounded execution/waiting during tests, subprocesses, environment setup, and long operations. Use before writing or editing code, tests, scripts, configuration, or infrastructure — including features, bug fixes, refactors, library or dependency selection, and changes to services, parsers, model loaders, clients, configuration, or infrastructure.
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

## Canonical source and discoverability

This file is the single canonical architecture skill for this repository:

```text
.agent-skills/reuse-first-architecture/SKILL.md
```

Do not copy this skill body into `AGENTS.md`, README files, agent rules,
Cursor rules, or other agent configuration. Copies drift. Other files may
contain only a short instruction to read this path. `AGENTS.md` is the
portable fallback pointer for agents without skill discovery.

## Self-application for skill maintenance

An agent that edits or revises this skill must follow the same gate before
changing it:

1. read `AGENTS.md` and the complete existing canonical skill;
2. search for skill references, duplicate copies, and existing wait/timeout
   conventions;
3. state reuse/adapt/extend and identify the public contract being changed
   (this path and its discoverability contract);
4. make the smallest backward-compatible change that preserves all existing
   architecture rules;
5. edit only this canonical skill and necessary short reference pointers;
6. validate that all required original rules remain present;
7. report exact validation evidence.

Do not rewrite the skill from memory or weaken the pre-coding gate while
adding new policy.

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

Use bounded execution while verifying (see **Bounded execution and waiting**
below). Run tests in this order:

1. the smallest relevant focused test;
2. the relevant module/API tests;
3. the broadest practical suite once.

Inspect the final diff for duplicate logic, dependency-direction violations,
untested public changes, and accidental behavior changes. If a suite cannot
run, report the exact command, elapsed time, the reason, and the remaining
risk. Never claim it passed. Do not report completion until implementation,
tests, documentation/configuration, and evidence are all complete.

Do not optimize or redesign tests solely because a suite completes in a
normal historical range. For this repository, a measured translation-server
test runtime of approximately **17.4 seconds** is acceptable.

## Bounded execution and waiting

Prevent blind waiting during tests, model operations, environment setup,
subprocesses, or delegated work. These rules apply to agent command execution,
not to production server design.

### Default command observation intervals

Use bounded observation by default. These are **poll intervals**, not
permission to block for the entire turn. If a command is still running at an
interval, inspect process state and fresh output before waiting again.

| Operation | Observation interval |
|-----------|---------------------|
| quick inspection, import, compile checks | up to 30 seconds |
| focused unit tests | up to 60 seconds |
| normal repository test suite | up to 120 seconds before inspecting progress |
| script/config/API smoke checks | up to 60 seconds per check |
| environment init, model download, real GPU inference | progress-aware long-operation procedure (below) |

**Never** issue an unconditional sleep or wait longer than **60 seconds**.
**Never** use an arbitrary 10-minute wait merely because a command might
eventually finish.

Distinguish:

- normal bounded verification that completes in seconds;
- genuinely long operations (environment creation, model download, GPU smoke
  tests, server processes);
- an operation that has stopped making progress.

### Long-operation procedure

For `Initialize-Env.ps1`, model downloads, server startup, GPU inference, or
other work expected to exceed two minutes:

1. start with captured stdout/stderr and a known process identifier or
   supported session handle;
2. record start time and why the operation may be long;
3. poll at bounded intervals of **15–60 seconds**;
4. after each poll, inspect fresh output, process liveness, and whether
   progress occurred (CPU/GPU/network/disk when available);
5. continue only while there is evidence of progress or a documented expected
   phase;
6. if there is no new output or measurable progress for **two consecutive
   intervals**, stop waiting and diagnose;
7. if the operation exceeds a reasonable task-specific budget, report the
   exact command, elapsed time, last output, and remaining risk; use a safer
   alternative check where possible.

Do not conceal an unresponsive command behind a long wait.

### Tests

- Do not wait for real model downloads during ordinary unit tests; use fakes/
  mocks. Reserve real model/GPU work for explicit smoke commands such as
  `RUN_HYMT2_SMOKE=1` or documented smoke scripts.
- Do not rerun an unchanged failing or timed-out command repeatedly. After the
  first failure or timeout, inspect the error and choose a targeted fix or
  diagnostic.
- Do not delete coverage, split tests, add artificial markers, or refactor
  production code solely to reduce runtime that is already within the normal
  range. Performance work is required only for demonstrated regressions or
  operations that are actually stuck.

Repository commands (from `AGENTS.md`):

```powershell
conda run -n dp python -m pytest tests/ -v
```

Focused suites under `tests/translation_server/` typically complete in
seconds; treat ~17.4 seconds as acceptable for that module suite.

### Server and subprocesses

For `Start-TranslationServer.ps1` or any started server/child process:

- poll health/readiness at a short bounded interval;
- distinguish process alive but model loading, process exited, and no progress;
- stop immediately on a clear startup error;
- do not wait 10 minutes without checking logs and readiness;
- preserve existing watchdog and graceful-shutdown behavior;
- never leave an abandoned child process running after the task ends.

The startup script uses `python -m translation_server --print-config-summary`
so JSONC config and the selected model profile match what the server launches.

### Environment and model downloads

Before `Initialize-Env.ps1` or a model download, state whether the operation
may create/update an environment or download weights. During the operation,
inspect progress. If blocked on network, disk, permissions, package solving,
or GPU memory, report that concrete cause.

Do not rerun environment initialization or model download when a prior
successful result is already available. Reuse the existing `dp` environment
and configured cache; verify with a short import/version/cache check instead.

## One-round execution checklist

Before finishing a task:

1. perform the architecture/reuse gate;
2. implement one complete coherent slice;
3. run focused verification;
4. run the normal suite once when appropriate;
5. use bounded observation intervals for anything still running;
6. stop diagnosing when evidence is sufficient;
7. report exact results, elapsed time, and any blocker.

Do not ask the user for permission for ordinary bounded waits or normal
tests. Ask only when new authority, a destructive action, external
credentials, or a materially different scope is required.

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
- [ImageTranslation] the XS client (`script/TestTranslationApi.xs`) and
  other HTTP callers depend on `POST /translate` with JSON fields
  `text`, `format`, and `style`, and a JSON response field `translation`;
  backend/model selection must not change that contract.

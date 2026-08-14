# CLAUDE.md — Claude Code bridge

Read `AGENTS.md` first — it is the repository's primary agent instruction
(dependency rules, test commands, translation-server notes).

For architecture-sensitive work — adding a feature, fixing a bug,
refactoring, selecting a library, or changing services, parsers, model
loaders, clients, configuration, or infrastructure — read the canonical
architecture skill and its references before editing:

> Read `.agent-skills/reuse-first-architecture/SKILL.md` and follow it for
> this task.

It is mandatory workflow, not optional advice:

- search before build: reuse / adapt / extend / replace existing
  capabilities, naming the component and its contract;
- apply SOLID and dependency inversion (shared translation never imports
  FastAPI/server code; model/tokenizer resolution stays centralized);
- preserve public contracts, HTML/entity protection invariants, and
  cache/offline/GPU/precision/language-pair policies;
- implement one coherent slice, verify with the relevant tests, and
  report evidence — never claim a suite passed without running it.

Do not duplicate the skill text here; the canonical skill is the single
source of truth.

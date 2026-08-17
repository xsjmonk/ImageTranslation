# CLAUDE.md — Claude Code bridge

Read `AGENTS.md` first — it is the repository's primary agent instruction
(dependency rules, test commands, translation-server notes).

Before writing or editing any programming code, tests, scripts,
configuration, or infrastructure definition, read the canonical
architecture skill and follow its mandatory pre-coding gate:

> Read `.agent-skills/reuse-first-architecture/SKILL.md` and follow it for
> this task.

The pre-coding gate is mandatory before every programming edit, even when
the change appears small or trivial:

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

# docs/

Documentation for the Sunday Album project.

---

## Current reference docs

### `SYSTEM_ARCHITECTURE.md`
**The canonical reference.** Covers:
- High-level architecture (CLI, macOS app, web/AWS)
- Image processing pipeline — steps, execution model, storage abstraction
- Key technical decisions (why OpenAI for glare, why Hough rotation is disabled, etc.)
- AWS resource names, environments (dev/prod), CI/CD workflows
- API key resolution, authentication flow, WebSocket live progress
- Planned / not yet built

Read this first when orienting to the project or making architectural decisions.

### `PIPELINE_STEPS.md`
Per-step implementation reference. For each pipeline step:
- What it does and which algorithm/model it uses
- Tunable `PipelineConfig` parameters
- Debug output files it produces
- Known limitations and disabled paths

Read this when working on a specific pipeline step or tuning output quality.

---

## Journal

`../journal/` — development log. See `journal/INDEX.md` for a navigable list.
- Phase summaries (Phases 1–9): what was built, decisions made, test results
- Post-phase fixes: algorithm changes, bugs fixed, decisions revisited

---

## Archive

`archive/` — historical planning documents, all complete. See `archive/README.md` for the full list.
Kept for design rationale; not maintained.

# docs/archive/

Historical planning documents. All phases described here are complete. These files are kept for
reference (design rationale, implementation decisions) but are **not maintained**.

For current system state, see `docs/SYSTEM_ARCHITECTURE.md` and `docs/PIPELINE_STEPS.md`.

---

| File | What it was | Superseded by |
|------|-------------|---------------|
| `PRD_Album_Digitizer.md` | Original product requirements doc (Feb 2026, "Draft") | Product is live at app.sundayalbum.com |
| `Implementation_Album_Digitizer.md` | Original technical implementation plan (Feb 2026) | `docs/SYSTEM_ARCHITECTURE.md` |
| `UI_Design_Album_Digitizer.md` | UI design spec — "Warm Archival" design language, screen-by-screen layout (Feb 2026) | Implemented in `web/` (Tailwind tokens, components) |
| `PHASED_PLAN_Claude_Code.md` | Step-by-step Claude Code prompts used to build the CLI pipeline (Phases 1–9) | All phases complete; see `journal/` for summaries |
| `MACOS_APP_PLAN.md` | macOS SwiftUI app implementation plan | App is production; see `mac-app/` and `docs/SYSTEM_ARCHITECTURE.md` |
| `WEB_UI_PLAN_Part1.md` | Web UI phases 0–2: infrastructure, auth, pipeline Lambdas | Complete; see `docs/SYSTEM_ARCHITECTURE.md` |
| `WEB_UI_PLAN_Part2.md` | Web UI phase 3: real-time progress, library UI, dev.sundayalbum.com | Complete |
| `WEB_UI_PLAN_Part3.md` | Web UI phases 4–6: macOS UI parity, step detail, re-processing | Complete |
| `WEB_UI_PLAN_Part4.md` | Web UI phases 7–9: testing, admin tools, production hardening | Phases 7–7.6 complete; Phase 8 (admin) and Phase 9 (prod hardening) not yet built — see `docs/SYSTEM_ARCHITECTURE.md#planned--not-yet-built` |
| `DEVELOPMENT_JOURNAL.md` | Phase 6 summary + stubs for phases 1–5 | Superseded by `journal/` folder (full phase summaries there) |

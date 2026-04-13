# 2026-04-12 — Filling Reprocess UI Gaps: macOS App and Web UI

## Background

A survey of pipeline step parameters across the CLI, web UI, and macOS app revealed
that three of the four interactive step views in the macOS app are incomplete, and that
the web UI has a parameter naming mismatch in the color restoration reprocess call.
This entry documents what's missing, the plan to close the gaps, what was actually
implemented, and the follow-on work on app state persistence.

---

## Gap Inventory

### Gap 1 — Web UI: `saturation_boost` naming mismatch (color restore)

`ColorRestoreView` in `jobs/[jobId]/page.tsx` posts `saturation_boost` to the
`/reprocess` endpoint, but `PipelineConfig` has no such field — the actual param is
`color_restore_vibrance_boost`. `sharpen_amount` is named correctly. The Lambda
`_handle_reprocess` passes `config_overrides` straight to Step Functions as-is, so the
key is silently dropped and the saturation slider has no effect when reprocessing.

Fix: rename the key in the POST payload from `saturation_boost` to
`color_restore_vibrance_boost`, and adjust the slider range to match the config range
(0.0–0.5 stored as 0%–50%).

### Gap 2 — macOS: Orientation "Apply & Reprocess" not wired to CLI

`OrientationStepView.swift` has a full rotation picker (0° / 90° / 180° / 270°) and an
"Apply & Reprocess" button. Pressing the button sets `photo.rotationOverride` and
`photo.sceneDescription` on the model — but no CLI subprocess is triggered.
`PipelineRunner` has no orientation reprocess entry point.

Fix: add `PipelineRunner.reprocessFromOrientation(rotationDegrees:sceneDescription:)`
that builds `--steps ai_orientation,...` with `--forced-rotation <degrees>` and connect
it to the "Apply & Reprocess" button action.

### Gap 3 — macOS: Glare removal has scene description input but no reprocess button

`GlareRemovalStepView.swift` shows the static OpenAI prompt template and a `TextField`
for scene description override — but has no "Re-run Glare Removal" button.

Fix: add a footer action row with "Discard" and "Re-run Glare Removal" buttons.
Add `PipelineRunner.reprocessFromGlare(sceneDescription:)`.

### Gap 4 — macOS: Color correction "Apply & Reprocess" not wired to CLI

`ColorCorrectionStepView.swift` has Saturation and Sharpness `InlineSlider` controls
and an "Apply & Reprocess" button with an empty `// TODO` action body.

Fix: add `PipelineRunner.reprocessFromColor(vibranceBoost:sharpenAmount:)` building
`--steps white_balance,color_restore,deyellow,sharpen` with `--color-vibrance` and
`--sharpen-amount` flags. Requires two new CLI flags.

---

## Scope of This Work

The four gaps above are the full scope. Steps that have no UI at all
(`page_detect`, `perspective`, `geometry`, `normalize`) are deliberately excluded —
their parameters are tuning knobs for edge cases, not things a typical user would
adjust. They remain CLI-only.

---

## Design

### New CLI flags (required by Gap 4)

Add to `src/cli.py` `process` command:

```
--color-vibrance FLOAT   color_restore_vibrance_boost (default: 0.25)
--sharpen-amount FLOAT   sharpen_amount (default: 0.5)
--forced-rotation INT    forced_rotation_degrees: apply this exact rotation (0/90/180/270),
                         skipping the Claude API call (already in PipelineConfig,
                         not yet in CLI)
```

### `PipelineRunner.swift` — three new entry points + testable static methods

Each entry point is backed by a `nonisolated static` method (e.g. `_colorArgs(...)`)
so the arg-building logic can be unit-tested without launching a subprocess.

```swift
func reprocessFromOrientation(rotationDegrees: Int, sceneDescription: String)
func reprocessFromGlare(sceneDescription: String)
func reprocessFromColor(vibranceBoost: Double, sharpenAmount: Double)
```

`--steps` lists:

| Entry point | `--steps` value |
|---|---|
| `reprocessFromOrientation` | `ai_orientation,glare_detect,...,sharpen` |
| `reprocessFromGlare` | `glare_detect,...,sharpen` |
| `reprocessFromColor` | `color_restore,...,sharpen` |

### Web UI: fix `saturation_boost` key (Gap 1)

In `jobs/[jobId]/page.tsx`:
```ts
// Before
config: { saturation_boost: saturation / 100, sharpen_amount: sharpness / 100 }
// After
config: { color_restore_vibrance_boost: saturation / 100, sharpen_amount: sharpness / 100 }
```

---

## Implementation — What Was Built

All four gaps were implemented and shipped on `feature/reprocess-ui-gaps`.

### Files changed

| File | Change |
|------|--------|
| `src/cli.py` | Added `--forced-rotation`, `--color-vibrance`, `--sharpen-amount`; wired to `PipelineConfig` |
| `web/src/app/(app)/jobs/[jobId]/page.tsx` | Renamed `saturation_boost` → `color_restore_vibrance_boost`; added per-photo reprocess running indicator scoped to the selected photo only |
| `mac-app/.../PipelineRunner.swift` | Added three reprocess entry points + static `_*Args` methods; `UITEST_MODE` simulation |
| `mac-app/.../OrientationStepView.swift` | Wired "Apply & Reprocess"; added scene description `TextField`; accessibility identifiers |
| `mac-app/.../GlareRemovalStepView.swift` | Added "Discard" + "Re-run Glare Removal" footer; `isProcessing` state; accessibility identifiers |
| `mac-app/.../ColorCorrectionStepView.swift` | Wired "Apply & Reprocess"; `isProcessing` state; accessibility identifiers |
| `mac-app/.../AlbumPageCard.swift` | Added `job-card-<inputName>` accessibility identifier |
| `mac-app/.../StepDetailView.swift` | Added `tree-row-<label>` accessibility identifier |
| `mac-app/project.yml` | Added `SundayAlbumUITests` target; added `Stamp BuildInfo` preBuildScript |
| `mac-app/.gitignore` | Gitignored `BuildInfo.swift` — generated fresh on every build with PST timestamp |
| `mac-app/.../PipelineStep.swift` | Fixed `debugImageURL`: flat file layout (`{baseName}_{suffix}`) not subdirectories; `.done` → `14_photo_NN_enhanced.jpg` |
| `mac-app/.../MockData.swift` | Replaced hardcoded 4-job list with dynamic debug folder scan (same flat-file logic) |
| `mac-app/.../CLIOutputParser.swift` | Fixed `photo_detect:` → `photo_split:` |
| `mac-app/.../CLIOutputParserTests.swift` | Corrected 8 tests that used invented log lines; updated to match actual `pipeline.py` output |

### AWS Step Functions fix (prerequisite)

Before the macOS work, a CDK bug was found: `PrepareMap` in the state machine was not
forwarding `start_from` and `reprocess_photo_index` to the Lambda handlers, so partial
reprocessing always ran the full pipeline on all photos. Fixed by patching the state
machine definition directly via `aws stepfunctions update-state-machine`.

---

## Automated Testing

Two test suites added to `mac-app/`:

### `SundayAlbumTests/PipelineRunnerArgsTests.swift` — 20 Swift Testing unit tests

Exercises the static `_*Args` methods without launching a subprocess:
- Color: correct steps, no pre-split steps, `--color-vibrance` present (regression guard: `--saturation-boost` must never appear), `--sharpen-amount`, 3-decimal formatting
- Glare: correct steps, scene-desc conditional on non-empty string
- Orientation: correct steps, no photo_detect/photo_split, rotation flag omitted at 0°, correct values at 90°/270°
- Photo split: correct steps including all downstream steps

### `SundayAlbumUITests/ReprocessUITests.swift` — 12 XCUITest UI tests

Launches the app with `MOCK_DATA=1` + `UITEST_MODE=1` (Python subprocess skipped;
job state transitions in-process):

**T-mac-01 Color Correction (4 tests)**
- Saturation slider is present
- "Apply & Reprocess" button hidden when sliders at defaults
- Button appears after moving slider
- Clicking button triggers processing state (button disappears)

**T-mac-02 Glare Removal (3 tests)**
- "Re-run" and "Discard" buttons are always present
- Scene description field is present
- Clicking "Re-run" disables the button

**T-mac-03 Orientation (5 tests)**
- All four rotation buttons (0°/90°/180°/270°) are present
- Scene description field is present
- "Apply & Reprocess" hidden at default 0° rotation
- Button appears after selecting non-zero rotation
- Clicking button triggers processing state

**Total test counts (all passing):**
- Unit: 45/45 (`PipelineRunnerArgsTests` 20, `CLIOutputParserTests` 16, `FileImporter` 9)
- UI: 12/12

### Testing notes

Tree-row labels use `PipelineStep.title` (short form): `"Glare"` not `"Glare Removal"`,
`"Orient"` not `"Orientation"`. Accessibility element lookup uses
`descendants(matching: .any)` because SwiftUI VStack/HStack containers register as
`.group` in the accessibility tree, not `.other`.

---

## AppState Persistence — Problem Found, Plan Decided

### Problem

`AppState.jobs` is an in-memory array with no serialization. Every app launch starts
from a blank slate. The only way jobs appear in the library is via `DebugFolderScanner`,
which was broken in two ways:

1. **Wrong path in production**: used `RuntimeManager.cliWorkingDirectory + "debug"`,
   which resolves to the app bundle's `Contents/Resources/debug/` — a directory that
   doesn't exist. Should use `AppSettings.debugFolder`.

2. **Wrong file layout**: expected subdirectories (`debug/IMG_cave_normal/01_loaded.jpg`)
   but the CLI writes flat files (`debug/IMG_cave_normal_01_loaded.jpg`).

### Decision: Option C — filesystem as source of truth + overrides JSON

The pipeline already writes all meaningful artifacts to disk (debug images, output
images). The only in-memory state that can't be re-derived from the filesystem is user
overrides: `rotationOverride` and `sceneDescription` on each `ExtractedPhoto`.

**Architecture:**
1. On startup, `DebugFolderScanner` scans `AppSettings.debugFolder` (flat-file aware,
   correct for both dev and production) and reconstructs one `ProcessingJob` per
   processed input. State is inferred from which step output files exist on disk.
2. A small `overrides.json` in `~/Library/Application Support/SundayAlbum/` stores only
   user-set values keyed by `(inputName, photoIndex)`. Applied after scan.
3. `AppSettings.debugFolder` defaults to `devProjectRoot/debug/` in dev builds and
   `~/Library/Application Support/SundayAlbum/debug/` in production builds.
4. The debug folder setting is exposed in the Settings screen so users (and developers)
   can point it at any folder. Changing the setting triggers a full library reload.

**Why not full serialization (Option A / SwiftData)?**
The schema is still evolving and the data model is tightly coupled to on-disk artifacts.
Option C gives robust reinstall behaviour without carrying a full serialization layer.
Full JSON/SwiftData serialization can be layered on top later once the schema stabilises.

### Files to change (next)

| File | Change |
|------|--------|
| `mac-app/.../DebugFolderScanner.swift` | Use `AppSettings.debugFolder`; flat-file layout; infer step from existing files (not just complete jobs) |
| `mac-app/.../AppSettings.swift` | Default `debugFolder` to `devProjectRoot/debug/` in dev builds |
| `mac-app/.../AppState.swift` | On `debugFolder` change, re-run scanner and replace `jobs` |
| `mac-app/.../SettingsView.swift` | Add "Debug folder" folder-picker row with reload-on-change behaviour |
| `mac-app/.../overrides.json` | New file in Application Support; written on `rotationOverride`/`sceneDescription` change; read on startup after scan |

---

## AppState Persistence — Implemented (Option C)

All five planned files were changed in the follow-on session:

| File | What changed |
|------|-------------|
| `Models/ExtractedPhoto.swift` | Added `photoIndex: Int` (1-based, default 1); used as overrides key |
| `Settings/AppSettings.swift` | `defaultDebugFolder` now returns `devProjectRoot/debug/` in dev builds |
| `Utils/DebugFolderScanner.swift` | Full rewrite: uses `AppSettings.shared.debugFolder`, flat-file layout, infers job state from which step files exist, sets `photoIndex` on each `ExtractedPhoto` |
| `Utils/OverridesStore.swift` | **New file**: reads/writes `~/Library/Application Support/SundayAlbum/overrides.json`; key format `"inputName:photoIndex"`; `apply(to:)` called on startup; `update(for:)` called when user commits rotation or scene-desc |
| `AppState.swift` | Added `reloadJobs()` (rescan + apply overrides + navigate to .library); added `saveOverrides(for:)`; `init` now calls `OverridesStore.apply` after scan |
| `Settings/SettingsView.swift` | Debug folder row always visible (not gated on `debugOutputEnabled`), with footnote; `FolderPickerRow` gained optional `onChange` callback; callback calls `appState.reloadJobs()` |
| `SundayAlbumApp.swift` | Settings scene now injects `appState` into environment so SettingsView can call `reloadJobs()` |
| `Views/Steps/OrientationStepView.swift` | "Apply & Reprocess" calls `appState.saveOverrides(for: photo)` after updating the model |
| `Views/Steps/GlareRemovalStepView.swift` | "Re-run" writes `sceneDesc` back to `photo.sceneDescription` and calls `appState.saveOverrides(for: p)` |
| `Mock/MockData.swift` | `withMockData()` now calls `AppState(loadDebugJobs: false)` to prevent double-loading; passes `photoIndex: i` to `ExtractedPhoto` |

**Key bug caught during testing:** `withMockData()` was calling `AppState()` (which
now correctly scans the debug folder), then appending mock jobs on top — resulting in
duplicate job cards. Switching to `AppState(loadDebugJobs: false)` fixed it.

---

## Test Runner Rules Added to CLAUDE.md

macOS UI tests (`SundayAlbumUITests`) hijack the screen for ~2 minutes and should
**never be triggered without explicit user request**. Documented in `CLAUDE.md`:

```bash
# Safe — unit tests only, no screen takeover
xcodebuild test -scheme SundayAlbum -destination 'platform=macOS' \
  -skip-testing:SundayAlbumUITests

# Disruptive — only on explicit request
xcodebuild test -scheme SundayAlbum -destination 'platform=macOS' \
  -only-testing:SundayAlbumUITests
```

Rule: never use bare `xcodebuild test -scheme SundayAlbum` without a filter flag.

---

## Manual Verification — Completed

All three reprocess flows verified manually against a real image:

- **Color correction**: moved saturation/sharpness sliders → clicked "Apply & Reprocess" → CLI ran with correct `--color-vibrance` and `--sharpen-amount` flags ✓
- **Glare removal**: typed scene description → clicked "Re-run Glare Removal" → CLI ran with `--scene-desc` passed correctly ✓
- **Orientation**: selected 90° rotation → clicked "Apply & Reprocess" → CLI ran with `--forced-rotation 90` ✓

All gaps from the original gap inventory are fully closed.

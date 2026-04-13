# 2026-04-12 — Filling Reprocess UI Gaps: macOS App and Web UI

## Background

A survey of pipeline step parameters across the CLI, web UI, and macOS app revealed
that three of the four interactive step views in the macOS app are incomplete, and that
the web UI has a parameter naming mismatch in the color restoration reprocess call.
This entry documents what's missing and plans the work to close the gaps.

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

Fix: add `PipelineRunner.reprocessFromOrientation(stem:photoIndex:rotationDegrees:)`
that builds:
```
--steps ai_orientation,glare_detect,...
--forced-rotation <degrees>
```
and connect it to the "Apply & Reprocess" button action.

### Gap 3 — macOS: Glare removal has scene description input but no reprocess button

`GlareRemovalStepView.swift` shows the static OpenAI prompt template and a
`TextField` for scene description override — matching the web UI — but has no
"Re-run Glare Removal" button. The `sceneDesc` state is populated and even pre-filled
from `photo.sceneDescription`, but there is no way to submit it.

Fix: add a footer action row (matching the pattern in `OrientationStepView` and the
web's `GlareRemovalView`) with a "Discard" and a "Re-run Glare Removal" button.
Add `PipelineRunner.reprocessFromGlare(stem:photoIndex:sceneDescription:)` that
builds:
```
--steps glare_detect,...
--scene-desc "<text>"
```

### Gap 4 — macOS: Color correction "Apply & Reprocess" not wired to CLI

`ColorCorrectionStepView.swift` has Saturation and Sharpness `InlineSlider` controls
and an "Apply & Reprocess" button, with a `// TODO: wire to reprocess API when CLI
bridge supports it` comment. The button action body is empty.

Fix: add `PipelineRunner.reprocessFromColor(stem:photoIndex:vibranceBoost:sharpenAmount:)`
that builds:
```
--steps color_restore
--color-vibrance <v>
--sharpen-amount <s>
```
This requires two new CLI flags: `--color-vibrance` and `--sharpen-amount`.

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

`--forced-rotation` is also needed for Gap 2 and has been documented as a "hidden gem"
in the parameter survey. Add it here so both macOS gaps can use it.

Wire all three into `PipelineConfig` construction in the `process()` command handler.

### `PipelineRunner.swift` — three new entry points

```swift
func reprocessFromOrientation(stem: String, photoIndex: Int, rotationDegrees: Int)
func reprocessFromGlare(stem: String, photoIndex: Int, sceneDescription: String)
func reprocessFromColor(stem: String, photoIndex: Int, vibranceBoost: Double, sharpenAmount: Double)
```

Each builds the correct `--steps` list (covering from the named step through
`color_restore`) and the relevant override flags, then calls the existing
`runPipeline(arguments:)` path. `photo_index` maps to the existing `--photo-index`
mechanism (or equivalent step filter targeting) if per-photo step selection is
supported; otherwise runs all photos from that step forward (acceptable for now).

**`--steps` lists for each entry point:**

| Entry point | `--steps` value |
|---|---|
| `reprocessFromOrientation` | `ai_orientation,glare_detect,keystone_correct,dewarp,rotation_correct,white_balance,color_restore,deyellow,sharpen` |
| `reprocessFromGlare` | `glare_detect,keystone_correct,dewarp,rotation_correct,white_balance,color_restore,deyellow,sharpen` |
| `reprocessFromColor` | `white_balance,color_restore,deyellow,sharpen` |

### `OrientationStepView.swift` — wire button

Replace the `Apply & Reprocess` button action with:
```swift
appState.pipelineRunner.reprocessFromOrientation(
    stem: stem,
    photoIndex: photoIndex + 1,
    rotationDegrees: pendingRotation
)
```
Update `isDirty` to include `pendingDescription` so typing a scene description (the
existing but invisible `pendingDescription` state) also enables the button. Since the
text field for `pendingDescription` is not rendered in the current UI, add it to the
footer — matching `GlareRemovalStepView`'s layout (label + `TextField`). This gives
the user a single place to set both rotation and scene description before reprocessing.

### `GlareRemovalStepView.swift` — add action row

Add an `HStack` footer row beneath the existing prompt display:
```
[Discard]  [Re-run Glare Removal]
```
"Re-run" calls `reprocessFromGlare(stem:photoIndex:sceneDescription:sceneDesc)`.
"Discard" resets `sceneDesc` to `photo?.sceneDescription ?? ""`.
The button is always enabled (reprocessing with no scene override is valid — it just
uses Claude's auto-detected description).

### `ColorCorrectionStepView.swift` — wire button

Replace the empty `// TODO` action with:
```swift
appState.pipelineRunner.reprocessFromColor(
    stem: stem,
    photoIndex: photoIndex + 1,
    vibranceBoost: saturation,   // 0.0–0.5
    sharpenAmount: sharpness     // 0.0–1.0
)
```

### Web UI: fix `saturation_boost` key (Gap 1)

In `jobs/[jobId]/page.tsx`, `ColorRestoreView.handleApply()`:
```ts
// Before
config: { saturation_boost: saturation / 100, sharpen_amount: sharpness / 100 }

// After
config: { color_restore_vibrance_boost: saturation / 100, sharpen_amount: sharpness / 100 }
```

No other web changes are needed — the other three web reprocess flows (orientation,
glare, photo split) are already correctly wired.

---

## Files Changed

| File | Change |
|------|--------|
| `src/cli.py` | Add `--forced-rotation`, `--color-vibrance`, `--sharpen-amount` flags; wire to `PipelineConfig` |
| `web/src/app/(app)/jobs/[jobId]/page.tsx` | Rename `saturation_boost` → `color_restore_vibrance_boost` in `ColorRestoreView.handleApply()` |
| `mac-app/.../PipelineRunner.swift` | Add `reprocessFromOrientation()`, `reprocessFromGlare()`, `reprocessFromColor()` |
| `mac-app/.../OrientationStepView.swift` | Wire "Apply & Reprocess" button; add scene description `TextField` to footer |
| `mac-app/.../GlareRemovalStepView.swift` | Add "Discard" + "Re-run Glare Removal" footer action row |
| `mac-app/.../ColorCorrectionStepView.swift` | Wire "Apply & Reprocess" button action |

---

## Implementation Order

1. **CLI flags** (`src/cli.py`) — no UI, easy to test, unblocks everything else.
2. **Web rename** (`jobs/[jobId]/page.tsx`) — one-line fix, ship immediately.
3. **`PipelineRunner` entry points** — shared foundation for the three macOS views.
4. **`ColorCorrectionStepView`** — simplest wiring (no new UI, just call the runner).
5. **`GlareRemovalStepView`** — adds action row UI.
6. **`OrientationStepView`** — most involved (adds text field + wires two params).

---

## Testing

### CLI

```bash
source .venv/bin/activate

# --forced-rotation: should apply 90° CW without making a Claude API call
python -m src.cli process test-images/IMG_harbor_normal.HEIC --output ./output/ --debug \
  --steps ai_orientation,glare_detect,white_balance,color_restore,deyellow,sharpen \
  --forced-rotation 90

# --color-vibrance + --sharpen-amount: re-run only color steps with custom values
python -m src.cli process test-images/IMG_harbor_normal.HEIC --output ./output/ --debug \
  --steps white_balance,color_restore,deyellow,sharpen \
  --color-vibrance 0.4 --sharpen-amount 0.8

# Confirm vibrance and sharpness are visibly different from defaults
```

### Web UI

1. Process any image through the full pipeline.
2. Open Color Restore step, move Saturation slider to 50%.
3. Click "Apply & Reprocess" — confirm job goes to `processing` and completes.
4. Verify output is visibly more saturated (previously the slider was a no-op).

### macOS App

1. Process a job fully so all step views are populated.
2. **Color correction:** move Saturation slider, click "Apply & Reprocess" — confirm
   pipeline re-runs from `white_balance` onward.
3. **Glare removal:** type a scene description, click "Re-run Glare Removal" — confirm
   CLI is called with `--scene-desc`.
4. **Orientation:** select 90°, click "Apply & Reprocess" — confirm CLI is called with
   `--forced-rotation 90` and the oriented image updates.

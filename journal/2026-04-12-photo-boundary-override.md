# 2026-04-12 — Photo Boundary Override: Let Users Correct Detection and Reprocess

## Problem Statement

The contour-based photo detector (`photo_detect.py`) occasionally misses a photo, merges two
adjacent photos into one detection, or clips a detection too tightly against the album border.
When this happens there is currently no way for the user to fix it — they must accept the wrong
split or reprocess the entire pipeline from scratch.

Additionally, `photo_split.py` has a latent bug: it reads the stored
`_05_photo_detections.json` (which contains the bboxes) but then discards it and **re-runs
`detect_photos()` on the page image**, producing a fresh (and potentially different) result.
This makes the step non-idempotent and breaks any future manual-override workflow.

## Goal

Let the user:
1. View the detected photo regions overlaid on the page image.
2. Drag regions to adjust position/size, add missing regions, or delete wrong ones.
3. Confirm → pipeline re-runs from `photo_split` onward using the corrected boundaries.

Both the macOS app and CLI must support this.

---

## Design

### Backend — `PipelineConfig.forced_detections`

Add one new optional field:

```python
forced_detections: Optional[list[dict]] = None
```

Each dict matches the stored detection format:
```json
{ "bbox": [x1, y1, x2, y2], "confidence": 1.0, "region_type": "photo", "orientation": "unknown" }
```

### `photo_detect.py` — short-circuit when forced

If `config.forced_detections` is set:
- Write those dicts directly to `debug/{stem}_05_photo_detections.json`
- Return immediately without running the contour detector
- Write the boundaries overlay image using the forced bboxes

### `photo_split.py` — fix re-detection bug

Replace the erroneous `detect_photos()` call with a reconstruction of `PhotoDetection` objects
directly from the stored `det_dicts`. Corners are derived from the bbox rectangle — good enough
for the per-photo perspective warp that `split_photos` applies.

When `forced_detections` are in play, `photo_detect` has already written the right JSON, so
`photo_split` picks them up transparently with no further changes.

### `pipeline.py` — `photo_split` as independent steps_filter entry

`photo_detect` and `photo_split` are currently coupled under the same `should_run_photo`
guard. Add `photo_split` as a separate filter token so the CLI can run just
`--steps photo_split,...` (uses the existing JSON without re-detecting).

### CLI — `--forced-detections` flag

```bash
# Run from photo_split onward using corrected boundaries
python -m src.cli process image.HEIC --output ./output/ --debug \
  --forced-detections '[{"bbox":[50,80,620,1100]}]' \
  --steps photo_detect,ai_orientation,glare_detect,...
```

`photo_detect` sees `config.forced_detections`, writes them to JSON, skips contour detection.
`photo_split` reads the JSON and extracts using the overridden bboxes.

### macOS app — interactive `PhotoSplitStepView`

**Loading**: on appear, read `{debugFolder}/{stem}_05_photo_detections.json` using
`FileManager`. Fall back gracefully when the file is missing (debug output not enabled).

**Interaction**:
- Each region has a drag gesture to **move** and corner/edge handles to **resize**.
- "Add Region" enters draw mode (drag to create new bbox).
- "Delete" button appears on each selected region.
- Regions are stored as normalized `CGRect` (0–1) converted from/to pixel bboxes at load/save.

**Confirm & Re-run**:
1. Serialize regions back to the `_05_photo_detections.json` format.
2. Write the file to `{debugFolder}/{stem}_05_photo_detections.json` via `FileManager`.
3. Trigger a new pipeline run with `--steps photo_split,ai_orientation,glare_detect,...`
   (skips load, normalize, page_detect, photo_detect — all earlier steps).

`PipelineRunner` gets a new `func reprocessFromPhotoSplit(stem:detections:)` entry point that
builds the correct argument list and restarts the subprocess.

---

## Files Changed

| File | Change |
|------|--------|
| `src/pipeline.py` | `PipelineConfig.forced_detections`; decouple `photo_split` in steps_filter |
| `src/steps/photo_detect.py` | Short-circuit on `config.forced_detections` |
| `src/steps/photo_split.py` | Fix re-detection bug; use stored bboxes |
| `src/cli.py` | `--forced-detections` flag |
| `mac-app/.../PhotoSplitStepView.swift` | Full implementation (interactive UI + re-run) |
| `mac-app/.../PipelineRunner.swift` | `reprocessFromPhotoSplit()` entry point |

---

## Testing

### CLI (local, no macOS app required)

```bash
source .venv/bin/activate

# 1. Full run to generate debug JSON
python -m src.cli process test-images/IMG_three_pics_normal.HEIC \
  --output ./output/ --debug

# 2. Confirm _05_photo_detections.json was written
cat debug/IMG_three_pics_normal_05_photo_detections.json

# 3. Re-run from photo_split using stored JSON (no --forced-detections)
python -m src.cli process test-images/IMG_three_pics_normal.HEIC \
  --output ./output/ --debug --steps photo_split,ai_orientation,glare_detect,...

# 4. Override one bbox and re-run
python -m src.cli process test-images/IMG_three_pics_normal.HEIC \
  --output ./output/ --debug \
  --forced-detections '[{"bbox":[50,80,620,1100],"confidence":1.0,"region_type":"photo","orientation":"portrait"}]' \
  --steps photo_detect,ai_orientation,glare_detect,keystone_correct,dewarp,rotation_correct,white_balance,color_restore,deyellow,sharpen
```

### macOS app

Manual: open a previously-processed image, navigate to Photo Split step, drag a boundary,
confirm, and verify the pipeline reruns from photo_split with the corrected crop.

---

## Implementation Results (2026-04-12)

### What was shipped

**Photo boundary override — fully working** (macOS app):
- `PhotoSplitStepView` completely rewritten. Previously used `.offset()` for handles which moves
  visual rendering but not hit areas, making handles unresponsive. Now mirrors
  `PageDetectionStepView` exactly: canvas-sized ZStack, `.position()` for each `RectCornerHandle`,
  `DragGesture(minimumDistance: 0)` with `v.location` (canvas-space coords). Each region shows
  four draggable amber corner handles that correctly resize the bounding box.
- Fixed Swift string literal with curly quotes that caused build failure.
- Confirmed by user: "bounding areas worked much better this time."

**Progress bar fix**:
- `CLIOutputParser.swift`: all step prefix strings were wrong (e.g. `"Load time:"` instead of
  `"load:"`). Updated to match actual pipeline log format.
- `pipeline.py`: per-photo steps (ai_orient, glare, geometry, color_restore) had no log lines at
  all. Added `logger.info()` calls for each after `photo_idx == 1` completes each step.
- Confirmed by user: "The progress bar also worked."

---

## Follow-on Requirements (2026-04-12)

### Req 1 — Per-step active animation in the step tree

When a step is actively executing, the step tree row should give a visible animated cue (pulsing
ring / ripple). When the step completes it becomes a solid green checkmark. Currently there is
only a small static amber dot for the active step, which is easy to miss.

Design:
- Add `isRunning: Bool` to `TreeRow`.
- When `isRunning`, show an expanding/fading `PulsingRing` behind the icon.
- Pass `isRunning: job.state == .running && step == job.currentStep` at each `TreeRow` call site
  in `StepTree` and `PhotoBranchGroup`.

### Req 2 — In-step progress overlay during reprocessing

When the user is already inside a step detail view (e.g. `PhotoSplitStepView`) and triggers
reprocessing, no progress feedback is shown. The step canvas stays static while the pipeline
re-executes behind the scenes. This is especially confusing when the photo count changes (1→N or
N→1) because the tree structure also changes.

Design:
- Show a `ProcessingProgressBanner` overlay anchored to the bottom of the step canvas whenever
  `job.state == .running`.
- Banner contains: spinner, "Processing step X of N: [step name]" label, amber progress bar,
  percentage.
- Appears/disappears with a slide+fade transition.
- Works identically for initial processing and reprocessing — one consistent UI path.

---

## Web UI Parity Plan (2026-04-12)

Port all macOS UX improvements to the Next.js web frontend on the same branch.
Implementation order: lowest-risk first, photo boundary override last.

### Priority 1 — Library card: next-step segment pulse
**File:** `web/src/components/library/ProgressWheel.tsx`

`ProgressWheel` renders 6 SVG pie segments. Add a `PulsingSegment` sub-component: while
`job.status === 'processing'`, the segment at index `BACKEND_TO_VISUAL[job.current_step]`
(= `completedCount`) gets a CSS `@keyframes` fill animation cycling between amber and stone.
All completed segments stay solid amber; future segments stay stone.
Mirrors macOS `PulsingPieSegment`.

### Priority 2 — Job detail: active step pulsing in sidebar
**File:** `web/src/app/(app)/jobs/[jobId]/page.tsx` (StepTree rows)

The step row for the currently-active step (matched by backend step key while
`job.status === 'processing'`) shows a ripple ring behind its icon using the same CSS
`@keyframes` as the library card. Thread an `isRunning` boolean into each row.

### Priority 3 — Job detail: processing progress banner
**File:** `web/src/app/(app)/jobs/[jobId]/page.tsx` (canvas area)

Whenever `job.status === 'processing'`, show a fixed strip at the bottom of the canvas with:
spinner · "Step X of 6: [step_detail]" · amber progress bar · percentage.
Slides in/out with CSS transition. Covers both initial processing and reprocessing.
Mirrors macOS `ProcessingProgressBanner`.

### Priority 4 — Results view: photo index
**File:** `web/src/components/step-detail/ResultsView.tsx` (or wherever the ResultsView is)

Verify the component receives the selected photo index from the step tree and opens on the
correct photo. If `selectedIndex` is always initialised to 0, fix it to use the photo index
passed from the parent — same as macOS `ResultsStepView` fix.

### Priority 5 — Photo boundary override (interactive)
**Files:** new `PhotoSplitView.tsx` · `web/src/lib/api.ts` · Lambda handler

Backend: expose `_05_photo_detections.json` URL in `debug_urls` map. Handler already writes it;
just needs the key added to the presigned-URL generation. Accept `forced_detections` in the
reprocess config and pass `--forced-detections` to the CLI.

Frontend:
- Load detection JSON from `debug_urls['05_photo_detections']`
- Render as SVG rects over the debug image
- 4 corner `<circle>` handles with `onMouseDown` drag (same geometry as macOS `RectCornerHandle`)
- "Add Region" → click-drag to draw new bbox
- Delete button per region
- "Confirm & Re-run" → `reprocessJob(jobId, { from_step: 'photo_detect', config: { forced_detections } })`
- Swap `DebugImageCanvas` for `PhotoSplitView` in `photo_split` step selection

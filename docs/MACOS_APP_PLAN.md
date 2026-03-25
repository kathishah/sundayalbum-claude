# Sunday Album — macOS Native App Plan

## Context

The Python CLI image processing pipeline is complete (phases 1–9, all 14 steps implemented). This document plans the next milestone: a native macOS app that gives users a full-featured GUI for the pipeline. All image processing continues to run in Python — the Mac app is a SwiftUI shell that orchestrates the CLI, displays results, and integrates with macOS.

**Tech decision: SwiftUI native app** (not Electron). Rationale: full macOS integration (Photos.app, Finder, drag-and-drop, macOS animations), lightweight app bundle (~5 MB vs ~300 MB for Electron), best performance on Apple Silicon M4.

---

## Design Language

The Mac app inherits the "Warm Archival" aesthetic from the product design system (`docs/UI_Design_Album_Digitizer.md`).

### Color Palette

```
Light Mode (review/editing):
  Background:           #FFF8F0   warm white, not stark
  Card fills:           #FEF0DC
  Borders:              #E7E5E4
  Primary text:         #44403C
  Secondary text:       #78716C
  Disabled/placeholder: #A8A29E
  Primary accent:       #D97706   warm amber
  Accent hover:         #B45309

Dark Mode (webcam capture):
  Background:           #0C0A09
  Surface/panels:       #1C1917
  Borders:              #292524
  Primary text:         #F5F5F4
  Secondary text:       #A8A29E

Status Colors:
  Success:              #16A34A   processing complete, good confidence
  Warning:              #EA580C   low confidence, attention needed
  Error:                #DC2626   processing error
  Glare overlay:        rgba(234, 88, 12, 0.4)
  Page detect green:    rgba(22, 163, 74, 0.6)
  Page detect red:      rgba(220, 38, 38, 0.6)
```

### Typography

```
Display/headings:   Fraunces (serif)           emotional moments, app name
Body/UI labels:     DM Sans (sans-serif)        all controls and text
Filenames/counters: JetBrains Mono (monospace)

Type scale:
  H1:      32px / 1.2   Fraunces 500    page titles
  H2:      24px / 1.3   Fraunces 500    section headings
  H3:      18px / 1.4   DM Sans 600     subsection headings
  Body:    16px / 1.5   DM Sans 400     body text
  Small:   14px / 1.5   DM Sans 400     secondary text, captions
  Caption: 12px / 1.4   DM Sans 500     labels, metadata
  Mono:    13px / 1.4   JetBrains 400   file names, counters
```

### Spacing & Shape

```
Spacing scale (pt): 4, 8, 12, 16, 20, 24, 32, 40, 48, 64

Border radii:
  6pt   buttons, inputs, small chips
  10pt  cards, panels
  16pt  modals, large containers
  24pt  hero cards, featured elements

Shadows:
  Card:   0 2px 8px rgba(28,25,23,0.12), 0 0 0 1px rgba(28,25,23,0.04)
  Float:  0 4px 6px -1px rgba(28,25,23,0.07), 0 2px 4px -2px rgba(28,25,23,0.05)
  Glow:   0 0 20px rgba(217,119,6,0.15)   selected/active states
```

### Motion & Animation

**Easing functions (SwiftUI equivalents):**
```
ease-out-expo  → .timingCurve(0.16, 1, 0.3, 1)    primary ease, smooth deceleration
ease-in-out    → .timingCurve(0.45, 0, 0.55, 1)    symmetric transitions
ease-spring    → .spring(response: 0.4, dampingFraction: 0.6)  playful overshoot

Duration scale:
  120ms   hover states, micro-interactions
  200ms   standard transitions
  350ms   panel slides, reveals
  600ms   before/after glare reveal (signature)
  500ms   view transitions
```

**Key animations to implement:**

1. **Glare reveal** (signature, 600ms): When a processed photo is first selected in the comparison view, the "after" image fades in over 600ms with ease-out-expo. Accompanied by a subtle warm amber glow pulse on the photo border (`rgba(217,119,6,0.15)` shadow expanding outward). This is the "wow moment" the product is designed around.

2. **Processing shimmer**: While a job is `.running`, a soft amber shimmer sweeps left-to-right across the queue row thumbnail (2s loop). On `.complete`, the shimmer resolves into a green checkmark with spring scale-up animation.

3. **Queue entry stagger**: When multiple files are dropped, queue rows stagger in with 60ms delay between each, sliding up + fading in (ease-out-expo, 350ms).

4. **Photo card hover**: On hover, photo thumbnail cards lift with `translate-y -2pt` and `shadow-md`. A semi-transparent action overlay (12pt buttons: "Open", "Finder", "Add to Photos") slides up from the bottom edge (200ms, ease-out-expo).

5. **Photo card selected**: Amber 2pt border + glow shadow replaces the default stone border. Transition: 120ms.

**Component specs (for SwiftUI implementation):**

```
Primary Button:
  Background: amber-500 (#D97706)
  Text: white, 16pt, weight .semibold
  Padding: 12×24pt
  Corner radius: 6pt
  Hover: amber-600, shadow-float
  Active: amber-700, scale 0.98

Photo Card:
  Background: white
  Border: 1pt stone-200 (#E7E5E4)
  Corner radius: 10pt
  Shadow: card shadow
  Selected: 2pt amber-500 border + glow shadow
  Hover: lift -2pt + shadow-float

Queue Row:
  Layout: HStack — thumbnail (60pt) | VStack(filename, stepName) | status icon
  Progress bar: 3pt height, amber fill on stone track, below filename
  Status: spinner (running) / green checkmark (done) / red X (failed)
```

### Core Design Principles

1. **Trust through transparency.** Every processing step is visible. No black boxes. Originals are always accessible.
2. **The photo is the hero.** UI recedes. Muted backgrounds, peripheral controls, minimal chrome.
3. **Dark during capture, light during review.** Dark mode during webcam capture reduces screen glare reflecting onto album pages (functional choice). Light mode for accurate color review.
4. **Progressive disclosure.** Advanced controls (crop handles, color sliders, manual warp) are available but tucked away. First-time users complete the full flow without opening any settings panel.

---

## Architecture

### Python Bridge Strategy

The app calls the existing CLI as a subprocess using `Foundation.Process`. No Python logic is re-implemented in Swift.

```
SwiftUI App (mac-app/)
    │
    ├── PipelineRunner.swift       Foundation.Process, one per input file
    │       │  env: ANTHROPIC_API_KEY, OPENAI_API_KEY, PYTHONUNBUFFERED=1
    │       │  cwd: <project root>   (required for Python imports to resolve)
    │       └── .venv/bin/python -m src.cli process <file> --output <outdir>
    │
    ├── CLIOutputParser.swift      stdout lines → typed PipelineEvent enum
    │
    └── SecretsLoader.swift        reads secrets.json → env vars for Process
```

**Output per job:** `~/Library/Application Support/SundayAlbum/output/<UUID>/`

**Output file naming** (from `src/cli.py:251–253`):
- Multi-photo page: `SundayAlbum_{stem}_Photo{i:02d}.jpg`
- Single photo: `SundayAlbum_{stem}.jpg`

**API keys (v1):** Read from `secrets.json` at project root. Injected as env vars into each subprocess. No user-facing Settings screen for keys in v1.

**Why one process per file (not `--batch`):** Spawning one process per input gives fine-grained per-file progress tracking and cancellation. Batch output would require multiplexed stdout parsing, which is more complex for no real benefit at 15–26s per file.

**Why `PYTHONUNBUFFERED=1`:** Without it, Python buffers stdout into 4 KB chunks. Progress lines arrive in batches at step boundaries rather than one at a time. This env var forces line-buffered output.

**Why set `currentDirectoryURL` to project root:** The CLI uses `from src.pipeline import ...` which requires the Python path to include the project root. Setting CWD is simpler than adding to `PYTHONPATH`.

---

## Xcode Project Structure

```
mac-app/
├── SundayAlbum.xcodeproj/
└── SundayAlbum/
    ├── SundayAlbumApp.swift           @main, inject AppState environment object
    ├── AppState.swift                 @Observable top-level state
    │
    ├── Bridge/
    │   ├── PipelineRunner.swift       Process launch, stdout streaming, output dir scan
    │   ├── CLIOutputParser.swift      Line → PipelineEvent, unit-testable
    │   └── SecretsLoader.swift        Read secrets.json
    │
    ├── Models/
    │   ├── ProcessingJob.swift        One input file + N extracted photos
    │   ├── ExtractedPhoto.swift       One JPEG output on disk
    │   └── PipelineEvent.swift        Typed stdout events enum
    │
    ├── Views/
    │   ├── ContentView.swift          NavigationSplitView root (3 columns)
    │   ├── DropZoneView.swift         Drop target + "Choose Files…" button
    │   ├── QueueView.swift            Sidebar: list of jobs
    │   ├── QueueRowView.swift         Per-job row with progress bar + step name
    │   ├── ResultsGridView.swift      LazyVGrid of extracted photos
    │   ├── PhotoThumbnailView.swift   Async thumbnail cell + selection ring
    │   ├── ComparisonView.swift       Side-by-side before/after inspector
    │   └── EmptyStateView.swift       Shown when queue is empty
    │
    └── Resources/
        └── Assets.xcassets/           App icon + color assets

SundayAlbumTests/
└── CLIOutputParserTests.swift         Unit tests using captured real stdout
```

**Xcode settings:**
- macOS 26.0 deployment target (Tahoe — developer tool, no need to support older versions)
- Swift 6 strict concurrency
- App Sandbox: **disabled** (developer tool, eliminates file permission friction)
- Bundle ID: `com.sundayalbum.mac`

---

## CLI Stdout Parsing Reference

Log format (`src/cli.py:28–31`): `%H:%M:%S - %(name)s - %(levelname)s - %(message)s`

| Stdout line pattern | PipelineEvent case |
|---|---|
| `INFO - Processing: <path>` | `.jobStarted` |
| `INFO - Load time: X.XXXs` | `.stepCompleted("Load Image")` |
| `INFO - Normalize time: X.XXXs` | `.stepCompleted("Normalize")` |
| `INFO - Page detection time: X.XXXs` | `.stepCompleted("Page Detection")` |
| `INFO - Glare removal time: X.XXXs` | `.stepCompleted("Glare Removal")` |
| `INFO - Geometry correction time: X.XXXs` | `.stepCompleted("Geometry")` |
| `INFO - Color restoration time: X.XXXs` | `.stepCompleted("Color Restoration")` |
| `INFO - Total processing time: X.XXXs` | `.processingComplete(totalTime:)` |
| `INFO - Saved: SundayAlbum_*.jpg` | `.outputSaved(filename:)` |
| `INFO - Photos extracted: N` | `.photosExtracted(count:)` |
| `ERROR - ...` | `.errorLine(message:)` |

Progress computation: 14 steps defined in `src/pipeline.py:33–132` (`PIPELINE_STEPS`). Each `.stepCompleted` event advances `progressFraction` by `1/14 ≈ 7%`.

---

## Data Models

```swift
// ProcessingJob
@Observable final class ProcessingJob: Identifiable {
    let id: UUID
    let inputURL: URL
    let outputDir: URL              // ~/Library/AS/SundayAlbum/output/<id>/
    var state: JobState             // .queued / .running / .complete / .failed
    var progressFraction: Double    // 0.0–1.0, advances per step completion
    var currentStepName: String     // e.g. "Glare Removal"
    var extractedPhotos: [ExtractedPhoto]
    var errorMessage: String?       // last ERROR line from CLI
    var processingTime: Double?
}

// ExtractedPhoto
@Observable final class ExtractedPhoto: Identifiable {
    let id: UUID
    let outputURL: URL              // SundayAlbum_*.jpg
    let jobID: UUID
    var thumbnail: NSImage?         // loaded async
    var isSelected: Bool
}

// PipelineEvent
enum PipelineEvent {
    case jobStarted
    case stepCompleted(name: String)
    case photosExtracted(count: Int)
    case outputSaved(filename: String)
    case processingComplete(totalTime: Double)
    case errorLine(message: String)
    case unknown(raw: String)
}
```

---

## UI Layout

### Main Window

```
┌─────────────────────────────────────────────────────────────────┐
│  [☀ Sunday Album]           [+ Add Photos]    [Export Selected] │
├──────────────────┬─────────────────────────┬────────────────────┤
│  QUEUE           │  RESULTS                │  COMPARISON        │
│                  │                         │                    │
│  cave.HEIC       │  [img] [img] [img]      │  BEFORE │ AFTER    │
│  ████████░░ 78%  │  [img] [img]            │         │          │
│  Glare Removal   │                         │  [orig] │ [result] │
│                  │                         │         │          │
│  harbor.HEIC     │                         │  cave.HEIC         │
│  ✓ 1 photo  18s  │                         │  Photo 1 of 1      │
│                  │                         │  Time: 18.3s       │
│  three_pics...   │                         │                    │
│  ⏳ Queued       │                         │ [Show in Finder]   │
│                  │                         │ [Add to Photos]    │
└──────────────────┴─────────────────────────┴────────────────────┘
```

When queue is empty: content area shows full-window `DropZoneView`.

### DropZoneView

- Dashed rounded-rect border, warm stone colors
- Text: "Drop album page photos here" / "or click to browse"
- `.onDrop(of: [.fileURL])` for drag-and-drop
- "Choose Files…" button opens `NSOpenPanel` (HEIC, DNG, JPG, PNG)
- Folder drop: walks folder for matching extensions

### QueueRowView

- Filename (truncated with ellipsis)
- Status badge: Queued (stone) / Processing (amber) / Done (green) / Failed (red)
- Thin `ProgressView(value:)` bar — visible only while running
- Current step name as 12pt caption
- When done: "N photos · 18.3s" badge

### ResultsGridView

- `LazyVGrid`, adaptive columns, minimum cell width ~200 pt
- Each `PhotoThumbnailView`: thumbnail loaded async, filename caption, amber selection ring on tap
- Toolbar: "Export Selected", "Select All", "Show in Finder"

### ComparisonView (inspector)

- Side-by-side: original input | extracted JPEG
- "After" image fades in with 0.4s opacity animation on first selection — the signature glare-reveal moment
- Metadata: filename, photo index (e.g. "Photo 2 of 3"), processing time
- Action buttons: "Show in Finder", "Add to Photos"

---

## Implementation Phases

### Phase 1: Full UI Layout ✅ COMPLETE

Built a full SwiftUI UI shell with mock data before wiring any real processing. This let us validate the layout and interaction design before investing in the Python bridge.

**What was built:**
- `AppState.swift` — `@Observable`, `AppScreen` enum, navigate/back helpers
- `ProcessingJob.swift`, `PipelineStep.swift`, `ExtractedPhoto.swift` — full model layer
- `DesignSystem.swift` — color tokens, custom fonts (Fraunces, DM Sans, JetBrains Mono), animations
- `LibraryView.swift` — full-window `LazyVGrid` of `AlbumPageCard`s; single-click expands overlay, double-click navigates
- `AlbumPageCard.swift` — compact card (60×88 before, natural-ratio afters, `PipelineProgressWheel`); `ExpandedAlbumCard` overlay
- `StepDetailView.swift` — `StepStrip` with debug thumbnails, step canvas switcher, `StepActionBar`, `ReprocessBar`
- Step canvases: `PageDetectionStepView`, `PhotoSplitStepView`, `GlareRemovalStepView`, `ColorCorrectionStepView`, `ResultsStepView`
- `MockData.swift` — 4 jobs at different pipeline states using real `debug/` folder images

**Key files:** All Swift sources under `mac-app/SundayAlbum/`

---

### Phase 2: File Import UI ✅ COMPLETE

Wire up the "Add Photos" toolbar button and library drag-and-drop so the app accepts real files. No pipeline processing yet — files are enqueued as `.queued` jobs and shown in the library.

**What was built:**
- `NSOpenPanel` behind the "Add Photos" toolbar button — `.heic`, `.dng`, `.jpg`, `.png`, multi-select, folder selection
- Folder handling: walks for matching extensions, enqueues each file as a separate job
- `.onDrop(of: [.fileURL])` on `LibraryView` — accepts drags from Finder
- On accept: creates `ProcessingJob`, appends to `appState.jobs`, shows card in library grid with `.queued` state

**Deliverable:** ✅ Drag a HEIC from Finder onto the library (or click "Add Photos") → new card appears in the grid with `.queued` state.

---

### Phase 3: Python Bridge

Prove the Swift → Python subprocess integration. This is the highest technical risk item.

1. `SecretsLoader.swift` — reads `secrets.json` at project root, returns `[String: String]` key map
2. `PipelineRunner.swift`:
   - `Foundation.Process`, executable = `.venv/bin/python`, args = `["-m", "src.cli", "process", <path>, "--output", <outdir>]`
   - `currentDirectoryURL` = project root (required for `from src.pipeline import ...` to resolve)
   - Environment: inherit + add `PYTHONUNBUFFERED=1`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
   - `Pipe` on stdout; `readabilityHandler` fires per line
   - Print raw lines to Xcode console for now
3. Hardcode a test HEIC path and run. **Verify lines arrive one at a time, not in batches.**

**Deliverable:** Xcode console shows live per-line pipeline stdout for a real HEIC file.

---

### Phase 4: Output Parser

Turn raw stdout into typed Swift state and wire into the existing UI models.

1. `CLIOutputParser.swift` — `parse(line:) -> PipelineEvent` (see parsing reference table above)
2. `CLIOutputParserTests.swift` — copy real terminal output into test cases, cover all `PipelineEvent` cases
3. Wire `PipelineRunner` to update `ProcessingJob` from parsed events:
   - `.stepCompleted` → advance `currentStep`, increment `progressFraction`
   - `.photosExtracted` → update count
   - `.outputSaved` → create `ExtractedPhoto`, append to `job.extractedPhotos`
   - `.processingComplete` → set `state = .complete`, `processingTime`
   - `.errorLine` → set `state = .failed`, `errorMessage`

**Deliverable:** All parser tests pass. A real job in the library progresses step-by-step and reaches `.complete` with thumbnails.

---

### Phase 5: Wire Drop → Pipeline

Replace mock data with real processing end-to-end.

1. Remove `MockData.swift` (or gate it behind a debug flag)
2. On file acceptance (from Phase 2): call `PipelineRunner.run(job:)` immediately (or queue for serial execution)
3. Output directory: `~/Library/Application Support/SundayAlbum/output/<job.id>/`
4. After `.complete`: `ExtractedPhoto` objects already created by the parser in Phase 4 — thumbnails load via existing `AfterThumb` async loader

**Deliverable:** Drop a real HEIC → library card shows live progress wheel → reaches "✓ N photos" → after-thumbnails appear in the card.

---

### Phase 6: Export & macOS Integration

Get processed photos out of the app.

1. **Show in Finder:** `NSWorkspace.shared.activateFileViewerSelecting([photo.imageURL])` — ~2 lines
2. **Add to Photos:** `Photos` framework + `NSPhotoLibraryAddUsageDescription` in Info.plist + `PHPhotoLibrary.shared().performChanges` — ~10 lines
3. **Export Selected:** "Export Selected" toolbar button → `NSOpenPanel` (directory picker mode) → copy selected JPEGs to chosen folder

**Deliverable:** Photos appear in Photos.app; Finder reveals the output JPEG.

---

### Phase 7: Polish

- App icon — warm amber palette, film/album motif
- Error state — red badge in card + last `ERROR -` log line shown on hover
- Job cancellation — "×" button on running cards calls `process.terminate()`
- Keyboard shortcuts: ⌘O open files, ⌘A select all photos, ⌘E export, ⌘⌫ remove job
- "Process More" toolbar button when library is non-empty (re-opens NSOpenPanel)

---

## Future Phases (Post-MVP)

### Webcam Capture Mode

Real-time guidance during album capture (from PRD sections 5.1.2 and 6.3.3):
- Dark-mode capture screen (`#0C0A09` background)
- Live webcam feed with page-detection green outline overlay
- Glare heat map overlay in amber (`rgba(234, 88, 12, 0.4)`) prompting user to tilt
- Stability ring that auto-captures when steady
- SPACE bar to capture without clicking
- Multi-camera selector dropdown
- Switches to light mode automatically when entering review

### Per-Photo Editor (from PRD section 6.3.8)

Split-view per-photo refinement:
- Left: photo with zoom/pan, draggable crop corner handles, "Auto-detect edges" button
- Right: exposure slider, white balance, saturation, "Restore fading" toggle, rotation slider ±15°, 90° CW/CCW buttons
- Bottom: "Before/After" toggle, "Revert to Auto", "Save"
- "Re-process glare" button reruns just the glare step via `--steps glare_detect`

### Settings Screen

- ANTHROPIC_API_KEY and OPENAI_API_KEY fields stored in macOS Keychain
- Output format: JPEG / PNG / TIFF
- JPEG quality slider 70–100%
- Output folder picker (default: `~/Pictures/SundayAlbum/`)
- Toggle: OpenAI glare removal on/off
- Toggle: AI orientation correction on/off

---

## Setup & Installation

Everything needed before writing or running any Swift code.

### 1. Xcode

Required to build and run the SwiftUI app. Not installable via Homebrew.

- **macOS 26 (Tahoe):** Download **Xcode 26** from [developer.apple.com/download](https://developer.apple.com/download) (requires free Apple Developer account). The Mac App Store may lag behind — use the direct download for beta/new OS versions.
- After installing, run once to accept the license and install additional components.
- Verify: `xcodebuild -version` should return `Xcode 26.x`.

### 2. xcodegen

Generates the `.xcodeproj` from a human-readable `project.yml` instead of hand-editing `project.pbxproj`.

```bash
brew install xcodegen
```

Used as: `xcodegen generate` from inside `mac-app/` whenever `project.yml` changes (e.g. adding a new Swift file).

### 3. Custom Fonts

Three fonts used by the design system. Install via Homebrew Cask — they become available system-wide and SwiftUI can reference them by name.

```bash
brew install --cask font-fraunces
brew install --cask font-dm-sans
brew install --cask font-jetbrains-mono
```

Font names for use in Swift code:
- `"Fraunces"` (variable, supports weight axis)
- `"DM Sans"` (variable, supports weight axis)
- `"JetBrains Mono"` (regular/bold variants)

Verify a font is registered after install:
```bash
python3 -c "import subprocess; print(subprocess.run(['fc-list', ':family=Fraunces'], capture_output=True, text=True).stdout)"
# or just open Font Book.app and search for "Fraunces"
```

### 4. Quick-Start After Installing All Three

```bash
# From the repo root:
cd mac-app/
xcodegen generate        # regenerates SundayAlbum.xcodeproj from project.yml
open SundayAlbum.xcodeproj
# In Xcode: select "My Mac" as destination → ▶ Run
```

---

## Technical Notes

**Swift 6 concurrency and @MainActor.** `PipelineRunner` must be `@MainActor`-isolated or use `await MainActor.run {}` when updating `@Observable` job state. The `readabilityHandler` closure fires on a background thread.

**Output directory isolation.** UUID-named subdirectory under `~/Library/Application Support/SundayAlbum/output/` prevents collisions when processing the same file twice. Directories persist until the user clears results.

**Before image for comparison.** Copy the original to `outputDir` at enqueue time (before the process starts) to avoid file access contention during processing.

**Xcode required.** Not available on Homebrew — download from Mac App Store or developer.apple.com.

---

## Verification Checklist

```
[x] Phase 1: Library grid, step detail, all step canvases render with mock data
[x] Phase 1: Cards show before/after thumbnails with correct aspect ratios
[x] Phase 1: PipelineProgressWheel animates for in-progress jobs
[x] Phase 2: Drag a HEIC from Finder → new card appears in library grid
[x] Phase 2: "Add Photos" button opens NSOpenPanel, multi-select works
[ ] Phase 3: Xcode console shows live per-line stdout from a real HEIC run
[ ] Phase 3: Lines arrive one-at-a-time (not batched), proving PYTHONUNBUFFERED=1 works
[ ] Phase 4: All CLIOutputParserTests pass
[ ] Phase 5: Drop IMG_three_pics_normal.HEIC → progress wheel advances step-by-step → "✓ 3 photos"
[ ] Phase 5: After thumbnails appear in card with natural portrait/landscape ratios
[ ] Phase 6: "Add to Photos" → photos appear in Photos.app
[ ] Phase 6: "Show in Finder" → Finder window reveals the output JPEG
```

---

## Critical Reference Files

| File | Relevant for |
|---|---|
| `src/cli.py:28–31` | Log format string |
| `src/cli.py:119–145` | All CLI flags (`--no-openai-glare`, `--no-ai-orientation`, `--workers`) |
| `src/cli.py:251–253` | Output filename pattern |
| `src/cli.py:262–271` | `Saved:` and `Processing Summary` log lines |
| `src/pipeline.py:33–132` | All 14 `PIPELINE_STEPS` names for progress mapping |
| `src/pipeline.py:269–905` | All `logger.info()` calls (parse targets) |
| `src/utils/secrets.py` | Key names in `secrets.json` |
| `docs/UI_Design_Album_Digitizer.md:37–160` | Full design token reference |
| `docs/PRD_Album_Digitizer.md` | Feature requirements for future phases |

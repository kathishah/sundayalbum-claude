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

### Phase 3: Python Bridge ✅ COMPLETE

Prove the Swift → Python subprocess integration. This is the highest technical risk item.

**What was built:**
- `SecretsLoader.swift` — reads `secrets.json` at project root, falls back to env vars, builds env dict with `PYTHONUNBUFFERED=1`
- `PipelineEvent.swift` — typed enum: `jobStarted`, `stepCompleted(name:)`, `photosExtracted(count:)`, `outputSaved(filename:)`, `processingComplete(totalTime:)`, `errorLine(message:)`, `unknown(raw:)`
- `CLIOutputParser.swift` — pure static parser (`parse(line:) -> PipelineEvent`), handles all log patterns from `src/pipeline.py`
- `PipelineRunner.swift` — `Foundation.Process` launch, `.venv/bin/python -m src.cli process <file> --output <outdir>`, `Pipe` stdout `readabilityHandler`, `@MainActor` job state updates per line; auto-detects project root by walking up from bundle looking for `.venv/`
- `ProcessingJob` — added `currentStepName: String?` and `photosExtractedCount: Int?` properties
- `CLIOutputParserTests.swift` — 16 tests covering all `PipelineEvent` cases, all passing

**Deliverable:** ✅ 29/29 tests pass. `PipelineRunner` compiles and is ready to wire into the UI.

---

### Phase 4: Output Parser + UI Wiring ✅ COMPLETE

Turn raw stdout into typed Swift state and wire into the existing UI models.

**What was built:**
- `AppState`: marked `@MainActor`, added `runners: [UUID: PipelineRunner]` dict; `addFiles` now starts a `PipelineRunner` per job immediately (`startProcessing: false` param for test isolation)
- `PipelineRunner`: maps CLI step names → `PipelineStep` enum so the progress wheel advances in sync with real processing; `@MainActor` job state updates on each parsed line
- `SundayAlbumApp`: uses real `AppState()` by default; set env var `MOCK_DATA=1` in Xcode scheme to restore mock data for UI work
- `AlbumPageCard / JobStatusLine`: shows live `currentStepName` from CLI stdout when available
- Compact card thumbnail layout: `GeometryReader` measures available width, divides it equally among all extracted photos — 1 photo fills full width, 2 split it, 3 split in thirds. All thumbnails use `.fill` cropping so cards are uniform width regardless of photo count
- `FileImporterTests`: updated for `@MainActor` + `startProcessing: false`

**Deliverable:** ✅ Drop a HEIC → library card shows live progress wheel step-by-step → reaches "N photos extracted" → all after-thumbnails appear scaled to fit the card.

---

### Phase 5: Wire Drop → Pipeline ✅ COMPLETE

Replace mock data with real processing end-to-end.

**What was built (completed as part of Phase 4):**
- `MockData.swift` gated behind `MOCK_DATA=1` env var in Xcode scheme
- `addFiles` starts `PipelineRunner` per job immediately on file drop or NSOpenPanel selection
- Output directory: `~/Library/Application Support/SundayAlbum/output/<job.id>/`
- `ExtractedPhoto` objects created by parser on each `.outputSaved` event; thumbnails load via `AfterThumb` async loader

**Deliverable:** ✅ Drop a real HEIC → library card shows live progress wheel → reaches "✓ N photos" → after-thumbnails appear scaled to fill card.

---

### Phase 6: Export & macOS Integration ✅ COMPLETE

Get processed photos out of the app.

**What was built:**
- `ExportActions.swift` — shared stateless helper enum with three actions:
  - `showInFinder(_:)` — `NSWorkspace.activateFileViewerSelecting`, reveals file selected in Finder
  - `addToPhotos(url:)` — `PHPhotoLibrary` with `.addOnly` authorization request + `PHAssetChangeRequest`
  - `exportToFolder(_:)` — `NSOpenPanel` directory picker → copies all job JPEGs → reveals in Finder
- `ComparisonView` — "Show in Finder" and "Add to Photos" buttons wired
- `ResultsStepView` — "Export All" button wired; thumbnail hover folder + photo-badge icons wired
- `project.yml` — `Photos.framework` dependency + `NSPhotoLibraryAddUsageDescription` key

**Deliverable:** ✅ "Show in Finder" reveals JPEG; "Add to Photos" saves to Photos.app; "Export All" copies to chosen folder.

---

### Phase 6b: Debug Folder Auto-Load + Dark Mode ✅ COMPLETE

Polish work added after Phase 6 export.

**What was built:**
- `DebugFolderScanner.swift` — scans `<projectRoot>/debug/` subdirectories at app launch, finds `15_photo_NN_final.jpg` (fallback: `14_photo_NN_enhanced.jpg`) per photo index, reconstructs completed `ProcessingJob`s so the library is pre-populated during development without re-running the pipeline
- `AppState.init(loadDebugJobs: Bool = true)` — `false` in tests to suppress debug job loading
- **Dark mode support** — `Color.dynamic(light:dark:)` via `NSColor(name:dynamicProvider:)` for all semantic tokens; app now follows system appearance setting
- **Light/dark color tweaks** — background = white (light) / black (dark); cards = soft warm grey in both modes
- **Card shadow** — adaptive `saShadow` token: dark drop shadow in light mode (opacity 0.45), subtle white glow in dark mode (opacity 0.12)
- Removed card divider between thumbnail row and filename; tightened padding; fixed vertical centering of thumbnails using `GeometryReader` `.frame(alignment: .center)`

---

### Phase 7: Polish ✅ COMPLETE

**What was built:**
- `AppState.removeJob(id:)` — cancels the running subprocess (if any) and removes the job from the library
- **Hover × button** on every `AlbumPageCard` — appears when the mouse enters the card; red `xmark.circle.fill` for queued/running jobs (cancels + removes), grey for completed/failed (removes only)
- **⌘O keyboard shortcut** — wired to the "Add Photos" toolbar button via `.keyboardShortcut("o", modifiers: .command)`, works from anywhere in the app

**Remaining Polish (future):**
- App icon — warm amber palette, film/album motif
- ⌘A select all photos in results grid, ⌘E export selected
- "Process More" toolbar button when library is non-empty (re-opens NSOpenPanel)

---

### Phase 9: Bundle Python Runtime (Option A — Guided Setup) ✅ COMPLETE

**Goal:** Make the app self-contained with the smallest possible download. Friends don't have `.venv/`, `src/`, or Python.

**Why the deps are unavoidably large:** OpenCV ~50MB, NumPy ~30MB, SciPy ~30MB, scikit-image ~30MB, Anthropic SDK ~15MB, OpenAI SDK ~15MB, Pillow + HEIC ~20MB, rawpy ~20MB ≈ ~200MB. There is no way around this — the choice is whether that weight is in the download or a one-time first-run fetch.

**Approach:** App bundle is ~5 MB. On first launch a setup screen downloads and installs deps (~3–5 min, internet required). Subsequent launches start instantly.

**Implementation steps:**

1. **Add `requirements-runtime.txt`** to the repo root — runtime-only deps, no pytest/mypy/matplotlib/ruff:
   ```
   opencv-python-headless
   numpy scipy scikit-image
   Pillow pillow-heif rawpy
   anthropic openai
   click python-dotenv
   ```

2. **Add `scripts/setup-runtime.sh`** — creates a venv and pip-installs runtime deps:
   ```bash
   #!/bin/bash
   set -e
   VENV_DIR="$1"
   python3 -m venv "$VENV_DIR"
   "$VENV_DIR/bin/pip" install --upgrade pip
   "$VENV_DIR/bin/pip" install -r "$2"   # $2 = path to requirements-runtime.txt
   ```
   Both files go in `Contents/Resources/` via `project.yml` resources.

3. **`SetupView.swift`** — full-window first-launch screen:
   - Shown when `~/Library/Application Support/SundayAlbum/venv/bin/python` does not exist
   - Title: "Setting up Sunday Album (one time only)"
   - `ScrollView` showing live pip output streamed line-by-line
   - Determinate `ProgressView` — parse `pip` output for package count to estimate progress
   - "Cancel" aborts the setup process; partially-installed venv is deleted
   - On completion: `withAnimation` transitions to `LibraryView`

4. **`PipelineRunner` changes:**
   - Production: use `~/Library/Application Support/SundayAlbum/venv/bin/python`
   - Dev fallback: walk up from bundle looking for `.venv/` (existing logic)
   - Remove the hardcoded `/Users/dev/dev/sundayalbum-claude` fallback
   - `currentDirectoryURL`: for production, any writable directory works (no `src/` import needed); for dev, still uses project root

**Uninstall — clean removal of all data:**

macOS has no system-level uninstaller, but the app owns everything under one directory. Add **"Uninstall Sunday Album…"** to the Help menu:

1. Show a confirmation sheet listing what will be removed:
   ```
   • Python runtime     ~/Library/Application Support/SundayAlbum/venv/    (~200 MB)
   • Processed photos   ~/Library/Application Support/SundayAlbum/output/
   • Debug images       ~/Library/Application Support/SundayAlbum/debug/
   • App preferences    (UserDefaults)
   • API keys           (Keychain)
   ```
2. On confirm: delete the entire `~/Library/Application Support/SundayAlbum/` directory, clear UserDefaults, delete Keychain entries
3. Show a final notice: *"All data removed. Drag Sunday Album from Applications to Trash to finish uninstalling."*
4. Quit the app

This means using AppCleaner or dragging to Trash alone leaves ~200MB behind — the in-app uninstaller is the correct path and should be documented in the README and beta invite.

**Architecture:** ARM64 only for v1 beta. Intel requires a separate build on an Intel Mac (defer).

**What was built:**

**`requirements-runtime.txt`** — repo root, runtime-only pip deps (no pytest/mypy/ruff/matplotlib/torch).

**`scripts/setup-runtime.sh`** — `set -euo pipefail` bash script; args: `<venv_dir> <requirements_file>`. Creates venv via `python3 -m venv`, upgrades pip, pip-installs the requirements file. Bundled in `Contents/Resources/` via `project.yml`.

**`RuntimeManager.swift`** — `@MainActor @Observable` singleton:
- `SetupState` enum: `.needsSetup` / `.installing` / `.ready` / `.failed(String)`
- State computed synchronously in `init()` — no first-frame flash
- **Dev mode**: detects `.venv/` walking up from bundle URL (+ hardcoded fallback); state immediately `.ready`
- **Production mode**: checks for `~/Library/AS/SundayAlbum/venv/bin/python`
- `startInstallation()` — streams `setup-runtime.sh` stdout line-by-line into `installLog`; estimates progress from "Downloading" + "Successfully installed" pip lines
- `cancelInstallation()` — terminates process, deletes partial venv, resets to `.needsSetup`
- `retrySetup()` — resets from `.failed` to `.needsSetup`
- `extraPythonPath` — returns `Bundle.main.resourceURL` for production (injected as `PYTHONPATH` so `import src.*` resolves against the bundled `src/` folder); `nil` in dev
- `promptAndUninstall()` — NSAlert confirmation → removes `~/Library/AS/SundayAlbum/` + UserDefaults → final notice → `NSApp.terminate`

**`SetupView.swift`** — full-window branded first-launch screen with three sub-states:
- **NeedsSetupBody**: descriptive text + amber "Set Up Sunday Album" CTA button
- **InstallingBody**: determinate progress bar + scrolling pip log (`ScrollViewReader` auto-scrolls to bottom) + Cancel button
- **FailedBody**: error icon + message + last 60 log lines + "Try Again" button

**`SundayAlbumApp.swift`** — gated root:
- `Group { if runtime.setupState == .ready { ContentView() } else { SetupView() } }` with `.animation(.saStandard, value:)` for smooth transition once setup completes
- `RuntimeManager.shared` injected via `.environment(runtime)`
- `.commands { CommandGroup(after: .help) { "Uninstall Sunday Album…" } }` calls `RuntimeManager.shared.promptAndUninstall()`

**`PipelineRunner.swift`** — path resolution delegated to `RuntimeManager`:
- `pythonURL` from `RuntimeManager.shared.pythonURL`
- `cliWorkingDirectory` from `RuntimeManager.shared.cliWorkingDirectory`
- `PYTHONPATH` env var set to `runtime.extraPythonPath` (production only)
- Removed static `projectRoot` computed property

**`SecretsLoader.swift`** — bug fix: `environment()` now layers SettingsStorage keys (highest priority) over `secrets.json` keys, so API keys saved in ⌘, Settings actually reach the subprocess.

**`project.yml`** — added three resource entries under the `SundayAlbum` target:
- `../scripts/setup-runtime.sh`
- `../requirements-runtime.txt`
- `../src` (type: folder) — Python package copied as folder reference into `Contents/Resources/src/`

**`DebugFolderScanner.swift`** — bug fix (2026-03-25): replaced `PipelineRunner.projectRoot` (removed in Phase 9 refactor) with `RuntimeManager.shared.cliWorkingDirectory`. In dev mode this is the project root (same value as before); in production it gracefully finds no `debug/` folder and returns `[]`.

**`mac-app/ExportOptions.plist`** — xcodebuild export config for ad-hoc distribution (no Apple Developer account required). Uses `method: mac-application` and `signingStyle: automatic`. Switch to `method: developer-id` when notarization is needed (Phase 10).

**`mac-app/build-release.sh`** — one-command release builder:
1. Runs `xcodegen generate`
2. Archives in Release config (`xcodebuild archive`)
3. Exports `.app` via `ExportOptions.plist`
4. Zips to `~/Desktop/SundayAlbum-<version>.zip`

Usage: `cd mac-app && ./build-release.sh`

**How to test:**

| Goal | How |
|------|-----|
| Dev mode (instant, no setup) | Build & Run in Xcode — `devProjectRoot` finds `.venv/` automatically |
| Production first-launch flow | `mv .venv .venv-hidden` then build & run; app shows SetupView and installs to `~/Library/Application Support/SundayAlbum/venv/`. Restore with `mv .venv-hidden .venv` |
| Test the actual distributable | `cd mac-app && ./build-release.sh` → unzip on Desktop → right-click Open |

**Deliverable:** ✅ App is self-contained. A new user receives a ~5 MB .app, sees the setup screen on first launch, waits ~3–5 min for pip to install ~200 MB of deps, then proceeds normally. The Help menu > "Uninstall Sunday Album…" removes all data cleanly. `build-release.sh` produces a shareable zip in one command.

---

### Phase 8: Settings Screen (API Keys + Storage) ✅ COMPLETE

**Goal:** Friends configure their own API keys and choose where files are stored. Processing is blocked until the Anthropic key is present. Keys can be tested or replaced at any time.

**What was built:**

**Storage: file-based JSON instead of Keychain**
- `SettingsStorage.swift` — stores API keys in `~/Library/Application Support/SundayAlbum/settings.json`. Simpler than Keychain, no entitlement required for a dev tool.
- `KeychainHelper.swift` — emptied (replaced by `SettingsStorage`)
- `Security.framework` dependency removed from `project.yml`

**`AppSettings.swift`** — `@MainActor @Observable` singleton:
- `canProcess` returns `true` for `.untested` and `.testing` (not just `.valid`), so the banner doesn't appear every launch while keys are re-testing in background
- Auto-tests saved keys via `.task` on app launch (background, non-blocking)
- `PipelineRunner` reads `outputFolder`, `debugFolder`, `debugOutputEnabled`, `useOpenCVFallback` from `AppSettings` instead of hardcoded paths

**`SettingsView.swift`** — opened by ⌘,:
- API Keys section: per-key row with text field, Test/Save/Discard buttons, `KeyStatusBadge`
- Storage section: output folder picker + debug folder picker (shown when debug enabled)
- Save triggers auto-test; Discard reverts to last saved value

**`LibraryView`** — amber banner + `@Environment(\.openSettings)` action (correct SwiftUI way to open Settings window).

**`src/cli.py`** — added `--debug-dir <path>` so the app can direct debug images to the user-chosen folder.

**Step detail tree navigation (also completed in this phase):**
- `StepDetailView` restructured: horizontal step strip replaced by scrollable left tree pane (196pt) + right canvas
- `StepSelection` enum: `.preSplit(PipelineStep)` for Load/Page/Split, `.photo(index:, step:)` for per-photo steps
- Multi-photo jobs show a branching tree — each extracted photo has its own Orient → Glare → Color → Done branch with a vertical connector line and thumbnail header
- Single-photo jobs show a flat list (no branching)
- `OrientationStepView`, `GlareRemovalStepView`, `ColorCorrectionStepView` all take a `photoIndex` parameter; thumbnail strips removed (tree handles navigation)
- `ExtractedPhoto` gains `rotationOverride: Int?` and `sceneDescription: String?` for user overrides in the Orientation step
- `PipelineStep.debugImageURL` uses `AppSettings.shared.debugFolder` instead of hardcoded dev path

**Deliverable:** ✅ Settings window opens via ⌘,; keys can be entered, tested, and saved; banner only appears when key is absent/invalid; step detail shows per-photo branches for multi-photo jobs.

---

### Phase 10: Code Signing + Notarization *(deferred — not required for v1 beta)*

**Can friends bypass the warning without notarization?** Yes — for a small beta group of technical friends, the unsigned app works fine. macOS blocks it by default but provides an easy bypass:

```
Option 1 (easiest): Right-click the app → "Open" → click "Open" in the dialog
Option 2: System Settings → Privacy & Security → scroll down → "Open Anyway"
Option 3 (one-liner in Terminal):
  xattr -d com.apple.quarantine /Applications/SundayAlbum.app
```

Include these instructions in the beta invite. All three work on macOS 15.

**When notarization becomes necessary:**
- Distributing to non-technical users who won't know how to bypass
- More than ~10–20 beta testers (managing the bypass instructions at scale is annoying)
- Any public release or App Store submission

**When you're ready to notarize (prerequisites):**
- Apple Developer Program membership ($99/year)
- Developer ID Application certificate in Keychain
- App Store Connect API key (for `notarytool`)

**Steps (for future reference):**

1. Enable Hardened Runtime in `project.yml`:
   ```yaml
   settings:
     ENABLE_HARDENED_RUNTIME: YES
   ```

2. Add `SundayAlbum.entitlements`:
   ```xml
   <key>com.apple.security.network.client</key><true/>
   <!-- PyInstaller / venv subprocess needs these: -->
   <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
   <key>com.apple.security.cs.disable-library-validation</key><true/>
   ```

3. Sign the embedded CLI binary (if using Option B/C from Phase 8):
   ```bash
   codesign --force --sign "Developer ID Application: You (TEAMID)" \
     --options runtime mac-app/SundayAlbum/Resources/sundayalbum-cli
   ```

4. Archive → Distribute App → Developer ID from Xcode, then:
   ```bash
   xcrun notarytool submit SundayAlbum.zip --apple-id you@email.com \
     --team-id TEAMID --password "@keychain:AC_PASSWORD" --wait
   xcrun stapler staple SundayAlbum.app
   spctl --assess --type execute SundayAlbum.app  # should print: accepted
   ```

**App Sandbox:** Keep disabled. Sandbox + subprocess launching has significant entitlement overhead; re-evaluate only if submitting to the Mac App Store.

---

### Phase 11: DMG + GitHub Release

**Goal:** A drag-to-install `.dmg` that friends can download from a single link.

**Steps:**

1. **Create DMG** using `create-dmg` (Homebrew):
   ```bash
   brew install create-dmg
   create-dmg \
     --volname "Sunday Album" \
     --background "assets/dmg-background.png" \
     --window-size 660 400 \
     --icon-size 128 \
     --icon "SundayAlbum.app" 180 170 \
     --app-drop-link 480 170 \
     "SundayAlbum-1.0.0-beta.dmg" \
     "export/SundayAlbum.app"
   ```

2. **GitHub Release:**
   - Tag: `v1.0.0-beta1`
   - Attach `SundayAlbum-1.0.0-beta.dmg` as a release asset
   - Release notes: system requirements, what to test, how to report bugs

3. **System requirements for beta invite:**
   ```
   macOS 15 Sequoia or later
   Apple Silicon Mac (M1/M2/M3/M4)
   Anthropic API key (Claude) — free tier works
   OpenAI API key — optional, enables higher-quality glare removal
   ~500 MB disk space (app bundle includes Python runtime)
   ```

4. **Feedback channel** — include a GitHub Issues link or simple Google Form in the app's Help menu

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
[x] Phase 3: CLIOutputParser, PipelineRunner, SecretsLoader built and compiling
[x] Phase 3: All CLIOutputParserTests pass (16/16 parser tests + 29/29 total)
[ ] Phase 3: Manual verify — Xcode console shows live per-line stdout from a real HEIC run
[x] Phase 4: PipelineRunner wired into UI — drop a file → progress wheel advances step-by-step
[x] Phase 4: After-thumbnails scale to fill card width for 1, 2, or 3 extracted photos
[x] Phase 5: Drop IMG_three_pics_normal.HEIC → progress wheel advances step-by-step → "✓ 3 photos"
[x] Phase 5: After thumbnails appear in card scaled to fill available width
[x] Phase 6: "Add to Photos" → photos appear in Photos.app
[x] Phase 6: "Show in Finder" → Finder window reveals the output JPEG
[x] Phase 6: "Export All" → folder picker → copies all JPEGs → reveals in Finder
[x] Phase 6b: App launch pre-populates library from debug/ folder (no reprocessing needed)
[x] Phase 6b: App follows system light/dark mode setting
[x] Phase 7: Hover × on card removes queued/complete jobs; cancels + removes running jobs
[x] Phase 7: ⌘O triggers "Add Photos" file picker
[x] Phase 8: ⌘, opens Settings window with API key entry and storage folder pickers
[x] Phase 8: Amber banner appears when Anthropic key is absent/invalid; "Open Settings →" navigates correctly
[x] Phase 8: Banner does not appear on launch when a valid key is already saved
[x] Phase 8: API keys stored in file-based JSON (not Keychain); no Security.framework required
[x] Phase 8: PipelineRunner uses AppSettings.outputFolder and AppSettings.debugFolder
[x] Phase 8: Step detail shows left tree pane with per-photo branches for multi-photo jobs
[x] Phase 8: Clicking Orient/Glare/Color/Done for any photo branch shows that photo's canvas
[x] Phase 8: Orientation step shows rotation picker + scene description editor per photo
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

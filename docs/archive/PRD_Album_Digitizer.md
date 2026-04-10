# Album Digitizer — Product Requirements Document

**Version:** 1.0  
**Date:** February 2026  
**Status:** Draft  
**Author:** [Product Owner]

---

## 1. Executive Summary

Album Digitizer is a free, consumer-first tool that transforms physical photo album pages into clean, individual digital photos. Unlike existing solutions (Google PhotoScan, paid scanner apps), Album Digitizer specifically targets the most painful problems: glare from glossy plastic sleeves, multiple photos per album page, perspective distortion, and faded/aged colors.

The product consists of a cloud processing engine (AWS) with a responsive web UI as the primary interface, with future native apps for Mac desktop and iOS to enable local pre-processing and camera integration.

**Target user:** Anyone with old family photo albums who wants to preserve them digitally — no technical expertise required.

---

## 2. Problem Statement

### 2.1 Current Pain Points

Digitizing physical photo albums is tedious and error-prone. Users face a cascade of issues that no single existing tool handles well:

- **Glossy sleeve glare:** Overhead and ambient light creates bright reflections on the plastic sleeves covering album pages, ruining photos. This is the #1 frustration.
- **Multiple photos per page:** Album pages often contain 1, 2, 3, or more photos of varying sizes and orientations. Users must manually crop each one.
- **Perspective distortion (keystoning):** Handheld or angled shots warp rectangular photos into trapezoids.
- **Photo bulging:** Photos behind plastic sleeves can bow or curl, creating non-planar surfaces that standard perspective correction cannot fix.
- **Color degradation:** Decades-old prints suffer from fading, yellowing, color shifts, and contrast loss.
- **Scale:** Users have hundreds of photos across multiple albums. The workflow must be efficient at volume.

### 2.2 Existing Solutions & Gaps

| Solution | Glare Removal | Multi-Photo Split | Keystone Fix | Color Restore | Price |
|----------|--------------|-------------------|-------------|---------------|-------|
| Google PhotoScan | Partial (multi-shot) | No | Basic | No | Free |
| Photomyne | No | Yes | Basic | Basic | $10/mo |
| Pic Scanner Gold | No | Yes | Manual | Basic | $8 one-time |
| Flatbed scanner | N/A (no glare) | Manual | N/A | No | Hardware cost |
| Album Digitizer | **Yes** | **Yes** | **Yes** | **Yes** | **Free** |

---

## 3. Target Audience

### 3.1 Primary Persona — "The Family Archivist"

- Age 30–70
- Has 1–10 physical photo albums from the 1960s–2000s
- Moderate smartphone proficiency, not technically expert
- Motivated by preserving family memories before photos degrade further
- Willing to invest time but not significant money
- Albums use glossy plastic sleeve pages (the most common album type)

### 3.2 Secondary Persona — "The Batch Processor"

- Has a large collection (10+ albums, 500+ photos)
- Values speed and automation over manual control
- Comfortable with desktop workflows
- Wants to process an entire album in one sitting

### 3.3 Anti-Personas (Not Targeting)

- Professional archivists requiring museum-grade fidelity (they use flatbed scanners)
- Users who want photo editing (filters, artistic effects, collage-making)
- Users digitizing loose photos or documents (no album context)

---

## 4. Product Vision & Principles

### 4.1 Vision

The best free tool in the world for digitizing physical photo albums — opinionated about quality, effortless in workflow.

### 4.2 Design Principles

1. **Glare-first design:** Every UX and processing decision assumes glossy plastic sleeves. This is our differentiator.
2. **Automation over manual effort:** The tool should do the right thing by default. Manual adjustments are available but never required.
3. **Batch-friendly:** The workflow must scale to hundreds of photos without per-photo friction.
4. **Forgiving capture:** Users should not need a tripod, controlled lighting, or photography skills. The processing engine compensates for imperfect input.
5. **Free and open:** No paywalls, no "premium" tiers for core features, no watermarks.

---

## 5. Feature Requirements

### 5.1 Image Capture & Input

#### 5.1.1 Supported Input Methods

- **Phone camera (via web UI):** User photographs album pages using their phone's browser. The web UI activates the rear camera and provides a live viewfinder with real-time guidance overlays.
- **Desktop webcam (via web UI):** User positions their album under a webcam (e.g., MacBook camera, Razer Kiyo Pro) and captures pages through the browser.
- **File upload:** User uploads previously taken photos (JPEG, PNG, HEIF/HEIC) from their device. Supports drag-and-drop and multi-file selection.
- **Batch upload:** User selects an entire folder or multiple files at once for processing as a set.

#### 5.1.2 Capture Guidance (Real-Time Viewfinder)

When using live camera capture (phone or webcam), the viewfinder provides:

- **Album page detection:** A green outline appears when the system detects a full album page in frame. Turns red/yellow if page is partially cut off.
- **Glare detection warning:** Real-time highlight overlay showing areas of detected glare. Prompts user: "Tilt slightly to reduce glare" with a directional arrow suggesting which way to tilt.
- **Stability indicator:** A steadiness ring (similar to iOS level) that fills when the phone is stable enough for a sharp capture. Auto-captures when stable (configurable).
- **Multi-shot glare mode:** For stubborn glare, prompts the user to take 3–5 shots at slightly different angles. The system composites a glare-free result from the set.
- **Distance/framing guide:** Suggests optimal distance so the album page fills the frame without excessive background.

#### 5.1.3 Capture Modes

- **Single-page mode:** Capture and process one album page at a time. Best for users who want to review each result immediately.
- **Continuous/batch mode:** Rapid-fire capture. User flips pages and taps to capture (or auto-captures on stability). Photos queue for batch processing. Minimal interruption between captures.

### 5.2 Core Processing Engine

#### 5.2.1 Glare & Reflection Removal (Priority 1)

This is the product's primary differentiator and must deliver best-in-class results.

**Single-shot glare removal:**
- Detect glare regions (specular highlights, washed-out areas) in a single input image.
- Reconstruct the underlying photo content in glare-affected regions using surrounding context and learned priors.
- Handle both small point-source reflections and large diffuse glare from overhead lighting.

**Multi-shot glare compositing:**
- Accept 3–5 images of the same album page taken at slightly different angles.
- Align images using feature matching (the album page content stays constant; glare moves with angle change).
- Composite a glare-free result by selecting non-glare pixels from across the set.
- This mode produces the best results and should be the recommended workflow for users with severe glare.

**Glare confidence map:**
- Generate a per-pixel confidence score indicating how well glare was removed.
- Surface this to the user as a subtle overlay (e.g., areas with low confidence highlighted in orange) so they know if a re-shoot would help.

#### 5.2.2 Photo Detection & Splitting (Priority 2)

Album pages contain varying numbers of photos in diverse arrangements.

**Auto-detection:**
- Detect individual photo boundaries within an album page image.
- Handle 1, 2, 3, 4, or more photos per page.
- Support photos in portrait and landscape orientations, mixed on the same page.
- Detect photos even when partially overlapping, slightly askew, or different sizes.
- Distinguish photos from decorative album page elements (borders, captions, stickers).

**Splitting & extraction:**
- Crop each detected photo into its own individual image.
- Apply per-photo corrections (each photo gets its own perspective correction, color adjustment, etc.).
- Maintain spatial ordering: photos are numbered left-to-right, top-to-bottom by default.
- Handle edge cases: single panoramic photo spanning a full page, tiny wallet-size photos, Polaroids.

**Manual override:**
- If auto-detection misidentifies boundaries, user can manually draw crop rectangles.
- User can merge two detected regions (if a single photo was split) or split one region (if two photos were merged).
- User can mark a region as "not a photo" (e.g., an album caption or decorative element).

#### 5.2.3 Perspective & Geometry Correction (Priority 3)

**Keystone correction:**
- Detect the four corners of each individual photo.
- Apply homographic transformation to produce a properly rectangular output.
- Handle significant perspective distortion (photos taken at up to ~40° angle).

**Bulge/warp correction:**
- Detect non-planar distortion caused by photos bowing behind plastic sleeves.
- Apply mesh-based dewarping to flatten the photo.
- Handle both convex (center bulging out) and concave (edges curling up) distortions.
- This is a differentiating feature — most competing apps ignore it entirely.

**Rotation correction:**
- Auto-detect and correct small rotational misalignment (< 15°).
- Detect if a photo is sideways or upside down and auto-rotate to correct orientation.
- Use content analysis (face detection, text direction, scene understanding) to determine correct orientation.

#### 5.2.4 Color Restoration & Enhancement (Priority 4)

**Auto white balance:**
- Correct color casts introduced by: the camera's white balance, the album page's tinted plastic sleeve, ambient lighting color.
- Use the album page border (visible around photos) as a white/neutral reference point when possible.

**Fade restoration:**
- Detect and compensate for overall fading (loss of saturation and contrast common in 1970s–1990s prints).
- Restore dynamic range without introducing artifacts.

**Yellowing correction:**
- Remove yellow/brown tint common in aged photos.
- Distinguish between intentional warm tones (sunset photo) and degradation-induced yellowing.

**Contrast & sharpness:**
- Auto-enhance contrast using adaptive tone mapping.
- Apply intelligent sharpening to compensate for slight defocus from handheld capture.
- All enhancements are subtle and photorealistic — never make photos look "filtered" or artificial.

**Manual color controls (optional):**
- Exposure slider (-2 to +2 EV equivalent).
- White balance temperature slider.
- Saturation slider.
- A single "restore" button that applies best-guess restoration to defaults.
- Before/after comparison toggle.

### 5.3 Output & Export

#### 5.3.1 Output Formats

Users choose output format at the batch level (with option to change per-photo):

- **JPEG:** Default for most users. Configurable quality (70–100%, default 90%).
- **PNG:** Lossless option for users who want maximum quality.
- **TIFF:** For archival users who want an industry-standard lossless format.
- **Original + enhanced:** Export both the uncorrected (but cropped/perspective-fixed) version and the color-enhanced version as separate files.

#### 5.3.2 Output Resolution

- Output resolution matches input resolution (no upscaling by default).
- Optional AI upscaling (2× or 4×) for low-resolution captures.
- Minimum recommended input: 8 MP (guidance shown if input is lower).

#### 5.3.3 File Naming & Organization

- Default naming: `Album_PageXX_PhotoYY.ext` (e.g., `Album_Page03_Photo02.jpg`).
- User can set a custom album name that prefixes all files.
- Photos are organized into folders by album page.
- Batch export downloads as a single ZIP file.

#### 5.3.4 Download & Delivery

- Individual photo download (one at a time).
- Batch download (all processed photos as ZIP).
- Per-page download (all photos from a single album page as ZIP).
- Download history persists for the duration of the browser session.

### 5.4 Review & Editing Workflow

#### 5.4.1 Processing Queue & Status

- After capture or upload, images appear in a processing queue.
- Each item shows: thumbnail, status (queued / processing / done / error), processing time estimate.
- Queue processes in parallel (multiple pages simultaneously).
- User can re-order the queue or cancel pending items.

#### 5.4.2 Results Review

- **Page view:** Shows the original album page alongside all extracted photos from that page.
- **Photo grid:** Shows all extracted photos across all pages in a scrollable grid.
- **Before/after slider:** Each photo has a draggable slider to compare original vs. processed.
- **Zoom:** Click any photo to view full-resolution with zoom/pan.

#### 5.4.3 Manual Adjustments

For any individual photo, the user can:

- Adjust crop boundaries (drag corners/edges).
- Re-run perspective correction with manually placed corner points.
- Toggle color enhancement on/off.
- Adjust color sliders (exposure, white balance, saturation).
- Rotate 90° CW/CCW.
- Flip horizontal/vertical.
- Mark as "re-shoot needed" (flags the original page for recapture).
- Revert all changes to the auto-processed defaults.

#### 5.4.4 Batch Operations

- Apply color settings from one photo to all photos in the batch.
- Select multiple photos and delete, re-process, or export.
- "Accept all" button to approve all auto-processed results and proceed to export.

### 5.5 Session & Data Management

#### 5.5.1 Session Persistence

- Processing sessions are tied to the browser (no account required).
- Sessions persist for 7 days. After 7 days, uploaded originals and processed results are automatically deleted from the server.
- User sees a countdown/expiration notice on their session.
- User can manually delete their session and all associated data at any time.

#### 5.5.2 Privacy & Data

- No account required for full functionality.
- No photos are used for training or shared with third parties.
- All photos are encrypted at rest on the server.
- Clear privacy policy accessible from every screen.
- GDPR/CCPA compliant data deletion.

#### 5.5.3 Optional Account (Future Phase)

- Optional free account for session persistence beyond 7 days.
- Sync sessions across devices (start capture on phone, review/export on desktop).
- Processing history and re-download past exports.

---

## 6. User Interface Design

### 6.1 Design Language

- **Clean, minimal, warm.** The UI should feel approachable, not technical. Think "family photo album" not "Photoshop."
- **Large touch targets** for mobile use (many users will capture from phones).
- **Dark mode by default** during capture (reduces screen glare reflecting onto album pages) with light mode toggle.
- **Progress and delight:** Gentle animations as photos process. Satisfaction moment when a glare-free result appears.

### 6.2 Information Architecture

```
Home (Landing Page)
├── Start Digitizing (primary CTA)
│   ├── Capture Mode
│   │   ├── Phone Camera Viewfinder
│   │   └── Webcam Viewfinder
│   ├── Upload Mode
│   │   └── File Picker / Drag-and-Drop
│   └── Processing Queue
│       ├── Page Results View
│       ├── Photo Grid View
│       ├── Individual Photo Editor
│       └── Export / Download
├── How It Works (explainer)
├── FAQ
└── Privacy Policy
```

### 6.3 Screen-by-Screen Design

#### 6.3.1 Landing Page

**Purpose:** Explain what the tool does and get the user started immediately.

**Layout:**
- Hero section with a before/after comparison (glossy album page → clean individual photos). The comparison should be interactive (draggable slider).
- One-line tagline: "Digitize your photo albums. No glare. No hassle. Free."
- Two primary CTAs side by side: "Start with Camera" and "Upload Photos."
- Brief "How It Works" section below: 3 steps illustrated with icons — (1) Photograph your album pages, (2) We remove glare, split photos, fix colors, (3) Download your clean photos.
- Footer: FAQ link, Privacy Policy link, GitHub link (open source).

#### 6.3.2 Capture Screen (Phone Camera)

**Purpose:** Guide the user through photographing album pages with real-time assistance.

**Layout — Full-Screen Viewfinder:**
- Camera feed occupies full screen.
- Semi-transparent overlays on top:
  - Green/red rectangle showing detected album page boundary.
  - Glare heat map overlay (toggle-able) — highlighted regions with detected glare shown in translucent orange/red.
  - Stability ring in bottom center — fills as phone steadies.
- Top bar (translucent): page counter ("Page 5 of ?"), settings gear icon, close (X).
- Bottom bar:
  - Capture button (large, centered). Pulses when conditions are optimal.
  - Mode toggle: "Single Shot" / "Multi-Shot (Glare)" — with brief tooltip explaining multi-shot mode.
  - Thumbnail of last captured page (tap to review).
- When multi-shot mode is active: progress indicator showing "Shot 2 of 4" and directional hint for angle change.
- After capture: brief flash confirmation, thumbnail slides into bottom bar, counter increments. Minimal disruption.

#### 6.3.3 Capture Screen (Desktop Webcam)

**Purpose:** Same as phone but adapted for desktop webcam setup.

**Layout:**
- Webcam feed in a large central panel (16:9 or square crop).
- Same overlay system as phone (page detection, glare, stability).
- Side panel showing the capture queue (thumbnails of captured pages, scrollable).
- Keyboard shortcut visible: "Press SPACE to capture" — no need to click a button.
- Webcam selector dropdown (for users with multiple cameras).

#### 6.3.4 Upload Screen

**Purpose:** Accept pre-taken photos for processing.

**Layout:**
- Large drag-and-drop zone occupying most of the screen. Text: "Drop your album page photos here" with a subtle upload icon.
- "Or browse files" button below the drop zone.
- Supports multi-select in the file picker.
- As files are added, they appear as thumbnails in a horizontal strip below the drop zone.
- Batch settings panel (collapsible):
  - Output format selector: JPEG / PNG / TIFF (radio buttons).
  - Quality slider (JPEG only): 70%–100%.
  - Album name text field (optional, for file naming).
- "Process All" button (prominent, bottom-right) — disabled until at least one file is added.

#### 6.3.5 Processing Queue Screen

**Purpose:** Show progress and allow the user to manage their batch.

**Layout:**
- Vertical list of album pages being processed.
- Each row shows: page thumbnail (original), status icon (spinner / checkmark / warning), progress bar, estimated time remaining.
- Completed pages expand to show extracted photo thumbnails inline.
- Top summary bar: "Processing 12 of 47 pages — ~3 min remaining."
- Buttons: "Pause All," "Cancel All."
- Clicking a completed page navigates to the Page Results View.

#### 6.3.6 Page Results View

**Purpose:** Review the extracted photos from a single album page.

**Layout:**
- Top half: original album page image (zoomable).
- Bottom half: extracted individual photos laid out in a grid.
- Each extracted photo shows:
  - Thumbnail with before/after slider (drag to compare).
  - Quick action icons: rotate, crop adjust, download.
  - Glare confidence indicator (green checkmark if clean, orange dot if some residual glare detected).
- Navigation arrows (or swipe) to move to next/previous page.
- "Accept & Next" button — approves all photos on this page and advances.

#### 6.3.7 Photo Grid View

**Purpose:** Overview of all extracted photos across the entire batch.

**Layout:**
- Masonry or uniform grid of all extracted photo thumbnails.
- Filter/sort options: by page order, by quality confidence, by "needs attention."
- Multi-select mode: tap/click to select multiple photos for batch actions (export, delete, re-process).
- Top bar: total count ("142 photos extracted from 47 pages"), export button.
- Clicking any photo opens the Individual Photo Editor.

#### 6.3.8 Individual Photo Editor

**Purpose:** Fine-tune a single photo's corrections.

**Layout — Split View:**
- Left panel: the photo with current corrections applied. Zoomable and pannable.
- Right panel (collapsible on mobile): adjustment controls.
  - **Crop & Geometry section:**
    - Draggable corner handles for crop adjustment.
    - "Auto-detect edges" button to re-run boundary detection.
    - "Fix perspective" button (with manual 4-point mode).
    - Rotation slider (-15° to +15°) with fine-tuning.
    - 90° rotation buttons (CW/CCW).
  - **Color section:**
    - "Auto enhance" toggle (on by default).
    - Exposure slider.
    - White balance slider.
    - Saturation slider.
    - "Restore fading" toggle.
  - **Glare section:**
    - Glare confidence map overlay toggle.
    - "Re-process glare" button (if multi-shot data is available).
    - Manual glare mask painting (user paints over remaining glare for targeted re-processing).
- Bottom bar: "Before/After" toggle, "Revert to Auto," "Save," "Download this photo."

#### 6.3.9 Export Screen

**Purpose:** Final export of all approved photos.

**Layout:**
- Summary: "Ready to export 142 photos."
- Output settings (pre-filled from batch settings, editable):
  - Format: JPEG / PNG / TIFF.
  - Quality (JPEG): slider.
  - Include originals: checkbox ("Also include uncorrected versions").
  - AI upscaling: toggle with 2× / 4× selector.
- File naming preview showing the pattern and an example filename.
- "Download All (ZIP)" button — large, prominent.
- Estimated file size shown next to the download button.
- Individual page downloads also available in a collapsible list below.

---

## 7. Platform Strategy & Phasing

### Phase 1 — Web App (MVP)

**Scope:** Responsive web application accessible via browser on any device.
- Full processing engine running on AWS.
- Capture via phone camera (using browser MediaDevices API).
- Capture via desktop webcam.
- File upload and batch processing.
- All core processing features (glare removal, photo splitting, perspective correction, color restoration).
- Export and download.

**Why web first:** Lowest barrier to entry (no install), works on all devices, fastest to ship.

### Phase 2 — Progressive Enhancements

- Improved mobile capture UX (installable PWA for home screen access, offline queue).
- Session sync across devices (capture on phone, review on desktop).
- Optional free account for persistent history.

### Phase 3 — Native Apps

**Mac Desktop App:**
- Direct integration with Mac/external webcam for continuous capture.
- Local pre-processing (photo detection, initial perspective correction) for faster feedback.
- Drag-and-drop from Finder.
- Integration with Photos.app for direct import of results.

**iOS App:**
- Optimized camera capture with real-time on-device glare detection.
- Local pre-processing using device GPU/Neural Engine.
- Photos.app integration.
- Offline mode: capture and queue for processing when online.

---

## 8. Non-Functional Requirements

### 8.1 Performance

- **Processing time:** Target < 15 seconds per album page (including all corrections) for server-side processing.
- **Real-time capture guidance:** Glare detection and page boundary overlays must run at ≥ 15 fps on-device.
- **Upload:** Support images up to 50 MB each. Handle batch uploads of 100+ images gracefully.
- **Concurrent users:** System should handle 100 concurrent processing sessions at launch, scaling to 1,000+.

### 8.2 Quality

- **Glare removal:** Processed photos should be free of visible glare artifacts in ≥ 90% of cases (single-shot mode) and ≥ 98% of cases (multi-shot mode).
- **Photo detection accuracy:** Correctly detect and split individual photos in ≥ 95% of standard album page layouts.
- **Color restoration:** Enhanced photos should look natural — no over-saturation, posterization, or color banding.
- **No data loss:** Processing should never discard image data. Originals are always preserved alongside processed versions.

### 8.3 Accessibility

- WCAG 2.1 AA compliance for the web UI.
- Screen reader support for all non-camera workflows.
- Keyboard navigation for all desktop interactions.
- High-contrast mode option.

### 8.4 Browser & Device Support

- **Browsers:** Chrome, Safari, Firefox, Edge (latest 2 versions each).
- **Mobile:** iOS 16+ Safari, Android Chrome.
- **Camera API:** Requires HTTPS for camera access. Graceful fallback to upload-only if camera is unavailable.

### 8.5 Localization

- English only for MVP.
- UI architecture supports future i18n (externalized strings, RTL-ready layout).

---

## 9. Success Metrics

### 9.1 Quality Metrics

- **Glare removal satisfaction rate:** % of processed photos where user does not manually re-adjust glare. Target: > 85%.
- **Auto-detection accuracy:** % of album pages where photo boundaries are correctly detected without manual override. Target: > 90%.
- **Processing success rate:** % of uploaded/captured images that complete processing without error. Target: > 99%.

### 9.2 Engagement Metrics

- **Batch completion rate:** % of users who start a batch and download at least one result. Target: > 70%.
- **Photos per session:** Average number of photos processed per session. Target: > 20.
- **Return rate:** % of users who come back for a second session. Target: > 30%.

### 9.3 Performance Metrics

- **Time to first result:** Time from first upload/capture to first viewable processed photo. Target: < 30 seconds.
- **Processing throughput:** Pages processed per minute per user. Target: > 4/min.

---

## 10. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Glare removal quality insufficient for glossy sleeves | High — core differentiator fails | Medium | Multi-shot mode as fallback; invest heavily in training data with glossy album pages specifically |
| AWS costs grow unsustainably with free users | High — threatens free model | Medium | Serverless architecture with aggressive auto-scaling down; per-session processing caps; monitor cost/photo closely |
| Phone camera quality varies wildly | Medium — inconsistent results | High | Strong capture guidance UX; minimum resolution warnings; graceful degradation of processing quality |
| Users don't understand multi-shot mode | Medium — miss best quality path | Medium | Clear in-app education; make multi-shot the default recommendation when glare is detected |
| Large batch uploads overwhelm backend | Medium — poor UX for power users | Low | Queue management with backpressure; per-user rate limits; progress communication |

---

## 11. Open Questions

1. Should we support video capture (user slowly pans across an album) as an alternative to individual page photos?
2. Should we add social/sharing features (share a digitized album as a link)?
3. Should we build in duplicate detection (same photo appears on multiple pages)?
4. How do we handle album pages with handwritten captions — preserve as separate images, attempt OCR, or ignore?
5. Should we offer a "print quality assessment" that tells users which originals are too degraded for good results?

---

## 12. Appendix

### A. Glossary

- **Album page:** A single page from a physical photo album, typically containing 1–4 photos behind a glossy plastic sleeve.
- **Glare/reflection:** Specular highlights caused by light reflecting off the glossy plastic sleeve.
- **Keystoning:** Trapezoidal distortion caused by photographing a flat surface at an angle.
- **Multi-shot compositing:** Taking multiple photos of the same page at different angles and combining them to eliminate glare.
- **Dewarping:** Correcting for non-planar distortion (e.g., a photo that bulges behind a sleeve).

### B. Competitive Reference

- Google PhotoScan: Best free option today but weak on glare (requires manual multi-angle shots with poor guidance) and no multi-photo splitting.
- Photomyne: Best photo splitting but expensive subscription and poor glare handling.
- CZUR scanners: Hardware-based overhead scanners with software — good quality but $200+ hardware investment.
- VueScan + flatbed: Gold standard quality but extremely slow (1–2 minutes per photo) and requires removing photos from albums.

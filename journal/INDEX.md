# Journal Index

Development log for Sunday Album. Entries fall into two categories:
- **Build log** — phase-by-phase summaries of what was built
- **Fixes & decisions** — post-phase algorithm changes, bug fixes, and decision records

---

## Build Log (phases)

| Date | File | Summary |
|------|------|---------|
| 2026-02-10 | [phase-1-project-setup](2026-02-10-phase-1-project-setup.md) | Project scaffold, HEIC/DNG loading, CLI skeleton, EXIF orientation |
| 2026-02-15 | [phase-2-page-detection](2026-02-15-phase-2-page-detection.md) | GrabCut page boundary detection, perspective correction |
| 2026-02-15 | [phase-3-glare-detection](2026-02-15-phase-3-glare-detection.md) | HSV-based glare detection, severity map, sleeve vs print classification |
| 2026-02-15 | [phase-4-glare-removal](2026-02-15-phase-4-glare-removal.md) | Single-shot OpenCV inpainting (intensity correction + TELEA/NS + contextual fill) |
| _(Phase 5 deferred)_ | — | Multi-shot glare compositing — requires multi-angle test images, not yet implemented |
| 2026-02-16 | [phase-6-photo-detection](2026-02-16-phase-6-photo-detection.md) | Contour-based photo detection, decoration filter, pipeline reorder (detect before glare) |
| 2026-02-16 | [phase-7-geometry-correction](2026-02-16-phase-7-geometry-correction.md) | Keystone, rotation detection, dewarp implementation |
| 2026-02-16 | [phase-8-color-restoration](2026-02-16-phase-8-color-restoration.md) | White balance, deyellowing, CLAHE-based fade restoration, sharpening |
| 2026-02-16 | [phase-9-pipeline-integration](2026-02-16-phase-9-pipeline-integration.md) | End-to-end pipeline wiring, AI quality check, output naming |

---

## Fixes & Decisions

| Date | File | Summary |
|------|------|---------|
| 2026-02-16 | [single-print-fix](2026-02-16-single-print-fix.md) | Fix: single-print false splits — adaptive max-area threshold + Canny edge fallback |
| 2026-02-17 | [multi-photo-detection-fix](2026-02-17-multi-photo-detection-fix.md) | Fix: three_pics detection — removed `_is_isolated_print` bypass, added projection-profile fallback |
| 2026-02-18 | [openai-glare-and-orientation](2026-02-18-openai-glare-and-orientation.md) | Feature: switched glare removal to OpenAI gpt-image-1 + added Claude AI orientation step |
| 2026-02-19 | [rotation-fix](2026-02-19-rotation-fix.md) | Fix: disabled Hough-line rotation (fires on content); clarified AI direction prompt |
| 2026-02-20 | [color-restoration-fix](2026-02-20-color-restoration-fix.md) | Fix: gray-world white balance tuning, conservative deyellowing |
| 2026-03-04 | [photo-boundary-detection-fix](2026-03-04-photo-boundary-detection-fix-and-background-stripping.md) | Fix: spine-separated album pages; background stripping improvements |
| 2026-04-09 | [color-restore-rethink](2026-04-09-color-restore-rethink.md) | Rework: replaced CLAHE with adaptive brightness lift + vibrance saturation |
| 2026-04-09 | [dev-branch-strategy](2026-04-09-dev-branch-strategy.md) | Decision: feature → dev → main branch model, dev as staging environment |

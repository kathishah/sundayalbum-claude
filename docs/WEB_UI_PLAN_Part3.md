# Sunday Album Web UI — Implementation Plan (Part 3 of 3)
# Phases 4–6: Step Detail, Re-processing, Prod Deployment

**Version:** 1.4
**Date:** March 2026
**Status:** Not started — pending Phase 3 completion
**See also:** WEB_UI_PLAN_Part1.md (Phases 0–2: completed), WEB_UI_PLAN_Part2.md (Phase 3: dev deployment)

---

## Phase 4: Step Detail Views

Replicate `mac-app/SundayAlbum/Views/StepDetailView.swift` and all step-specific views.

### 4.1 StepDetailLayout

3-pane layout: breadcrumb (top), StepTree (left 196px), StepCanvas (right).

### 4.2 Step-Specific Views

| View | macOS Reference | Key Interactions |
|------|----------------|------------------|
| PageDetectionView | `Views/Steps/PageDetectionStepView.swift` | SVG overlay with draggable corner handles |
| PhotoSplitView | `Views/Steps/PhotoSplitStepView.swift` | Colored region rectangles |
| OrientationView | `Views/Steps/OrientationStepView.swift` | Rotation picker (0/90/180/270) + scene desc editor |
| GlareRemovalView | `Views/Steps/GlareRemovalStepView.swift` | Before/after with `.saReveal` animation |
| ColorCorrectionView | `Views/Steps/ColorCorrectionStepView.swift` | Sliders (brightness, saturation, warmth, sharpness) |
| ResultsView | `Views/Steps/ResultsStepView.swift` | Photo grid + ComparisonView + export/download |

All images loaded from S3 via presigned URLs from `GET /jobs/{jobId}` response (which includes `debug_keys` map).

### 4.3 Verification

- Navigate to step detail for a completed job
- Click through all steps in the tree, see corresponding images
- Verify animations match macOS app timing

---

## Phase 5: Re-processing + Polish

### 5.1 Re-process from Step

- `POST /jobs/{jobId}/reprocess { from_step, photo_index, overrides }`
- Backend starts a new Step Functions execution with `start_from` parameter
- State machine uses Choice states to skip steps before `start_from`
- Overrides (e.g., adjusted corners, rotation change) saved to `debug/{stem}_overrides.json`

### 5.2 Wire Up Interactive Controls

- PageDetectionView corner drag → reprocess from perspective
- OrientationView rotation change → reprocess from ai_orient
- ColorCorrectionView slider change → reprocess from color_restore

### 5.3 Polish

- Error handling: retry buttons, error messages per step
- Mobile responsive: library + step detail adapt to smaller screens
- Loading states and skeleton placeholders for images

### 5.4 Verification

- Adjust corners in PageDetectionView, see pipeline re-run from perspective step
- Change rotation, see glare removal re-run with correct orientation

---

## Phase 6: Production Hardening

- CloudFront distribution for frontend (S3 static hosting)
- CORS configuration on API Gateway
- Per-user rate limiting (3 concurrent jobs, 50 photos/day)
- CloudWatch dashboards + alarms
- "Delete my data" endpoint
- Provisioned concurrency on critical Lambdas (optional, for cold start mitigation)

---

## Prod Deployment: app.sundayalbum.com

**Prerequisite:** Phase 3 dev verification complete (3.7 checklist all green).

### Prod App Runner Service

- Service name: `sundayalbum-web-prod`
- URL: `https://yjb7t3tngm.us-west-2.awsapprunner.com` (deployed, awaiting custom domain)
- Environment variables: `NEXT_PUBLIC_API_URL=https://e1f6o1ah49.execute-api.us-west-2.amazonaws.com`

### Custom Domain Setup

1. Associate `app.sundayalbum.com` on `sundayalbum-web-prod` App Runner service
2. Add ACM cert CNAME validation record to Route 53 hosted zone
3. Add Route 53 ALIAS record: `app.sundayalbum.com` → prod App Runner domain

### Branch Strategy (for prod releases)

- Feature branches off `web-ui-implementation` → PR to `web-ui-implementation` (deploys to dev, integration test)
- When dev is verified → PR `web-ui-implementation` → `main` (deploys to prod)
- Backend (CDK) deployments remain manual

### Prod CDK Stack

- Stack name: `SundayAlbumStack` (no suffix — existing prod resources)
- API Gateway: `https://e1f6o1ah49.execute-api.us-west-2.amazonaws.com`
- WebSocket: `wss://l7t59cvhyh.execute-api.us-west-2.amazonaws.com/$default`
- S3 bucket: `sundayalbum-data-680073251743-us-west-2`

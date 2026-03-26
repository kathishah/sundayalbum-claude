# Sunday Album Web UI — Implementation Plan

**Version:** 1.3
**Date:** March 2026
**Status:** Phase 2 complete — branch `web-ui-implementation`
**Companion Documents:** PRD_Album_Digitizer.md, UI_Design_Album_Digitizer.md, PHASED_PLAN_Claude_Code.md

---

## Context

Sunday Album has a feature-complete Python CLI pipeline and a native macOS SwiftUI app. We're building a web UI backed by AWS Lambda (one per pipeline step), Step Functions orchestration, and S3 storage. Before building the web layer, we refactor the existing codebase to support pluggable storage backends so the CLI, macOS app, and web all share the same pipeline code.

**Key decisions:**
- **Refactor first** — abstract storage I/O so existing CLI + macOS app keep working
- One Lambda per pipeline step (~10 Lambdas) for maximum granularity
- Full step-detail UI (replicating macOS app) from the start
- Next.js 14+ (App Router) + Tailwind CSS + Framer Motion
- HEIC/JPEG/PNG only (no DNG on web)
- Email-based auth with verification codes via SES
- Flat S3 structure: `{user_hash}/uploads/`, `{user_hash}/debug/`, `{user_hash}/output/`
- No job_id subfolders — filenames use input stem as prefix, DynamoDB tracks job→file mapping

---

## Phase 0: Refactor Existing Codebase for Pluggable Storage ✅ COMPLETE (2026-03-25)

**Goal:** Abstract file I/O so the same pipeline code works with local filesystem (CLI/macOS) and S3 (web). Each step becomes independently callable with clear serializable I/O. CLI and macOS app continue to work unchanged.

### 0.1 Storage Backend Abstraction

Create `src/storage/` module:

```
src/storage/
  __init__.py           # exports StorageBackend, LocalStorage, get_backend()
  backend.py            # StorageBackend protocol (ABC)
  local.py              # LocalStorage implementation (current behavior)
  s3.py                 # S3Storage implementation (new, for Lambda)
```

**`src/storage/backend.py`** — Protocol defining the I/O contract:

```python
class StorageBackend(Protocol):
    def read_image(self, key: str) -> np.ndarray:
        """Read image from storage. Returns float32 RGB [0,1]."""
        ...

    def write_image(self, key: str, image: np.ndarray,
                    format: str = "png", quality: int = 95) -> str:
        """Write image to storage. Returns the key/path written."""
        ...

    def read_json(self, key: str) -> dict:
        """Read JSON metadata from storage."""
        ...

    def write_json(self, key: str, data: dict) -> str:
        """Write JSON metadata to storage."""
        ...

    def exists(self, key: str) -> bool:
        ...

    def get_url(self, key: str) -> str:
        """Get a URL for the file (file:// for local, presigned for S3)."""
        ...
```

**`src/storage/local.py`** — Wraps current `Path`-based I/O. `key` is a relative path resolved against a `base_dir`. This is what the CLI and macOS app use. No behavior change.

**`src/storage/s3.py`** — Uses boto3. `key` is an S3 object key under a `prefix` (e.g., `{user_hash}/`). `get_url()` returns presigned URLs. Created later in Phase 2 but the interface is defined now.

### 0.2 Refactor Pipeline to Use StorageBackend

**Modify `src/pipeline.py`:**

- `Pipeline.__init__()` accepts an optional `storage: StorageBackend` parameter (defaults to `LocalStorage(base_dir)`)
- Each step in `process()` reads input from and writes output to `storage` using key conventions
- Key naming uses input stem as prefix: `debug/{stem}_01_loaded.jpg`, `output/SundayAlbum_{stem}_Photo01.jpg`
- Remove hardcoded `Path` operations from the pipeline body — delegate to storage backend

**Key convention (flat, stem-prefixed):**

```
uploads/{original_filename}              # e.g., uploads/IMG_cave_normal.HEIC
debug/{stem}_01_loaded.jpg               # e.g., debug/IMG_cave_normal_01_loaded.jpg
debug/{stem}_02_normalized.jpg
debug/{stem}_03_page_detected.jpg
debug/{stem}_04_page_corrected.jpg
debug/{stem}_05_photo_boundaries.jpg
debug/{stem}_05_photo_01_raw.jpg
debug/{stem}_05b_photo_01_oriented.jpg
debug/{stem}_07_photo_01_deglared.jpg
debug/{stem}_14_photo_01_enhanced.jpg
output/SundayAlbum_{stem}_Photo01.jpg    # final output
```

Intermediate data between steps (metadata, not images displayed to user):

```
debug/{stem}_03_page_detection.json      # corners, confidence, quads
debug/{stem}_05_photo_detections.json    # bounding boxes, corners per photo
debug/{stem}_05b_photo_01_analysis.json  # rotation, scene_description
```

### 0.3 Refactor `save_debug_image`

**Modify `src/utils/debug.py`:**

- `save_debug_image()` accepts a `StorageBackend` instead of a raw `Path`
- Signature: `save_debug_image(image, key, storage, description=None, quality=95)`
- Internally converts float32→uint8→JPEG bytes, then calls `storage.write_image()`

### 0.4 Extract Step Functions

Each pipeline step should be callable as a standalone function that:
1. Reads its input from storage (image + metadata from previous step)
2. Runs the processing
3. Writes its output to storage (image + metadata for next step)
4. Returns a result dict (metadata summary, not the image itself)

Create `src/steps/` module that wraps existing processing functions:

```
src/steps/
  __init__.py
  load.py               # wraps src/preprocessing/loader.load_image
  normalize.py           # wraps src/preprocessing/normalizer.normalize
  page_detect.py         # wraps src/page_detection/detector.detect_page
  perspective.py         # wraps src/page_detection/perspective.correct_perspective
  photo_detect.py        # wraps src/photo_detection/detector.detect_photos
  photo_split.py         # wraps src/photo_detection/splitter.split_photos
  ai_orient.py           # wraps src/ai/claude_vision.analyze_photo_for_processing
  glare_remove.py        # wraps src/glare/remover_openai.remove_glare_openai
  geometry.py            # wraps src/geometry (passthrough currently)
  color_restore.py       # wraps src/color (all 4 sub-steps)
```

Each step module exports a `run()` function:

```python
def run(storage: StorageBackend, stem: str, config: PipelineConfig,
        photo_index: Optional[int] = None) -> dict:
    """Run this pipeline step.

    Reads input from storage using key conventions.
    Writes output to storage using key conventions.
    Returns metadata dict (not image data).
    """
```

### 0.5 Refactor `Pipeline.process()` to Use Steps

Rewrite the body of `Pipeline.process()` to call `steps.load.run(storage, stem, config)`, `steps.normalize.run(storage, stem, config)`, etc. in sequence. This is a pure refactor — same behavior, same output, but using the storage abstraction and step functions.

### 0.6 Verify CLI + macOS App Still Work ✅

- Run `python -m src.cli process test-images/IMG_cave_normal.HEIC --output ./output/ --no-openai-glare --no-ai-orientation`
- All debug images written flat to `debug/` with stem prefix (e.g. `IMG_cave_normal_01_loaded.jpg`)
- Output JPEG written to `output/SundayAlbum_IMG_cave_normal_Photo01.jpg`
- `pytest tests/ -v` — 72 passed, 6 skipped ✅

**Implementation notes vs original plan:**
- `src/utils/debug.py` was NOT modified — pipeline steps now write to storage directly; `save_debug_image` is only used by CLI utility commands (compare, check).
- Debug images are always written (even without `--debug` flag) because intermediate images serve as inter-step state. This is intentional for the storage-driven architecture.
- Output filenames always use `_PhotoNN` suffix (e.g. `SundayAlbum_{stem}_Photo01.jpg`), even for single photos, matching the S3 key convention.

**Files added/modified in Phase 0:**
- New: `src/storage/__init__.py`, `src/storage/backend.py`, `src/storage/local.py`
- New: `src/steps/__init__.py`, `src/steps/load.py`, `src/steps/normalize.py`, `src/steps/page_detect.py`, `src/steps/perspective.py`, `src/steps/photo_detect.py`, `src/steps/photo_split.py`, `src/steps/ai_orient.py`, `src/steps/glare_remove.py`, `src/steps/geometry.py`, `src/steps/color_restore.py`
- Modified: `src/pipeline.py` — uses StorageBackend + step functions
- Modified: `src/cli.py` — creates LocalStorage per file, passes to pipeline
- Modified: `.gitignore` — added `uploads/`

---

## Phase 1: AWS Infrastructure + Auth + Upload ✅ COMPLETE (2026-03-25)

### 1.1 S3 Bucket: `sundayalbum-data`

Flat structure per user:

```
{user_hash}/
  uploads/                    # original uploaded files
  debug/                      # debug images + intermediate metadata JSONs
  output/                     # final output JPEGs
```

- `user_hash` = SHA-256 of lowercase email
- Files are prefixed with input stem to avoid collisions
- DynamoDB `sa-jobs` tracks which files belong to which job
- Lifecycle policies: uploads/ 30 days, debug/ 7 days, output/ 30 days

### 1.2 DynamoDB Tables

**`sa-sessions`** — email auth
```
PK: email (string)
code: string (6-digit)
code_expires_at: number (Unix timestamp, 10min TTL)
session_token: string (UUID)
token_expires_at: number (Unix timestamp, 7-day TTL)
user_hash: string (SHA-256)
```

**`sa-jobs`** — job tracking
```
PK: user_hash (string)
SK: job_id (string, ULID for ordering)
status: string (uploading | processing | complete | failed)
current_step: string (load | normalize | page_detect | ... | done)
step_detail: string (e.g. "Photo 2 of 3: glare removal")
input_filename: string
input_stem: string                  # used as file prefix in S3
photo_count: number
output_keys: List[string]           # S3 keys of final output JPEGs
debug_keys: Map[string, string]     # step_name → S3 key for debug images
created_at: string (ISO)
updated_at: string (ISO)
error_message: string (optional)
processing_time: number (optional)
execution_arn: string (Step Functions ARN)
ttl: number (30 days)
```

**`sa-ws-connections`** — WebSocket tracking
```
PK: connection_id
job_id: string
user_hash: string
```

### 1.3 Auth Flow

1. User enters email → `POST /auth/send-code`
2. Backend generates 6-digit code, stores in `sa-sessions` (10min TTL), sends via SES
3. User enters code → `POST /auth/verify`
4. Backend validates, creates session_token (UUID, 7-day TTL), returns `{ session_token, user_hash }`
5. Frontend stores token in localStorage
6. Rate limits: 3 code sends/email/hour, 5 verify attempts/code

### 1.4 API Endpoints (API Gateway HTTP API)

```
POST   /auth/send-code         { email }
POST   /auth/verify             { email, code } → { session_token, user_hash }
POST   /auth/logout

GET    /jobs                    → list user's jobs
POST   /jobs                    { filename, format, size } → { job_id, upload_url }
GET    /jobs/{jobId}            → job status + presigned URLs for debug/output images
DELETE /jobs/{jobId}            → cancel/delete

POST   /jobs/{jobId}/start      → trigger Step Functions after upload
POST   /jobs/{jobId}/reprocess  { from_step, photo_index?, overrides? }
```

### 1.5 API Lambda Handlers

New: `api/` directory at project root:
- `api/auth.py` — send-code, verify, logout (SES + DynamoDB)
- `api/jobs.py` — CRUD, presigned URLs, trigger Step Functions
- `api/websocket.py` — connect, disconnect, subscribe, broadcast

### 1.6 Infrastructure as Code

AWS CDK stack defining all resources. New: `infra/` directory.

### 1.7 Verification ✅

- Sign in with email, receive code, verify ✅
- Create job, get presigned upload URL, upload HEIC file ✅
- Confirm file lands in S3 at `{user_hash}/uploads/{filename}` ✅
- POST /jobs/{jobId}/start responds with processing status ✅
- GET /jobs/{jobId} returns job state with correct metadata ✅
- GET /jobs lists all user jobs in reverse-ULID order ✅
- POST /auth/logout revokes token; subsequent requests return 401 ✅

**Implementation notes:**
- CDK stack name: `SundayAlbumStack` in `infra/infra/sundayalbum_stack.py`
- API base URL: `https://e1f6o1ah49.execute-api.us-west-2.amazonaws.com`
- S3 bucket: `sundayalbum-data-680073251743-us-west-2`
- DynamoDB tables: `sa-sessions` (GSI: `token-index`), `sa-jobs` (DynamoDB Streams enabled), `sa-ws-connections` (GSI: `job-index`)
- Lambda code in `api/` directory — shared zip across all 3 functions
- SES sender uses verified identity; configured via `ses_sender_email` CDK context key
- S3 client uses regional endpoint (`s3.{REGION}.amazonaws.com`) + SigV4 to avoid 307 redirects on presigned PUT URLs
- `session_token` omitted from DynamoDB Item when not yet set (GSI key cannot be empty string)
- `POST /jobs/{jobId}/start` is a no-op placeholder until Step Functions state machine is added in Phase 2

**Files added in Phase 1:**
- New: `infra/app.py` (updated), `infra/infra/sundayalbum_stack.py`
- New: `api/__init__.py`, `api/common.py`, `api/auth.py`, `api/jobs.py`, `api/websocket.py`

---

## Phase 2: Pipeline Lambdas + Step Functions ✅ COMPLETE (2026-03-26)

### 2.1 Lambda Container Image

Single Dockerfile, shared by all 10 Lambdas (different handler entry points):

```dockerfile
FROM public.ecr.aws/lambda/python:3.12
RUN dnf install -y libheif-devel && dnf clean all
COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY handlers/ ${LAMBDA_TASK_ROOT}/handlers/
```

`requirements-lambda.txt` = production deps only (exclude torch, diffusers, transformers, pytest, mypy, ruff, matplotlib, rawpy).

### 2.2 Lambda Handlers

New: `handlers/` directory. Each handler:
1. Receives event from Step Functions: `{ user_hash, job_id, stem, config, photo_index? }`
2. Creates `S3Storage(bucket, prefix=user_hash)`
3. Calls `steps.{step_name}.run(storage, stem, config, photo_index)`
4. Updates DynamoDB job status via `handlers/common.py`
5. Returns result dict for Step Functions

```
handlers/
  common.py               # S3Storage init, DynamoDB status updates, config parsing
  load.py                 # → steps.load.run()
  normalize.py            # → steps.normalize.run()
  page_detect.py          # → steps.page_detect.run()
  perspective.py          # → steps.perspective.run()
  photo_detect.py         # → steps.photo_detect.run()
  photo_split.py          # → steps.photo_split.run()
  ai_orient.py            # → steps.ai_orient.run()
  glare_remove.py         # → steps.glare_remove.run()
  geometry.py             # → steps.geometry.run()
  color_restore.py        # → steps.color_restore.run()
```

### 2.3 S3Storage Implementation

New: `src/storage/s3.py` — implements `StorageBackend` protocol using boto3.

### 2.4 Step Functions State Machine

```
sa-load → sa-normalize → sa-page-detect → sa-perspective →
sa-photo-detect → sa-photo-split →
ProcessPhotosMap (parallel per photo):
  sa-ai-orient → sa-glare-remove → sa-geometry → sa-color-restore
→ Complete
```

Each state updates DynamoDB `sa-jobs.current_step`. DynamoDB Streams trigger WebSocket broadcaster.

### 2.5 Lambda Specifications

| Lambda | Handler | RAM | Timeout | Notes |
|--------|---------|-----|---------|-------|
| `sa-load` | `handlers/load.handler` | 2048MB | 60s | Loads HEIC/JPEG/PNG from S3 |
| `sa-normalize` | `handlers/normalize.handler` | 1024MB | 30s | Resize + thumbnail |
| `sa-page-detect` | `handlers/page_detect.handler` | 1024MB | 60s | GrabCut segmentation |
| `sa-perspective` | `handlers/perspective.handler` | 1024MB | 30s | Homographic warp |
| `sa-photo-detect` | `handlers/photo_detect.handler` | 1024MB | 60s | Contour detection |
| `sa-photo-split` | `handlers/photo_split.handler` | 1024MB | 30s | Extract photo crops |
| `sa-ai-orient` | `handlers/ai_orient.handler` | 512MB | 30s | Claude Haiku API call |
| `sa-glare-remove` | `handlers/glare_remove.handler` | 1024MB | 120s | OpenAI API call (slow) |
| `sa-geometry` | `handlers/geometry.handler` | 1024MB | 30s | Passthrough currently |
| `sa-color-restore` | `handlers/color_restore.handler` | 1024MB | 30s | WB + deyellow + CLAHE + sharpen |

### 2.6 Verification ✅

- Trigger pipeline via `POST /jobs/{jobId}/start` for `IMG_three_pics_normal.HEIC` ✅
- All 11 Lambda steps run in sequence, per-photo steps fan out in Map state (MaxConcurrency=4) ✅
- 3 photos detected and processed ✅
- Output JPEGs appear at `{user_hash}/output/SundayAlbum_{stem}_PhotoNN.jpg` ✅
- DynamoDB job record shows `status: complete`, `photo_count: 3`, `processing_time: 44s` ✅
- Presigned GET URLs in `GET /jobs/{jobId}` response work for all 3 output photos ✅

**Implementation notes:**
- Lambda container image built from repo root with `--platform linux/arm64 --provenance=false` (OCI attestation manifests are not supported by Lambda)
- Docker CMD override set via `DockerImageCode.from_image_asset(cmd=[handler])` — CDK deduplicates the image build; each function gets its own Lambda imageConfig.command override
- Step Functions `OutputPath: "$.Payload"` unwraps the Lambda response wrapper so each handler receives `{user_hash, job_id, stem, ...}` directly (not `{Payload: {...}}`)
- Memory raised to 3008MB for all image-processing Lambdas (24MP HEIC decoding requires ~600MB float32 array + Python overhead; 1024MB caused OOM in Load step)
- `start_time` and `config` must be included in Step Functions execution input from `_handle_start()` — the PrepareMap Pass state passes them through to each per-photo task
- `finalize_job()` prefixes `output_keys` with `{user_hash}/` so DynamoDB stores full S3 paths; `jobs.py` `presign_get()` can then generate correct presigned URLs
- Step Functions state machine ARN: `arn:aws:states:us-west-2:680073251743:stateMachine:sa-pipeline`
- ECR repo: `680073251743.dkr.ecr.us-west-2.amazonaws.com/cdk-hnb659fds-container-assets-680073251743-us-west-2`

**Files added in Phase 2:**
- New: `src/storage/s3.py` — S3Storage backend (boto3, regional endpoint, SigV4)
- New: `handlers/__init__.py`, `handlers/common.py`, `handlers/load.py`, `handlers/normalize.py`, `handlers/page_detect.py`, `handlers/perspective.py`, `handlers/photo_detect.py`, `handlers/photo_split.py`, `handlers/ai_orient.py`, `handlers/glare_remove.py`, `handlers/geometry.py`, `handlers/color_restore.py`, `handlers/finalize.py`
- New: `Dockerfile`, `requirements-lambda.txt`, `.dockerignore`
- Modified: `infra/infra/sundayalbum_stack.py` — added 11 container Lambda functions + Step Functions state machine
- Modified: `api/jobs.py` — `_handle_start()` now sends `start_time` + `config` in execution input

---

## Phase 3: Real-time Progress + Library UI

### Hosting Decision (decided 2026-03-26)

**Choice: AWS App Runner** for the Next.js frontend.

**Options considered:**

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Amplify Hosting** | Native Next.js SSR, Git CI/CD, free tier | SSR on Lambda@Edge (cold starts); Amplify Gen1/Gen2 doc confusion; SSR untested in this account — only `caboose` (static HTML/JS) is deployed on Amplify here | Rejected |
| **App Runner** | Already proven in this account (`workorder-invoice` → `veritasa.ai`); Docker-based like pipeline Lambdas; same ECR repo; no cold starts; managed HTTPS | ~$15-20/month minimum; manual CI/CD setup | **Selected** |
| **EC2 + Nginx** | Full control, simple to reason about | Manual OS/cert/PM2 management; single point of failure; `vix-trading-monitor` and `SuperTrendBot` on EC2 are bots, not web frontends — no proven pattern for web | Rejected |
| **CloudFront + S3** | Pennies/month | Requires `output: 'export'`; loses SSR, server components, streaming; wrong for dynamic auth + real-time app | Rejected |

**Why App Runner won:** The `workorder-invoice` app on App Runner is already serving `veritasa.ai` with a custom domain and ACM cert — the exact pattern needed here. No unknowns. The pipeline Lambdas already use ECR, so the Docker build and push workflow is established. Amplify's SSR support is real but untested in this account.

**Environments (decided 2026-03-26):**

Two environments only — dev and prod. No local Docker Compose backend stack.

| Environment | Branch | App URL | API | Purpose |
|-------------|--------|---------|-----|---------|
| **Dev** | `dev` | `dev.sundayalbum.com` | dev API Gateway | Development + testing against real AWS |
| **Prod** | `main` | `app.sundayalbum.com` | prod API Gateway | Live users |

**Rationale:** Sunday Album is being built for public distribution, not just personal use. Dev needs to run against real AWS services (real DynamoDB, real S3, real Lambda) to catch AWS-specific issues before they reach prod users — the Phase 2 bugs (presigned URL 307 redirects, OCI manifest format) would not have been caught by a local emulator. Docker on the dev Mac is used only to build and test the Next.js container image before pushing to ECR, not to run a local backend stack.

**Local development workflow:**
- `npm run dev` in `web/` → Next.js dev server on `localhost:3000`
- `.env.local` → `NEXT_PUBLIC_API_URL` points at dev API Gateway URL
- Real dev DynamoDB, real dev S3, real dev Lambda — all suffixed `-dev` via CDK `stage` context
- Docker used only for `docker build` + `docker run` to verify the container before pushing

**CDK stage parameterization:**
- Single CDK stack parameterized with `stage` context (default: `prod`)
- `cdk deploy --context stage=dev` → all resources suffixed `-dev` (tables, bucket, Lambdas, state machine)
- `cdk deploy --context stage=prod` → production resources (current naming)
- Each stage is fully isolated: separate S3 bucket, DynamoDB tables, Lambda functions, API Gateway, Step Functions

**Domain setup:**
- `sundayalbum.com` registered on Namecheap (no hosting configured yet)
- Create Route 53 hosted zone for `sundayalbum.com`
- Update Namecheap custom DNS to Route 53 NS records (one-time)
- `app.sundayalbum.com` → prod App Runner service
- `dev.sundayalbum.com` → dev App Runner service (or App Runner default URL is fine for dev)
- Future: `sundayalbum.com` and `www.sundayalbum.com` → public marketing site (separate repo, Phase 6+)

**Infrastructure:**
- Separate ECR repo: `sundayalbum-web` (distinct from CDK pipeline asset repo)
- App Runner service per stage: `sundayalbum-web-prod`, `sundayalbum-web-dev`
- Each service: 0.25 vCPU / 0.5 GB to start, auto-scale to 1 vCPU / 2 GB
- Environment variables per stage: `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_REGION`, `NEXT_PUBLIC_S3_BUCKET`

**CI/CD (GitHub Actions):**
- Push to `dev` branch → build image → push to ECR → deploy to `sundayalbum-web-dev`
- Push to `main` branch → build image → push to ECR → deploy to `sundayalbum-web-prod`
- Backend (CDK) deployments remain manual (`cdk deploy`) — infrastructure changes are deliberate, not automatic

### 3.1 WebSocket Progress (backend)

Wire up the DynamoDB Streams → Lambda → WebSocket push path that is already stubbed in `api/websocket.py`:

- Add DynamoDB Stream trigger on `sa-jobs` table → broadcaster Lambda
- Broadcaster queries `sa-ws-connections` GSI `job-index`, pushes step update to all connected clients via `apigatewaymanagementapi`
- Add API Gateway WebSocket API alongside existing HTTP API (separate CDK construct)

Progress message format:
```json
{
  "type": "step_update",
  "jobId": "...",
  "step": "glare_remove",
  "detail": "Photo 2 of 3: glare removal",
  "progress": 0.55
}
```

Frontend fallback: poll `GET /jobs/{jobId}` every 3s if WebSocket is unavailable.

### 3.2 Next.js App Scaffold

```
web/
  app/
    layout.tsx              # root layout, fonts, providers
    page.tsx                # redirect → /library or /login
    login/page.tsx          # auth flow
    library/page.tsx        # main library view
    library/[jobId]/page.tsx  # step detail (Phase 4)
  components/
    AlbumPageCard.tsx       # job card (before → progress → after)
    DropZone.tsx            # drag-drop upload
    ProgressWheel.tsx       # animated step progress
    AuthForm.tsx            # email + code entry
  lib/
    api.ts                  # typed wrappers around API Gateway endpoints
    store.ts                # Zustand: jobs, auth, WebSocket
    useJobProgress.ts       # WebSocket hook with polling fallback
  Dockerfile                # FROM node:20-alpine, next build + next start
  next.config.ts
  tailwind.config.ts
```

### 3.3 Design System

Translate from `mac-app/SundayAlbum/Theme/DesignSystem.swift` into Tailwind config:
- Colors: `sa-amber-{50..700}`, `sa-stone-{50..950}`, `sa-success`, `sa-error`
- Fonts: Fraunces (display), DM Sans (body) via `next/font`
- Animations: `sa-standard` (200ms ease), `sa-slide` (350ms ease-out), `sa-reveal` (600ms ease-in-out)

### 3.4 Library Page

Replicate `mac-app/SundayAlbum/Views/LibraryView.swift`:

- Adaptive grid of `AlbumPageCard` components
- `DropZone` when library is empty (drag-drop + "Choose Files" button)
- Cards show: before thumbnail → animated progress wheel with step label → output thumbnails grid
- Single-click → expanded overlay; double-click → step detail (Phase 4)
- Real-time updates: WebSocket event → Zustand store → card re-renders

### 3.5 Auth Pages

- `/login` — email input → send code → 6-digit code entry → redirect to `/library`
- Session token in `localStorage`; auth guard on `/library` and `/library/[jobId]`

### 3.6 Verification

- `sundayalbum.com` resolves to App Runner service ✓
- Login flow works end-to-end (email → code → session) ✓
- Upload HEIC from browser → card enters processing state ✓
- Step labels update in real-time as pipeline runs ✓
- Output thumbnails appear when job completes ✓
- Presigned URLs load correctly in `<img>` tags ✓

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

## Frontend File Structure

```
web/
  src/
    app/
      layout.tsx                    # root: fonts, providers
      page.tsx                      # landing → /login or /library
      (auth)/login/page.tsx         # email + code verification
      (app)/
        layout.tsx                  # authenticated wrapper
        library/page.tsx            # library grid
        jobs/[jobId]/page.tsx       # step detail

    components/
      ui/                           # Button, Input, Slider, Card, ProgressWheel
      auth/                         # EmailForm, CodeVerification
      library/
        DropZone.tsx
        AlbumPageCard.tsx
        ExpandedCard.tsx
      step-detail/
        StepDetailLayout.tsx
        StepTree.tsx
        StepCanvas.tsx
        ActionBar.tsx
        ReprocessBar.tsx
      steps/
        PageDetectionView.tsx
        PhotoSplitView.tsx
        OrientationView.tsx
        GlareRemovalView.tsx
        ColorCorrectionView.tsx
        ResultsView.tsx
        ComparisonView.tsx
      shared/
        ImagePane.tsx
        BeforeAfterReveal.tsx

    lib/
      api.ts
      auth.ts
      websocket.ts
      s3-upload.ts
      types.ts
      constants.ts

    stores/
      auth-store.ts
      jobs-store.ts
      step-store.ts

    hooks/
      useJobProgress.ts
      usePresignedUpload.ts
      useStepImages.ts
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| S3 round-trips between every Lambda add ~1-2s overhead per step | Accept for granularity; intermediate PNGs are <5MB |
| Lambda cold starts (5-15s for container) | Provisioned concurrency or show "warming up" message |
| HEIC can't be previewed in `<img>` tags | Use `debug/{stem}_02_normalized.jpg` thumbnail as "before" preview |
| OpenAI rate limits with parallel per-photo calls | Step Functions Map MaxConcurrency:4 + existing backoff logic |
| `pillow-heif` on Lambda Linux | Test container build early in Phase 1; fallback: convert on upload |
| Phase 0 refactor breaks CLI/macOS | Run existing tests after each refactor step; compare debug output |

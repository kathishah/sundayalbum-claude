# Sunday Album — System Architecture

**Last updated:** 2026-04-10  
**Status:** Current — this is the canonical reference for system design, components, and SDLC.

---

## Overview

Sunday Album digitizes physical photo album pages. A phone camera photo of one or more glossy prints goes in; clean, individually corrected digital photos come out.

The system has three delivery surfaces sharing one image processing pipeline:

| Surface | Status | Entry point |
|---|---|---|
| **Python CLI** | Production | `python -m src.cli process ...` |
| **macOS app** | Production | `mac-app/` — SwiftUI shell over Python CLI |
| **Web app** | Production (dev), Production (prod) | Next.js → AWS Lambda pipeline |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Client surfaces                                                         │
│                                                                          │
│  ┌──────────────┐   ┌──────────────────────┐   ┌─────────────────────┐ │
│  │  Python CLI  │   │   macOS SwiftUI App  │   │  Next.js Web App    │ │
│  │  (local)     │   │   (Python CLI bridge)│   │  (App Runner)       │ │
│  └──────┬───────┘   └──────────┬───────────┘   └──────────┬──────────┘ │
│         │                      │                           │            │
└─────────┼──────────────────────┼───────────────────────────┼────────────┘
          │                      │                           │
          ▼                      ▼                           ▼
┌──────────────────────┐                        ┌───────────────────────────┐
│  Local filesystem    │                        │  AWS API Gateway (REST)   │
│  StorageBackend      │                        │  + WebSocket              │
└──────────────────────┘                        └───────────────┬───────────┘
                                                                │
                                                                ▼
                                                ┌───────────────────────────┐
                                                │  API Lambda Functions     │
                                                │  (zip, Python)            │
                                                │  sa-auth / sa-jobs /      │
                                                │  sa-websocket /           │
                                                │  sa-settings /            │
                                                │  sa-broadcaster           │
                                                └───────────────┬───────────┘
                                                                │
                                                                ▼
                                                ┌───────────────────────────┐
                                                │  AWS Step Functions       │
                                                │  sa-pipeline state machine│
                                                └───────────────┬───────────┘
                                                                │
                                                    ┌───────────┴───────────┐
                                                    │  Pipeline Lambda Steps │
                                                    │  (Docker/ARM64)        │
                                                    │  11 Lambdas, one each  │
                                                    └───────────────────────┘
```

---

## Image Processing Pipeline

The core pipeline is implemented as a sequence of pure-function steps in `src/steps/`. Each step has the same signature:

```python
def run(storage: StorageBackend, stem: str, config: PipelineConfig,
        photo_index: Optional[int] = None) -> dict
```

The `StorageBackend` abstraction (`src/storage/`) lets the same step code run unchanged in all three surfaces:
- **CLI / macOS app**: `LocalStorage` — reads/writes local filesystem paths under `debug/` and `output/`
- **Web / Lambda**: `S3Storage` — reads/writes S3 object keys under `{user_hash}/debug/` and `{user_hash}/output/`

### Pipeline Steps

| Step | Handler | What it does | Notes |
|---|---|---|---|
| `load` | `sa-pipeline-load` | Decode HEIC/JPEG/PNG from storage; apply EXIF orientation | 2048 MB RAM |
| `normalize` | `sa-pipeline-normalize` | Resize to working resolution (4000px max); thumbnail | |
| `page_detect` | `sa-pipeline-page-detect` | GrabCut segmentation → page boundary quadrilateral | |
| `perspective` | `sa-pipeline-perspective` | Homographic warp to fronto-parallel view | |
| `photo_detect` + `photo_split` | `sa-pipeline-photo-detect` / `photo-split` | Contour detection of individual photo boundaries; extract crops | |
| `ai_orient` | `sa-pipeline-ai-orient` | Claude Haiku — detect 90°/180°/270° rotation + get scene description | One API call per photo |
| `glare_remove` | `sa-pipeline-glare-remove` | OpenAI `gpt-image-1.5` semantic inpainting (default); OpenCV fallback | One API call per photo |
| `geometry` | `sa-pipeline-geometry` | Dewarp + small-angle rotation correction (both currently disabled — false positives) | |
| `color_restore` | `sa-pipeline-color-restore` | White balance → deyellowing → adaptive brightness lift + vibrance → sharpening | |
| `finalize` | `sa-pipeline-finalize` | Collect output keys; update job status to `complete` | |

### Resuming from a Step (Reprocessing)

The web app supports resuming the pipeline from any step via `POST /jobs/{jobId}/reprocess?from_step={step}`. Step Functions evaluates `start_from` in the event and skips earlier steps. The CLI supports `--steps load,normalize,...` to run only specific steps.

Per-photo steps (`ai_orient`, `glare_remove`, `color_restore`) can be rerun for a specific photo using `reprocess_photo_index`.

---

## Web Application (AWS)

### Environments

| Environment | Branch | App Runner | API Gateway | WebSocket |
|---|---|---|---|---|
| **dev** | `dev` | `sundayalbum-web-dev` → `dev.sundayalbum.com` | `https://nodcooz758.execute-api.us-west-2.amazonaws.com` | — |
| **prod** | `main` | `sundayalbum-web-prod` → `app.sundayalbum.com` | `https://e1f6o1ah49.execute-api.us-west-2.amazonaws.com` | `wss://l7t59cvhyh.execute-api.us-west-2.amazonaws.com/$default` |

### AWS Resources

**Account:** `680073251743` | **Region:** `us-west-2`

| Resource | Dev name | Prod name / ARN |
|---|---|---|
| **S3 bucket** | `sundayalbum-data-680073251743-us-west-2` | same (single bucket, path-separated by `user_hash`) |
| **DynamoDB: jobs** | `sa-jobs-dev` | `sa-jobs` |
| **DynamoDB: sessions** | `sa-sessions-dev` | `sa-sessions` |
| **DynamoDB: WebSocket** | `sa-ws-connections-dev` | `sa-ws-connections` |
| **DynamoDB: user settings** | `sa-user-settings-dev` | `sa-user-settings` |
| **Step Functions** | `sa-pipeline-dev` | `arn:aws:states:us-west-2:680073251743:stateMachine:sa-pipeline` |
| **ECR (pipeline image)** | `sundayalbum-pipeline-dev` (MUTABLE tags) | `cdk-hnb659fds-container-assets-680073251743-us-west-2` (IMMUTABLE tags) |
| **Secrets Manager** | `sundayalbum/api-keys-dev` | `sundayalbum/api-keys` |
| **CDK stack** | `SundayAlbumStack-dev` | `SundayAlbumStack` |

**S3 path layout:**
```
{user_hash}/uploads/{stem}.{ext}        ← original upload
{user_hash}/debug/{stem}_NN_*.jpg       ← intermediate debug images
{user_hash}/output/SundayAlbum_{stem}_PhotoNN.jpg  ← final output
```

### API Lambda Functions (zip-based, `api/` directory)

All deployed together as a shared zip. Entry points are individual files.

| Lambda | Handler | Purpose |
|---|---|---|
| `sa-auth{suffix}` | `api/auth.py` | Email OTP auth (send-code, verify, logout) via SES |
| `sa-jobs{suffix}` | `api/jobs.py` | Job CRUD, presigned S3 URLs, trigger Step Functions |
| `sa-websocket{suffix}` | `api/websocket.py` | WebSocket connect/disconnect/ping |
| `sa-settings{suffix}` | `api/settings.py` | Store/retrieve user API keys in DynamoDB |
| `sa-broadcaster{suffix}` | `api/broadcaster.py` | DynamoDB Streams → push step updates to WebSocket clients |

### Pipeline Lambda Functions (Docker/ARM64, `handlers/` + `src/` directories)

Single Dockerfile (`Dockerfile`) builds one image shared by all 11 Lambdas — each gets a different `CMD` override. Image tagged by git commit SHA.

| Lambda | Handler | RAM | Timeout |
|---|---|---|---|
| `sa-pipeline-load{suffix}` | `handlers/load.handler` | 3008 MB | 60s |
| `sa-pipeline-normalize{suffix}` | `handlers/normalize.handler` | 3008 MB | 60s |
| `sa-pipeline-page-detect{suffix}` | `handlers/page_detect.handler` | 3008 MB | 60s |
| `sa-pipeline-perspective{suffix}` | `handlers/perspective.handler` | 3008 MB | 60s |
| `sa-pipeline-photo-detect{suffix}` | `handlers/photo_detect.handler` | 3008 MB | 60s |
| `sa-pipeline-photo-split{suffix}` | `handlers/photo_split.handler` | 3008 MB | 60s |
| `sa-pipeline-ai-orient{suffix}` | `handlers/ai_orient.handler` | 3008 MB | 120s |
| `sa-pipeline-glare-remove{suffix}` | `handlers/glare_remove.handler` | 3008 MB | 300s |
| `sa-pipeline-geometry{suffix}` | `handlers/geometry.handler` | 3008 MB | 60s |
| `sa-pipeline-color-restore{suffix}` | `handlers/color_restore.handler` | 3008 MB | 120s |
| `sa-pipeline-finalize{suffix}` | `handlers/finalize.handler` | 1024 MB | 30s |

`{suffix}` is `-dev` for dev, empty for prod.

### Authentication Flow

Email-only, no passwords:
1. User enters email → `POST /auth/send-code` → SES sends 6-digit OTP to email
2. User enters OTP → `POST /auth/verify` → returns `session_token` (stored in localStorage)
3. All API calls include `Authorization: Bearer {session_token}`
4. `require_auth()` validates token → returns `user_hash` (SHA-256 of email, used as S3 path prefix)

Users can optionally store their own Anthropic and OpenAI API keys via `PUT /settings/api-keys`. Stored encrypted in DynamoDB `sa-user-settings`. Pipeline resolves API keys: user-supplied key > system key (Secrets Manager) > empty string (step falls back to OpenCV/pass-through).

### Live Progress (WebSocket)

1. DynamoDB Streams monitors `sa-jobs` table for step completion writes
2. `sa-broadcaster` Lambda consumes the stream → looks up WebSocket connection IDs for the job → sends step update messages
3. Frontend receives updates → advances progress wheel without page refresh

---

## macOS App

**Location:** `mac-app/`  
**Technology:** SwiftUI (native macOS), Python CLI bridge  
**Features:** Drag-drop album page input, step-by-step debug view, output photo export to Photos.app / Finder

The macOS app shells out to `python -m src.cli process ...` — the same CLI used for local development. No separate macOS-specific processing code. API keys are stored in macOS Keychain.

---

## SDLC — Branch Strategy & Deployments

### Branch Model

```
feature/my-change  ──PR──►  dev  ──PR──►  main
                              │              │
                              ▼              ▼
                           dev env        prod env
                      (auto-deploy)    (auto-deploy)
```

- **Feature work:** Create a branch from `dev` (e.g. `feature/my-change`). Open PR to `dev`.
- **Dev environment:** Every merge to `dev` automatically deploys to dev (App Runner `sundayalbum-web-dev` + dev Lambda functions).
- **Production release:** Open PR from `dev` → `main`. Merge triggers prod deploy.
- **Backend (CDK) changes:** Deployed manually via `cdk deploy` from the `infra/` directory. Not automated in CI.

### GitHub Actions Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `deploy-web.yml` | Push to `dev` or `main` with `web/**` changes | Build Next.js Docker image → push to ECR → deploy to App Runner (dev or prod) |
| `deploy-lambda.yml` | Push to `dev` or `main` with `api/**`, `handlers/**`, `src/**` changes | Zip `api/` → update API Lambdas. Build Docker image (linux/arm64) → push to ECR → update pipeline Lambdas |
| `test-api.yml` | Push to any branch with `api/**` or `tests/api|handlers/**` changes | Run 36 pytest tests (moto, no real AWS) |
| `test-web.yml` | Manual (`workflow_dispatch`) only | Run 12 Playwright E2E tests against `dev.sundayalbum.com` |

**IAM role:** `github-actions-sundayalbum` — used by all workflows via OIDC (no long-lived credentials).

### ECR Image Tagging

- **Dev pipeline ECR** (`sundayalbum-pipeline-dev`): MUTABLE tags — tagged by short SHA. Overwrites are allowed.
- **Prod pipeline ECR** (`cdk-hnb659fds-container-assets-*`): IMMUTABLE tags — tagged by short SHA. Each deploy must push a new unique SHA.

### Pre-commit Hook

Runs automatically on every `git commit`:
1. `pytest tests/api/ tests/handlers/ -q` — 36 tests, ~5s (moto-backed, no AWS calls)
2. `npx playwright test` from `web/` — 12 Playwright E2E tests against `dev.sundayalbum.com`, ~33s (only if `web/.auth/session.json` exists)

Install once per clone:
```bash
git config core.hooksPath .githooks
```

Skip for a one-off commit: `SKIP_HOOKS=1 git commit -m "wip"`

---

## API Key Resolution

The pipeline requires Anthropic and OpenAI API keys for AI orientation and glare removal. Resolution priority (highest wins):

```
1. User-supplied keys  (stored in DynamoDB sa-user-settings by the user)
2. System keys         (stored in AWS Secrets Manager sundayalbum/api-keys{suffix})
3. Empty string        (step falls back: OpenCV for glare, pass-through for orientation)
```

Keys are resolved once at the execution boundary (`handlers/common.py` → `make_config()` for Lambda; `src/utils/secrets.py` → `load_secrets()` for CLI/macOS). Pipeline steps never read environment variables directly.

---

## Color Restoration Algorithm

The `color_restore` step (`src/steps/color_restore.py` → `src/color/restore.py`) applies a 4-stage chain:

1. **White balance** — gray-world method. Skipped if color-cast score < 0.08 (protects intentional scene colors).
2. **Deyellowing** — adaptive LAB b* channel shift.
3. **Fade restoration** — 3-stage adaptive algorithm:
   - *Stage 1: White-point stretch* — scale so 99th-percentile luminance reaches 0.96. Self-limiting: already-bright photos barely change.
   - *Stage 2: Shadow lift* — quadratic tone-curve lift on dark pixels only. Skipped if mean luminance ≥ 0.60. Preserves dark scenes (cave, candlelit dinner).
   - *Stage 3: Vibrance saturation* — per-pixel saturation boost weighted by `(1 − S) × (1 − V²)`. Faded/muted tones lift most; vivid colors and near-white highlights are protected.
   - *Ceiling guard* — scale back if mean luminance exceeds 0.75.
4. **Sharpening** — adaptive unsharp mask. No sigmoid contrast (contrast handled by step 3).

CLAHE was used in earlier versions but replaced in April 2026 due to dimming and grain artifacts on well-exposed photos. See `journal/2026-04-09-color-restore-rethink.md`.

---

## Domains & DNS

| Domain | Points to | Notes |
|---|---|---|
| `dev.sundayalbum.com` | App Runner `sundayalbum-web-dev` | Active, used for dev testing |
| `app.sundayalbum.com` | App Runner `sundayalbum-web-prod` | Active prod frontend |
| `sundayalbum.com` | Not yet wired | www / apex not configured |

---

## Planned / Not Yet Built

- Phase 8: Admin tools (user impersonation panel) — designed, not implemented
- Phase 9: Production hardening (CloudFront, per-user rate limiting, CloudWatch alarms)
- `www.sundayalbum.com` / apex domain routing
- Multi-shot glare compositing (`src/glare/remover_multi.py` exists but not integrated)
- Border-based small-angle rotation detection (Hough-line version disabled — fires on content)

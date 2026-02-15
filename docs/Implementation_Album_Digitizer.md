# Album Digitizer — Technical Implementation Plan

**Version:** 1.0  
**Date:** February 2026  
**Status:** Draft  
**Companion Document:** PRD_Album_Digitizer.md

---

## 1. Architecture Overview

### 1.1 High-Level Architecture

The system follows a three-tier architecture: a client layer (web app), an API/orchestration layer, and a processing layer — all running on AWS with a serverless-first approach to minimize cost for a free product.

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                           │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Web App    │  │  iOS App     │  │  Mac Desktop App │   │
│  │  (React/PWA) │  │  (Phase 3)   │  │  (Phase 3)       │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
│         │                 │                    │             │
│         │    On-device: capture guidance,      │             │
│         │    glare detection (TF.js/CoreML)    │             │
└─────────┼─────────────────┼────────────────────┼─────────────┘
          │                 │                    │
          ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    API / ORCHESTRATION                       │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  API Gateway  │  │   Lambda     │  │   Step Functions │   │
│  │  (REST/WS)   │──▶  (Handlers) │──▶  (Pipeline Orch.) │   │
│  └──────────────┘  └──────────────┘  └────────┬─────────┘   │
│                                                │             │
└────────────────────────────────────────────────┼─────────────┘
                                                 │
          ┌──────────────────────────────────────┐│
          ▼                                      ▼▼
┌─────────────────────────────────────────────────────────────┐
│                   PROCESSING LAYER                          │
│                                                             │
│  ┌───────────┐  ┌───────────────┐  ┌─────────────────────┐  │
│  │  Lambda    │  │  ECS Fargate  │  │   SageMaker         │  │
│  │  (light    │  │  (heavy CV    │  │   Endpoints         │  │
│  │  tasks)    │  │  processing)  │  │   (AI models)       │  │
│  └───────────┘  └───────────────┘  └─────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
          │                 │                    │
          ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                     DATA / STORAGE                          │
│                                                             │
│  ┌───────────┐  ┌───────────────┐  ┌─────────────────────┐  │
│  │    S3      │  │  DynamoDB     │  │   ElastiCache       │  │
│  │ (images)   │  │ (sessions)    │  │   (Redis - queue)   │  │
│  └───────────┘  └───────────────┘  └─────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Why Serverless-First

For a free tool with unpredictable traffic, serverless is the right default. The cost model is pay-per-use — when nobody is processing photos, the bill approaches zero. Specific choices:

- **API Gateway + Lambda** for all API endpoints: scales to zero, no idle cost.
- **ECS Fargate** (not Lambda) for heavy CV/ML processing: Lambda has a 15-min timeout and limited memory (10 GB). CV pipelines with large images can exceed both. Fargate tasks spin up on demand and terminate after processing.
- **SageMaker Serverless Inference** for ML model endpoints: auto-scales to zero with a cold-start penalty of ~30–60 seconds. Acceptable because processing isn't real-time — users see a queue.
- **Step Functions** for orchestrating the multi-step processing pipeline: provides retry logic, error handling, and parallel execution natively.

Estimated cost at moderate scale (1,000 pages/day): $50–150/month. At low usage (personal use, 50 pages/day): under $10/month.

### 1.3 Cost Optimization Strategies

- **S3 Intelligent-Tiering** for stored images — automatically moves to cheaper storage classes.
- **7-day TTL** on all user data — S3 lifecycle rules auto-delete. No indefinite storage costs.
- **Spot Fargate** for processing tasks — 50–70% cheaper than on-demand; processing tasks are fault-tolerant and retryable.
- **SageMaker Serverless** endpoints scale to zero — no cost when idle.
- **CloudFront CDN** for the web app — reduces origin requests.
- **Per-session processing caps** as a safety net: 200 pages per session (configurable).

---

## 2. Processing Pipeline — Detailed Design

### 2.1 Pipeline Overview

Each album page image goes through a sequential pipeline orchestrated by AWS Step Functions. Each step is an independent microservice that can be developed, tested, and scaled independently.

```
Input Image (album page photo)
    │
    ▼
┌─────────────────────────┐
│  Step 1: Pre-Processing │  (Lambda)
│  - Decode & validate    │
│  - EXIF orientation fix │
│  - Resize for pipeline  │
│  - Generate thumbnail   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Step 2: Page Detection │  (Lambda)
│  - Detect album page    │
│  - Crop to page bounds  │
│  - Initial perspective  │
│    correction            │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Step 3: Glare Detection &  │  (Fargate + SageMaker)
│           Removal           │
│  - Detect glare regions     │
│  - Single-shot inpainting   │
│  - OR multi-shot composite  │
│  - Generate confidence map  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Step 4: Photo Detection &  │  (Fargate + SageMaker)
│           Splitting         │
│  - Detect photo boundaries  │
│  - Classify regions         │
│  - Extract individual crops │
└────────────┬────────────────┘
             │
             ▼ (for each detected photo)
┌─────────────────────────────┐
│  Step 5: Per-Photo          │  (Fargate)
│          Correction         │
│  - Fine perspective fix     │
│  - Bulge/warp correction    │
│  - Rotation correction      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Step 6: Color Restoration  │  (Lambda or Fargate)
│  - White balance            │
│  - Fade restoration         │
│  - Yellowing removal        │
│  - Contrast/sharpness       │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Step 7: Post-Processing    │  (Lambda)
│  - Encode to output format  │
│  - Generate before/after    │
│  - Store results in S3      │
│  - Update session in DB     │
└─────────────────────────────┘
```

### 2.2 Step 1: Pre-Processing (Lambda)

**Runtime:** Python 3.12 on Lambda (up to 10 GB memory)  
**Libraries:** Pillow, piexif

**Operations:**
- Validate file format (JPEG, PNG, HEIF/HEIC). Reject unsupported formats with clear error.
- Read and apply EXIF orientation tag (many phone photos are stored rotated with an EXIF flag).
- Strip all EXIF metadata from stored copies (privacy — location data, device info).
- Generate a working-resolution copy for pipeline processing (cap at 4000px on longest edge to keep processing fast; original resolution preserved for final output).
- Generate thumbnail (400px) for UI display.
- Upload working copy and thumbnail to S3 processing bucket.

### 2.3 Step 2: Page Detection & Perspective (Lambda or Fargate)

**Runtime:** Python 3.12  
**Libraries:** OpenCV, NumPy

**Album page detection approach:**
- Convert to grayscale. Apply Gaussian blur and Canny edge detection.
- Use Hough line transform to detect dominant straight lines.
- Find the largest quadrilateral (4-sided polygon) in the image — this is the album page.
- If no clear quadrilateral is found (e.g., page fills entire frame), assume full image is the page.
- Apply homographic transformation to produce a fronto-parallel view of the album page.

**Edge cases:**
- Album page on a patterned surface (tablecloth, carpet) — edge detection can be confused. Fallback: use color segmentation (album pages are usually a uniform color distinct from the surface).
- Multiple album pages in frame — detect the largest one, warn user if multiple detected.

### 2.4 Step 3: Glare Detection & Removal (Fargate + SageMaker)

This is the most technically challenging and differentiated component. Two approaches are implemented, used based on input.

#### 2.4.1 Single-Shot Glare Removal

**Approach: AI-powered detection + inpainting**

**Glare detection model:**
- Fine-tuned segmentation model (U-Net or Segment Anything variant) trained to detect specular highlights on glossy surfaces.
- Input: album page image. Output: per-pixel glare probability mask.
- Training data: synthetic dataset (render 3D glossy surfaces with known glare patterns) + real-world captured dataset (manually annotated album page photos with glare masks).

**Glare inpainting:**
- For detected glare regions, use a generative inpainting model to reconstruct the underlying photo content.
- Primary approach: fine-tuned diffusion-based inpainting model (Stable Diffusion Inpainting or similar) conditioned on the surrounding photo context.
- The model needs to understand that glare regions contain partial information (they're washed out, not fully occluded) — use the alpha/intensity of the glare mask to blend reconstructed content with partially visible original content.

**Foundation model integration (Anthropic Claude / OpenAI Vision):**
- Use Claude's vision API (claude-sonnet-4-5-20250929) or OpenAI's GPT-4o as a quality-check and fallback reasoning layer:
  - After glare removal, send the before/after pair to the vision API with the prompt: "Compare these two images. The second is a glare-removed version of the first. Rate the quality of the removal on a 1–10 scale and describe any remaining artifacts."
  - If the score is below threshold (e.g., 6/10), flag the photo for the user as "may need re-shoot" and apply the confidence overlay.
  - Use vision API to detect cases where glare removal introduced hallucinated content (e.g., inpainting created a face that wasn't there).

#### 2.4.2 Multi-Shot Glare Compositing

**Approach: classical computer vision (feature matching + pixel selection)**

This is a more reliable approach when 3–5 shots are available. No AI model needed — pure signal processing.

**Steps:**
1. **Feature matching & alignment:** Use ORB or SIFT feature detectors to find corresponding points across all input images. Compute homographic transforms to align all images to a common reference frame.
2. **Glare detection per image:** For each aligned image, detect glare regions (bright specular highlights significantly above local median intensity).
3. **Pixel-wise selection:** For each output pixel, select the value from the input image where that pixel has the lowest glare probability. Blend across multiple sources for smooth transitions.
4. **Gap filling:** If a pixel is glare-affected in ALL input images (rare with 4+ shots), fall back to the single-shot inpainting approach for that small region.

**Implementation:**
- Python with OpenCV for feature matching and alignment.
- NumPy for pixel-wise operations.
- This runs on Fargate (not Lambda) due to memory requirements with multiple high-res images.

### 2.5 Step 4: Photo Detection & Splitting (Fargate + SageMaker)

**Approach: object detection model + contour refinement**

**Photo detection model:**
- Fine-tuned YOLO v8 or similar real-time object detection model.
- Single class: "photo." Trained to detect rectangular photo boundaries within album pages.
- Training data: annotated album page images with bounding boxes around each individual photo.
- Handles varying numbers of photos (1–6+ per page), mixed orientations, and different sizes.

**Contour refinement:**
- YOLO provides approximate bounding boxes. Refine using OpenCV:
  - Within each detected bounding box, apply edge detection to find the precise photo border.
  - Snap to straight lines (photos have rectangular borders).
  - Handle rounded corners (some album slots have rounded corners).

**Foundation model fallback:**
- For complex or unusual layouts where YOLO struggles, send the page image to Claude's vision API:
  - Prompt: "This is a photo album page. Identify each individual photo in the image. For each photo, provide the approximate bounding box coordinates as [x1, y1, x2, y2] normalized to image dimensions. Also describe each photo briefly."
  - Parse the response and use the coordinates as detection results.
  - This is slower and more expensive than YOLO but handles edge cases (unusual layouts, decorative elements that look like photos, photos with no clear border).
- Use vision API selectively: only when YOLO confidence is below threshold.

**Region classification:**
- For each detected region, classify as: photo, decorative element, caption/text, album page background.
- Simple CNN classifier or use the vision API for ambiguous cases.

### 2.6 Step 5: Per-Photo Geometry Correction (Fargate)

**Libraries:** OpenCV, NumPy, SciPy

#### 2.6.1 Fine Perspective Correction

- Detect the four corners of each individual photo (after extraction from the page).
- Apply homographic warp to produce a perfect rectangle.
- Use Harris corner detection or line intersection for corner finding.

#### 2.6.2 Bulge/Warp Correction

This is a differentiating feature. Photos behind plastic sleeves often bulge.

**Detection:**
- Analyze straight lines within the photo (edges of buildings, horizon lines, text baselines).
- If lines that should be straight are curved, the photo is warped.
- Alternatively, use the album page grid lines (if visible through the sleeve) as a reference grid.

**Correction:**
- Model the bulge as a parametric surface (typically a low-order polynomial or thin-plate spline).
- Fit the warp model to detected line curvatures.
- Apply inverse warp to flatten the photo.
- Parameters: bulge center (x, y), magnitude, spread.

**Foundation model assistance:**
- Send the photo to Claude's vision API with the prompt: "Does this photo appear to have any barrel distortion, pincushion distortion, or bulging/warping? If so, describe the type and approximate severity."
- Use the response to initialize warp correction parameters.

#### 2.6.3 Rotation Correction

- Detect dominant lines (edges, horizons) and compute the rotation offset.
- Apply affine rotation to correct.
- For orientation detection (is the photo upside down?): use a pre-trained orientation classifier or send to vision API: "Is this photo right-side up? If not, how should it be rotated?"

### 2.7 Step 6: Color Restoration (Lambda or Fargate)

**Libraries:** OpenCV, scikit-image, Pillow

#### 2.7.1 Auto White Balance

- **Gray-world assumption:** Adjust channel means to be equal. Works well for most photos.
- **Album page reference:** If the album page background is visible around the photos, sample it as a neutral reference. Album page backgrounds are typically white/cream — use this to compute a color correction matrix.
- **Illuminant estimation:** Use a lightweight neural net (pre-trained color constancy model) for more accurate white balance in complex scenes.

#### 2.7.2 Fade Restoration

- Analyze the image histogram. Faded photos have compressed dynamic range (histogram clustered in the mid-tones).
- Apply adaptive histogram equalization (CLAHE) to restore contrast.
- Apply gentle saturation boost (typically 10–20%) to compensate for desaturation from aging.
- Use a perceptual color space (LAB) for adjustments to avoid unnatural results.

#### 2.7.3 Yellowing Removal

- Detect yellow/brown color cast by analyzing the average chrominance of the image.
- Apply complementary color correction in LAB space (shift the b* channel).
- Use a classification model to distinguish intentional warm tones (sunset, candlelight) from degradation. Heuristic: if the entire image has a uniform yellow shift, it's degradation; if only parts are warm, it's intentional.

#### 2.7.4 Sharpening

- Apply unsharp mask with conservative parameters (radius 1–2px, amount 50–100%).
- Only sharpen luminance channel (LAB) to avoid amplifying color noise.

### 2.8 Step 7: Post-Processing (Lambda)

- Encode to requested output format (JPEG at specified quality, PNG, or TIFF).
- Generate before/after comparison image (side-by-side or slider-ready pair).
- Upload final results to S3 output bucket.
- Update DynamoDB session record with processing results, confidence scores, and output locations.
- Send WebSocket notification to client: "Page X processing complete."

---

## 3. AI/ML Model Strategy

### 3.1 Model Inventory

| Model | Purpose | Base Architecture | Deployment | API |
|-------|---------|-------------------|------------|-----|
| Glare Segmentation | Detect glare regions | U-Net / SAM variant | SageMaker Serverless | Internal |
| Glare Inpainting | Reconstruct glare regions | Stable Diffusion Inpainting | SageMaker Serverless | Internal |
| Photo Detection | Find photos in album pages | YOLO v8 | SageMaker Serverless | Internal |
| Orientation Classifier | Detect up/down/rotation | ResNet-18 | Lambda (ONNX) | Internal |
| Color Constancy | White balance estimation | FC4 or similar | Lambda (ONNX) | Internal |
| Quality Assessment | Judge glare removal quality | Claude claude-sonnet-4-5-20250929 | Anthropic API | Anthropic |
| Layout Analysis | Complex page layout fallback | Claude claude-sonnet-4-5-20250929 | Anthropic API | Anthropic |
| Warp Detection | Identify photo bulging | Claude claude-sonnet-4-5-20250929 or GPT-4o | Anthropic / OpenAI API | External |

### 3.2 Foundation Model Usage (Anthropic & OpenAI)

Foundation models serve as quality assurance layers and fallback processors. They are not in the critical path for every image — they're invoked selectively to handle edge cases and validate results.

**Anthropic Claude (claude-sonnet-4-5-20250929) — Primary:**
- Vision API for layout analysis of complex album pages.
- Vision API for quality assessment of glare removal results.
- Vision API for orientation detection fallback.
- Text API for generating descriptive metadata (optional future feature).

**OpenAI GPT-4o — Secondary/Fallback:**
- Alternative vision API if Claude is unavailable or rate-limited.
- Comparative quality assessment (send the same image to both and use consensus).

**Usage patterns:**
- **Quality gate:** After Steps 3 (glare removal) and 4 (photo splitting), send results to vision API for quality scoring. Only flag results below threshold — don't block processing.
- **Fallback reasoning:** When traditional CV models have low confidence (YOLO detection confidence < 0.6, glare removal confidence < 0.7), invoke vision API for a second opinion.
- **Cost control:** At ~$0.01–0.03 per vision API call, budget approximately $0.05–0.10 per album page for AI quality checks. For 1,000 pages/day, this is $50–100/day in API costs. Optimize by only invoking for low-confidence results (estimated 20–30% of pages), reducing effective cost to $10–30/day.

**API integration code pattern:**

```python
import anthropic
import openai
import base64

# Anthropic Claude Vision - Quality Assessment
def assess_glare_removal_quality(original_image_bytes, processed_image_bytes):
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    
    original_b64 = base64.standard_b64encode(original_image_bytes).decode("utf-8")
    processed_b64 = base64.standard_b64encode(processed_image_bytes).decode("utf-8")
    
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": original_b64}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": processed_b64}},
                {"type": "text", "text": (
                    "Image 1 is an album page photo with glare from a glossy sleeve. "
                    "Image 2 is the same photo after glare removal processing. "
                    "Respond ONLY with a JSON object: "
                    '{"quality_score": 1-10, "remaining_glare": true/false, '
                    '"artifacts_detected": true/false, "notes": "brief description"}'
                )}
            ]
        }]
    )
    
    return json.loads(response.content[0].text)


# Anthropic Claude Vision - Photo Layout Detection (Fallback)
def detect_photos_in_page(page_image_bytes):
    client = anthropic.Anthropic()
    
    page_b64 = base64.standard_b64encode(page_image_bytes).decode("utf-8")
    
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": page_b64}},
                {"type": "text", "text": (
                    "This is a photo album page. Identify each individual photo. "
                    "Respond ONLY with a JSON array of objects: "
                    '[{"photo_number": 1, "bbox": [x1, y1, x2, y2], '
                    '"orientation": "portrait"|"landscape", '
                    '"description": "brief description"}] '
                    "Coordinates should be normalized 0.0-1.0 relative to image dimensions."
                )}
            ]
        }]
    )
    
    return json.loads(response.content[0].text)


# OpenAI GPT-4o - Secondary fallback
def assess_quality_openai(original_b64, processed_b64):
    client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{processed_b64}"}},
                {"type": "text", "text": "Same prompt as above..."}
            ]
        }],
        max_tokens=500
    )
    
    return json.loads(response.choices[0].message.content)
```

### 3.3 Training Data Strategy

**Synthetic data generation:**
- Render 3D scenes with glossy surfaces and controlled lighting to generate glare masks automatically.
- Composite real photos behind simulated glossy sleeves with varying glare patterns.
- Generate thousands of training pairs: (image with glare, glare mask, clean image).

**Real-world data collection:**
- Capture a diverse dataset of real album pages across album types, lighting conditions, and photo eras.
- Manually annotate glare regions, photo boundaries, and warp characteristics.
- Use data augmentation (rotation, brightness variation, color shifts) to expand the dataset.
- Target: 5,000+ annotated album page images for initial training.

**Continuous improvement:**
- With user consent, collect anonymized processing feedback (did the user manually adjust after auto-processing? Did they re-shoot?).
- Use feedback to identify failure modes and retrain models.

### 3.4 On-Device Models (Phase 3)

For native apps, lightweight models run on-device for real-time capture guidance:

- **Glare detection (TensorFlow.js / CoreML):** A tiny U-Net (~2 MB) that runs at 15+ fps on the camera feed to show glare heat map overlay. Does not need to be highly accurate — it's a guide, not the final processor.
- **Page detection (TensorFlow.js / CoreML):** Simple edge detection model (~1 MB) to show the green page-boundary rectangle in real-time.
- **Stability detection:** Accelerometer + gyroscope based — no ML model needed. Pure signal processing.

---

## 4. Infrastructure & Deployment

### 4.1 AWS Services Map

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **S3** | Image storage (uploads, processing, outputs) | 3 buckets: `upload`, `processing`, `output`. Lifecycle: 7-day expiration. Encryption: SSE-S3. |
| **CloudFront** | CDN for web app + image delivery | HTTPS only. Custom domain. Cache static assets aggressively. |
| **API Gateway** | REST API + WebSocket API | REST for uploads/actions. WebSocket for real-time processing status updates. |
| **Lambda** | API handlers + lightweight processing steps | Python 3.12. Memory: 256 MB–10 GB depending on function. Timeout: 15 min max. |
| **ECS Fargate** | Heavy CV/ML processing tasks | Spot instances. Task memory: 8–16 GB. vCPU: 2–4. Auto-scaling: 0 to N based on queue depth. |
| **Step Functions** | Processing pipeline orchestration | Standard workflow. Per-page state machine. Error handling with retry and fallback. |
| **SageMaker Serverless** | ML model inference endpoints | Serverless endpoints for glare seg, inpainting, photo detection. Cold start: ~30–60s. |
| **DynamoDB** | Session metadata, processing status | On-demand capacity. TTL: 7 days. |
| **ElastiCache (Redis)** | Processing queue, rate limiting | Single-node t4g.micro. Used as a lightweight job queue and for WebSocket connection management. |
| **SQS** | Dead letter queue, async processing triggers | Standard queues for decoupling. DLQ for failed processing retries. |
| **CloudWatch** | Monitoring, logging, alarms | Custom metrics: processing time, queue depth, error rate, cost/page. |
| **Secrets Manager** | API keys (Anthropic, OpenAI) | Rotation enabled. Accessed by Lambda/Fargate via IAM role. |
| **ECR** | Container images for Fargate tasks | Processing pipeline Docker images. |

### 4.2 Infrastructure as Code

Use **AWS CDK (TypeScript)** for all infrastructure. Organized into stacks:

```
infrastructure/
├── bin/
│   └── app.ts                    # CDK app entry point
├── lib/
│   ├── storage-stack.ts          # S3, DynamoDB, ElastiCache
│   ├── api-stack.ts              # API Gateway, Lambda handlers
│   ├── processing-stack.ts       # Step Functions, Fargate, SageMaker
│   ├── cdn-stack.ts              # CloudFront, Route 53
│   └── monitoring-stack.ts       # CloudWatch dashboards, alarms
└── config/
    ├── dev.ts
    ├── staging.ts
    └── prod.ts
```

### 4.3 CI/CD Pipeline

- **GitHub Actions** for CI/CD.
- **Trunk-based development** with feature flags.
- Pipeline: lint → unit test → build → deploy to staging → integration test → deploy to prod.
- Separate pipelines for: web app (frontend), API/Lambda (backend), processing containers (ML/CV), infrastructure (CDK).

### 4.4 Environments

| Environment | Purpose | Cost |
|-------------|---------|------|
| Dev | Local development + unit tests | ~$0 (LocalStack for AWS emulation) |
| Staging | Integration testing, model validation | ~$20–30/month (minimal capacity) |
| Production | Live users | ~$50–150/month at moderate scale |

---

## 5. Web Application

### 5.1 Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Framework | Next.js 14 (App Router) | SSR for landing page SEO, client-side for app. React ecosystem. |
| Language | TypeScript | Type safety across the codebase. |
| Styling | Tailwind CSS | Rapid UI development, small bundle. |
| State Management | Zustand | Lightweight, fits the session-based data model. |
| Camera | MediaDevices API (getUserMedia) | Browser-native camera access. No plugins. |
| Real-time | WebSocket (native) | Processing status updates pushed from server. |
| Image Processing (client) | TensorFlow.js | On-device glare/page detection for capture guidance. |
| Canvas | HTML5 Canvas / OffscreenCanvas | Image manipulation for overlays and previews. |
| File Handling | File API + drag-and-drop | Multi-file upload, folder support. |
| PWA | next-pwa | Installable, offline capture queue (Phase 2). |

### 5.2 Project Structure

```
web/
├── app/
│   ├── page.tsx                  # Landing page
│   ├── digitize/
│   │   ├── page.tsx              # Main app shell
│   │   ├── capture/
│   │   │   ├── CameraViewfinder.tsx
│   │   │   ├── GlareOverlay.tsx
│   │   │   ├── StabilityRing.tsx
│   │   │   └── CaptureControls.tsx
│   │   ├── upload/
│   │   │   ├── DropZone.tsx
│   │   │   └── BatchSettings.tsx
│   │   ├── queue/
│   │   │   ├── ProcessingQueue.tsx
│   │   │   └── QueueItem.tsx
│   │   ├── review/
│   │   │   ├── PageResults.tsx
│   │   │   ├── PhotoGrid.tsx
│   │   │   ├── PhotoEditor.tsx
│   │   │   └── BeforeAfterSlider.tsx
│   │   └── export/
│   │       ├── ExportSettings.tsx
│   │       └── DownloadManager.tsx
│   └── layout.tsx
├── components/
│   ├── ui/                       # Shared UI components
│   └── providers/                # Context providers
├── lib/
│   ├── api.ts                    # API client
│   ├── websocket.ts              # WebSocket manager
│   ├── camera.ts                 # Camera abstraction
│   ├── glare-detector.ts         # TF.js glare detection
│   ├── page-detector.ts          # TF.js page detection
│   └── image-utils.ts            # Client-side image helpers
├── stores/
│   ├── session-store.ts          # Session state
│   ├── queue-store.ts            # Processing queue state
│   └── settings-store.ts         # User preferences
└── public/
    └── models/                   # TF.js model files
```

### 5.3 Key Frontend Flows

#### Camera Capture Flow

```
1. User taps "Start with Camera"
2. Request camera permission (getUserMedia with rear camera preference)
3. Start camera feed in full-screen viewfinder
4. Load TF.js models (glare detector, page detector) — show loading indicator
5. Per frame (throttled to 15 fps):
   a. Run page detector → draw green/red boundary overlay
   b. Run glare detector → draw orange heat map overlay
   c. Check accelerometer → update stability ring
6. User taps capture (or auto-capture triggers on stability threshold)
7. Capture high-res still from video stream
8. Show capture confirmation (flash + thumbnail slide)
9. Upload image to S3 via presigned URL
10. Invoke processing pipeline via API
11. Show status in queue
```

#### Upload & Batch Processing Flow

```
1. User drops files or selects via file picker
2. Validate files client-side (format, size)
3. Show thumbnails in upload strip
4. User configures batch settings (format, quality, album name)
5. User clicks "Process All"
6. For each file:
   a. Request presigned S3 upload URL from API
   b. Upload directly to S3 (bypasses API Gateway for large files)
   c. Notify API of completed upload
7. API triggers Step Functions pipeline for all pages
8. WebSocket pushes status updates per page
9. As pages complete, results appear in the review UI
```

### 5.4 Client-Side Real-Time Processing

For capture guidance, two TF.js models run in the browser:

**Glare detection (real-time overlay):**
- Model: Tiny U-Net, ~2 MB, quantized to int8.
- Input: 256×256 downscaled camera frame.
- Output: 256×256 glare probability mask.
- Rendering: Draw mask as semi-transparent orange/red overlay on the viewfinder canvas.
- Performance target: 15+ fps on iPhone 15 Pro, 10+ fps on mid-range Android.

**Page detection (boundary overlay):**
- Model: MobileNet-based edge detector, ~3 MB.
- Input: 256×256 downscaled camera frame.
- Output: 4 corner coordinates.
- Rendering: Draw quadrilateral overlay. Green if all corners in frame, red if page is cut off.
- Performance target: 15+ fps.

---

## 6. API Design

### 6.1 REST API Endpoints

**Base URL:** `https://api.albumdigitizer.com/v1`

```
POST   /sessions                    # Create new processing session
GET    /sessions/{id}               # Get session status and results
DELETE /sessions/{id}               # Delete session and all data

POST   /sessions/{id}/pages         # Register a new page for processing
GET    /sessions/{id}/pages         # List all pages in session
GET    /sessions/{id}/pages/{pid}   # Get page processing status + results

POST   /sessions/{id}/pages/{pid}/upload-url    # Get presigned S3 upload URL
POST   /sessions/{id}/pages/{pid}/process       # Trigger processing
POST   /sessions/{id}/pages/{pid}/reprocess     # Re-process with adjusted settings

GET    /sessions/{id}/pages/{pid}/photos        # Get extracted photos for a page
GET    /sessions/{id}/photos/{phid}             # Get individual photo details
PUT    /sessions/{id}/photos/{phid}/adjustments # Apply manual adjustments

POST   /sessions/{id}/export                    # Generate export ZIP
GET    /sessions/{id}/export/{eid}/status       # Check export status
GET    /sessions/{id}/export/{eid}/download-url # Get presigned download URL
```

### 6.2 WebSocket API

**Endpoint:** `wss://ws.albumdigitizer.com`

**Events (server → client):**
```json
{"event": "page.processing_started", "page_id": "...", "step": "glare_removal"}
{"event": "page.processing_progress", "page_id": "...", "step": "photo_detection", "progress": 0.6}
{"event": "page.processing_complete", "page_id": "...", "photos": [...], "confidence": {...}}
{"event": "page.processing_error", "page_id": "...", "error": "...", "retrying": true}
{"event": "export.ready", "export_id": "...", "download_url": "..."}
```

### 6.3 Authentication

No authentication for MVP (session-based, no accounts). Sessions are identified by a UUID generated client-side and stored in localStorage.

Security considerations:
- Rate limiting: 10 pages/minute per IP, 200 pages/session.
- Upload size limit: 50 MB per file.
- Session isolation: S3 paths and DynamoDB partition keys include session ID; no cross-session access is possible.

---

## 7. Data Model

### 7.1 DynamoDB Schema

**Sessions Table:**
```
PK: session_id (UUID)
Attributes:
  created_at: ISO 8601 timestamp
  ttl: Unix timestamp (created_at + 7 days)
  status: "active" | "expired"
  settings: {
    output_format: "jpeg" | "png" | "tiff"
    jpeg_quality: 70-100
    album_name: string | null
  }
  page_count: number
  photo_count: number
```

**Pages Table:**
```
PK: session_id
SK: page_id (UUID)
Attributes:
  created_at: ISO 8601 timestamp
  ttl: Unix timestamp
  status: "uploaded" | "processing" | "complete" | "error"
  current_step: string
  original_s3_key: string
  thumbnail_s3_key: string
  processing_results: {
    glare_removal: { confidence: float, method: "single_shot" | "multi_shot" }
    photos_detected: number
    processing_time_ms: number
  }
  photos: [
    {
      photo_id: UUID
      bbox: [x1, y1, x2, y2]
      output_s3_key: string
      thumbnail_s3_key: string
      corrections_applied: {
        perspective: boolean
        dewarp: boolean
        rotation_degrees: float
        color_enhanced: boolean
      }
      manual_adjustments: { ... } | null
    }
  ]
```

### 7.2 S3 Bucket Structure

```
album-digitizer-data/
├── uploads/
│   └── {session_id}/
│       └── {page_id}/
│           ├── original.jpg          # User's uploaded image
│           └── shots/                # Multi-shot mode
│               ├── shot_01.jpg
│               ├── shot_02.jpg
│               └── ...
├── processing/
│   └── {session_id}/
│       └── {page_id}/
│           ├── working.jpg           # Resized working copy
│           ├── page_detected.jpg     # After page detection
│           ├── glare_mask.png        # Glare detection output
│           ├── deglared.jpg          # After glare removal
│           └── ...                   # Intermediate results
└── output/
    └── {session_id}/
        └── {page_id}/
            ├── photo_01.jpg          # Final processed photo
            ├── photo_01_original.jpg # Uncorrected version
            ├── photo_02.jpg
            └── ...
```

---

## 8. Development Plan

### 8.1 Phase 1: MVP (Weeks 1–8)

**Week 1–2: Foundation**
- Set up monorepo (Turborepo or Nx).
- AWS CDK infrastructure: S3, DynamoDB, API Gateway, Lambda.
- Next.js web app skeleton with routing and Tailwind.
- File upload flow: drag-and-drop → presigned URL → S3.
- Basic API: create session, upload page, get status.

**Week 3–4: Core Processing Pipeline (Traditional CV)**
- Step Functions pipeline skeleton.
- Step 1 (pre-processing): EXIF handling, resize, thumbnail.
- Step 2 (page detection): OpenCV edge detection + perspective correction.
- Step 5 (perspective correction): per-photo keystone fix.
- Step 6 (color restoration): white balance, CLAHE, sharpening.
- Step 7 (post-processing): encode + store results.
- End-to-end test: upload a photo → get processed result.

**Week 5–6: AI Processing Steps**
- Step 3 (glare removal): train/fine-tune glare segmentation model. Implement single-shot inpainting. Deploy to SageMaker Serverless.
- Step 4 (photo detection): train YOLO on album page dataset. Deploy to SageMaker Serverless.
- Integrate Anthropic Claude vision API for quality assessment.
- Multi-shot glare compositing (OpenCV feature matching).

**Week 7–8: Web UI Completion**
- Camera viewfinder with capture flow (no ML overlays yet — Phase 2).
- Processing queue with WebSocket status updates.
- Results review UI (page view, photo grid, before/after slider).
- Photo editor (crop, rotate, color sliders).
- Export/download flow (individual + ZIP).
- Landing page.

### 8.2 Phase 2: Polish & Enhancement (Weeks 9–12)

- TF.js on-device models for capture guidance (glare overlay, page detection).
- Bulge/warp detection and correction.
- Multi-shot capture UX flow.
- PWA support (installable, offline capture queue).
- Performance optimization (processing time target < 15s/page).
- Error handling, edge cases, retry logic.
- Privacy policy, data deletion flows.
- Load testing and cost optimization.

### 8.3 Phase 3: Native Apps (Weeks 13–20+)

- Mac desktop app (Electron or Swift).
- iOS app (Swift/SwiftUI).
- On-device CoreML models.
- Platform-specific camera integration.
- Photos.app integration.

---

## 9. AI Coding Tool Integration

### 9.1 Recommended Setup

Use **Claude Code** (Anthropic's CLI coding tool) as the primary AI coding assistant. It excels at multi-file codebases and can directly edit files, run tests, and iterate.

**Workflow per component:**
1. Seed Claude Code with the relevant section of this implementation doc.
2. Have it scaffold the component (file structure, types, interfaces).
3. Iterate on implementation with tests.
4. Review and commit.

### 9.2 Cursor / Windsurf Instructions File

Create a `.cursorrules` or `.windsurfrules` file in the repo root for AI coding tools:

```
# Album Digitizer - AI Coding Instructions

## Project Overview
Album digitizer web app. Processes physical photo album pages to extract clean
individual photos with glare removal, photo splitting, perspective correction,
and color restoration.

## Tech Stack
- Frontend: Next.js 14 (App Router), TypeScript, Tailwind CSS, Zustand
- Backend: AWS Lambda (Python 3.12), API Gateway, Step Functions
- Processing: OpenCV, ONNX Runtime, TensorFlow (Python)
- ML Inference: SageMaker Serverless, Anthropic Claude API, OpenAI API
- Infrastructure: AWS CDK (TypeScript)
- Storage: S3, DynamoDB, ElastiCache (Redis)

## Conventions
- TypeScript strict mode everywhere
- Python type hints on all functions
- All API responses follow: { success: boolean, data?: T, error?: string }
- S3 keys follow: {bucket}/{session_id}/{page_id}/{filename}
- DynamoDB uses single-table design with PK/SK patterns
- Environment variables for all configuration (never hardcode)
- Anthropic API key env var: ANTHROPIC_API_KEY
- OpenAI API key env var: OPENAI_API_KEY

## Key Files
- PRD: ./docs/PRD_Album_Digitizer.md
- Implementation: ./docs/Implementation_Album_Digitizer.md
- Infrastructure: ./infrastructure/lib/
- Frontend: ./web/app/
- Processing pipeline: ./processing/
- ML models: ./models/

## Testing
- Frontend: Vitest + React Testing Library
- Backend: pytest
- Integration: pytest with moto (AWS mocks)
- E2E: Playwright

## Important Notes
- All image processing must preserve original resolution
- Never store images longer than 7 days (TTL)
- Foundation model API calls are for quality gates and fallbacks, not critical path
- Multi-shot glare compositing is the highest-quality path and should be recommended
- EXIF data must be stripped for privacy before storage
```

### 9.3 Component-by-Component Coding Order

This is the recommended order for building with an AI coding tool, designed to produce testable increments:

1. **Infrastructure CDK stacks** — get AWS resources provisioned first.
2. **API Lambda handlers** — session CRUD, upload URL generation, status endpoints.
3. **Pre-processing Lambda** — simplest processing step, validates the pipeline skeleton.
4. **Page detection** — OpenCV-based, testable with sample images.
5. **Color restoration** — independent step, produces visible results quickly.
6. **Photo detection (YOLO)** — requires training data but has clear evaluation metrics.
7. **Perspective correction** — builds on page detection output.
8. **Glare removal** — most complex step, tackle after simpler steps are solid.
9. **Step Functions orchestration** — wire all steps together.
10. **Web app: upload flow** — simplest user-facing flow.
11. **Web app: processing queue + WebSocket** — real-time status.
12. **Web app: results review** — depends on processing output format.
13. **Web app: photo editor** — most UI-intensive component.
14. **Web app: camera capture** — requires device testing.
15. **Web app: TF.js overlays** — depends on trained lightweight models.

---

## 10. Monitoring & Observability

### 10.1 Key Metrics

| Metric | Source | Alarm Threshold |
|--------|--------|----------------|
| Processing time per page | Step Functions | > 60 seconds (P95) |
| Processing error rate | Step Functions | > 5% |
| API latency | API Gateway | > 2 seconds (P95) |
| SageMaker cold start time | CloudWatch | > 120 seconds |
| S3 storage usage | S3 metrics | > 100 GB |
| Anthropic API error rate | Application logs | > 10% |
| Anthropic API cost per day | Application logs | > $50 |
| Concurrent sessions | DynamoDB | > 500 |
| Queue depth | Redis | > 100 pages waiting |

### 10.2 CloudWatch Dashboard

A single operational dashboard showing: processing pipeline health, API latency and error rates, cost tracking (per-page cost breakdown), model inference latency, and active session count.

---

## 11. Security Considerations

- **No PII stored:** Photos are the only user data. No accounts, no emails, no names in MVP.
- **Encryption at rest:** S3 SSE-S3. DynamoDB encryption enabled by default.
- **Encryption in transit:** HTTPS everywhere (CloudFront, API Gateway).
- **EXIF stripping:** All location, device, and personal metadata removed before storage.
- **Session isolation:** S3 paths and DynamoDB keys are session-scoped. No IAM or API mechanism exists to access another session's data.
- **API keys:** Anthropic and OpenAI keys stored in AWS Secrets Manager, accessed via IAM role (not environment variables in code).
- **Rate limiting:** API Gateway throttling + per-IP limits in Lambda.
- **Content scanning:** Optionally integrate AWS Rekognition for content moderation (detect and reject inappropriate uploads).
- **7-day data retention:** Automatic deletion via S3 lifecycle rules and DynamoDB TTL. No data persists beyond the session window.

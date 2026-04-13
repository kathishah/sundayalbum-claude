# 2026-04-13 — Move Skip Routing into Step Functions

## Problem

Pipeline handlers currently carry workflow-position awareness that belongs in the
orchestrator, not in application code.

`handlers/common.py` maintains two ordered lists:

```python
PRE_SPLIT_STEP_ORDER = [
    "load", "normalize", "page_detect", "perspective", "photo_detect", "photo_split"
]
PER_PHOTO_STEP_ORDER = ["ai_orient", "glare_remove", "geometry", "color_restore"]
```

Every handler calls into these lists at startup:

```python
# handlers/load.py
if should_skip_pre_split(event, "load"):
    return event

# handlers/ai_orient.py
if should_skip_per_photo(event, "ai_orient"):
    return event
```

**Consequences:**
- Adding or removing a step requires editing both the ordered list in `handlers/common.py`
  and the handler file itself.
- Handlers are coupled to the workflow — a pure processing function knows it is step 1
  of 6 in a sequence.
- Skip logic and business logic are mixed in the same function; test coverage must
  cover both paths in every handler.
- The Step Functions state machine, which *is* the canonical step order, does not own
  this routing decision.

---

## Scope: AWS Web Path Only

This change affects **only the AWS Lambda / Step Functions path**. The local pipeline
(`src/pipeline.py`) used by the CLI and macOS app is entirely separate:

| Surface | Step selection | Affected? |
|---|---|---|
| Web (Step Functions) | `start_from` field in event, evaluated by handlers | **Yes** — this is what we're changing |
| CLI | `--steps load,normalize,...` flag → `steps_filter` list in `Pipeline.process()` | No |
| macOS app | Shells out to CLI with `--steps` flag | No |

`src/pipeline.py` uses a simple `if steps_filter is None or 'load' in steps_filter` pattern
directly in the orchestrator body. It already owns its own routing and has no dependency on
`handlers/common.py`. No changes to `src/`, `src/cli.py`, or `mac-app/`.

---

## Goal

Lambda handlers become pure processors: receive event, run step, return enriched event.
No handler inspects `start_from`. No handler knows its position in the pipeline.

The Step Functions state machine owns all skip routing. Adding a new step means adding
it to the ASL and writing a handler file — no edits to `handlers/common.py`.

---

## Design

### Two distinct skip scenarios

**Scenario A — `start_from` (reprocess from a later step)**
User clicks "Re-run from page_detect". The state machine should skip `load` and
`normalize` entirely and invoke `page_detect` first.

→ Decision is purely about step ordering. Belongs in the state machine as Choice states.

**Scenario B — `reprocess_photo_index` (single-photo reprocess)**
User clicks "Re-run glare removal for photo 2 only". Within the per-photo Map state,
iterations for photos 1, 3, 4 should be skipped.

→ Decision is "is this the right photo?", not "is this the right step". This can be a
single position-agnostic helper that each per-photo handler calls. No ordering lists
involved.

---

### Scenario A: Choice states in the state machine (ASL)

Each pipeline step gets a guard Choice state inserted before its Task state. The
Choice evaluates `$.start_from` against a fixed set of values and either invokes the
Lambda or jumps to a Pass (skip) state that forwards the event unchanged.

Example — guard for `load` (skip when `start_from` is any later step):

```json
"SkipLoad?": {
  "Type": "Choice",
  "Choices": [
    {
      "Or": [
        { "Variable": "$.start_from", "StringEquals": "normalize" },
        { "Variable": "$.start_from", "StringEquals": "page_detect" },
        { "Variable": "$.start_from", "StringEquals": "perspective" },
        { "Variable": "$.start_from", "StringEquals": "photo_detect" },
        { "Variable": "$.start_from", "StringEquals": "photo_split" },
        { "Variable": "$.start_from", "StringEquals": "ai_orient" },
        { "Variable": "$.start_from", "StringEquals": "glare_remove" },
        { "Variable": "$.start_from", "StringEquals": "geometry" },
        { "Variable": "$.start_from", "StringEquals": "color_restore" }
      ],
      "Next": "LoadSkipped"
    }
  ],
  "Default": "Load"
},
"LoadSkipped": {
  "Type": "Pass",
  "Next": "SkipNormalize?"
},
"Load": {
  "Type": "Task",
  "Resource": "arn:aws:lambda:...:sa-pipeline-load",
  "Next": "SkipNormalize?"
}
```

The guard for `normalize` only lists steps that come after it (`page_detect` onward),
and so on. Each guard's enumerated set is exactly "all steps after me" — determined
once at state machine definition time.

**State machine size:** 11 steps × ~2 states (Choice + Pass) added = ~22 new states.
Well within the Step Functions 1 MB definition limit.

### Scenario B: Position-agnostic photo guard in handlers

Replace `should_skip_per_photo(event, "ai_orient")` with a simple helper that has no
ordering knowledge:

```python
# handlers/common.py
def skip_this_photo(event: dict) -> bool:
    """Return True if this photo iteration should be skipped.

    Only applies when reprocess_photo_index targets a specific photo.
    Has no knowledge of step order.
    """
    reprocess_idx = event.get("reprocess_photo_index")
    photo_idx = event.get("photo_index")
    if reprocess_idx is not None and photo_idx is not None:
        return int(photo_idx) != int(reprocess_idx)
    return False
```

Each per-photo handler calls `skip_this_photo(event)` — a check about photo identity,
not step position.

### The `output_key` problem for skipped photos

`color_restore` is the last per-photo step and its return value provides the
`output_key` that `finalize` collects. When a photo is skipped, `color_restore` calls
`get_existing_output_key` to read the prior output key from DynamoDB and return it so
`finalize` still has a complete list.

This is not ordering knowledge; it is domain knowledge about what `finalize` needs.
It stays in `color_restore.handler` unchanged:

```python
# handlers/color_restore.py
def handler(event, context):
    if skip_this_photo(event):
        output_key = (
            get_existing_output_key(user_hash, job_id, photo_index)
            or f"output/SundayAlbum_{stem}_Photo{idx}.jpg"
        )
        return {"photo_index": photo_index, "output_key": output_key, ...}
    # ... run step normally
```

All other per-photo handlers return `event` unchanged when `skip_this_photo` is true.

---

### State machine definition: ASL JSON file (not CDK constructs)

The state machine is defined as a JSON file (`infra/pipeline_state_machine.json`) and
loaded into CDK via:

```python
sfn.StateMachine(
    self, "Pipeline",
    definition_body=sfn.DefinitionBody.from_file("pipeline_state_machine.json"),
    ...
)
```

**Why JSON over CDK constructs:**
CDK's `stepfunctions` constructs (Chain, Parallel, Map) become deeply nested and
hard to read for a machine with ~30 states, multiple branches, and a fan-out Map. The
ASL JSON is the canonical format for Step Functions — it is what the AWS console,
`aws stepfunctions describe-state-machine`, and all Step Functions tooling emit and
consume. Writing it directly avoids a translation layer and keeps the definition
readable alongside the execution graph in the console.

The current CDK stack (`infra/infra/infra_stack.py`) is a stub — the live state
machine was created and patched directly via AWS CLI. This change formalises it into
CDK for the first time, with the JSON file as the single source of truth.

---

### `start_from` validation in `api/jobs.py` (in scope)

`_handle_reprocess` currently passes `from_step` to Step Functions without validating
it. An unknown value would silently run the full pipeline (all Choice guards evaluate
False → Default → Task for every step).

Add explicit validation against the known step list:

```python
# api/jobs.py
VALID_STEP_NAMES = [
    "load", "normalize", "page_detect", "perspective",
    "photo_detect", "photo_split",
    "ai_orient", "glare_remove", "geometry", "color_restore",
]

def _handle_reprocess(job_id, event, user_hash):
    ...
    from_step = (body.get("from_step") or "").strip()
    if from_step and from_step not in VALID_STEP_NAMES:
        return bad_request(f"Unknown step '{from_step}'. Valid steps: {VALID_STEP_NAMES}")
    ...
```

An empty `from_step` (full reprocess) remains valid. The `VALID_STEP_NAMES` list is the
single declaration of valid steps on the API side — it does not need to stay in sync
with `handlers/common.py` (those lists are deleted) but does need to stay in sync with
the state machine ASL. A comment in `api/jobs.py` documents this.

---

## UI Indicators: How They Work and What to Test

Understanding how the donut wheel and step-tree nav derive their state is essential
for writing correct regression tests.

### Current state flow during a reprocess

When `POST /jobs/{jobId}/reprocess?from_step=page_detect` is called:

1. **`api/jobs.py` immediately** writes `current_step = "page_detect"`,
   `status = "processing"` to DynamoDB — before the state machine even starts.
2. DynamoDB Streams → `sa-broadcaster` → WebSocket push → frontend receives
   `{ current_step: "page_detect", status: "processing" }`.
3. **Library card donut wheel** (`AlbumPageCard.tsx`):
   - `completedCount = BACKEND_TO_VISUAL["page_detect"] = 1` → 1 of 6 segments amber
   - `isRunning = true` → segment 1 (0-indexed) pulses amber
4. **Step tree left nav** (`jobs/[jobId]/page.tsx`):
   - `activeJobStepKey = BACKEND_TO_JOB_TREE["page_detect"] = "page_detect"`
   - `page_detect` row shows pulsing orange dot; `load`/`normalize` rows do not pulse
   - `load` row remains `available` (its `debug_keys["01_loaded"]` still exists from the prior run)
5. As each subsequent Lambda runs and calls `update_step()`, the broadcaster fires
   again, advancing both indicators.

**This flow is identical with the new design.** `api/jobs.py` sets `current_step`
before the state machine starts in both cases. The only difference is that skipped
Lambdas are no longer invoked at all (Choice → Pass) — but they never called
`update_step()` anyway, so the UI sees no difference.

### What the indicator tests must cover

The Playwright tests below verify two scenarios: a normal full run and a reprocess-
from-step. For each, they assert:

- **Donut wheel pulsing**: the correct segment is pulsing amber (`[data-pulse]` or
  by checking the `sa-segment-pulse` animation class is on the correct `<path>`).
- **Donut segment count**: the number of filled amber segments matches the expected
  visual step index.
- **Tree nav dot**: only the row for the currently running step has the pulsing dot
  (`animate-ping` span), not skipped steps.
- **Tree row availability**: rows for steps before `start_from` remain clickable (not
  `disabled`); rows for steps that haven't run yet are disabled.
- **After completion**: all processed steps are clickable; no row shows a pulsing dot.

---

## Files to Change

| File | Change |
|---|---|
| `infra/pipeline_state_machine.json` | **New file** — full ASL definition with Choice + Pass guard before every Task state |
| `infra/infra/sundayalbum_stack.py` | Replace stub with real CDK stack: `sfn.StateMachine` + `DefinitionBody.from_file(...)` |
| `api/jobs.py` | Add `VALID_STEP_NAMES`; validate `from_step` in `_handle_reprocess`; return 400 for unknown step |
| `handlers/common.py` | Delete `PRE_SPLIT_STEP_ORDER`, `PER_PHOTO_STEP_ORDER`, `should_skip_pre_split`, `should_skip_per_photo`. Add `skip_this_photo`. Keep `get_existing_output_key`. |
| `handlers/load.py` | Remove `should_skip_pre_split` import and call. |
| `handlers/normalize.py` | Same. |
| `handlers/page_detect.py` | Same. |
| `handlers/perspective.py` | Same. |
| `handlers/photo_detect.py` | Same. |
| `handlers/photo_split.py` | Same. |
| `handlers/ai_orient.py` | Replace `should_skip_per_photo` with `skip_this_photo`. |
| `handlers/glare_remove.py` | Same. |
| `handlers/geometry.py` | Same. |
| `handlers/color_restore.py` | Replace `should_skip_per_photo` with `skip_this_photo`; keep `get_existing_output_key` call. |

**No changes to `src/`, `src/cli.py`, `mac-app/`, or `web/`.**

---

## Implementation Order

1. **Fetch live ASL** — `aws stepfunctions describe-state-machine --state-machine-arn ...`
   to capture the current state machine definition as the baseline.

2. **Write `infra/pipeline_state_machine.json`** — add Choice + Pass guards for all
   10 skippable steps (pre-split and per-photo). Verify against the live definition:
   all existing Task states, retry/catch config, and Map iterator must be preserved.

3. **Wire CDK** — replace `infra/infra/infra_stack.py` stub with real constructs;
   `cdk synth` and diff against live state machine.

4. **Strip handlers** — remove `should_skip_*` calls from all 10 handlers; add
   `skip_this_photo` where needed (4 per-photo handlers).

5. **Update `handlers/common.py`** — delete the two ordered lists and the two skip
   functions; add `skip_this_photo`.

6. **Add `start_from` validation** — add `VALID_STEP_NAMES` and the 400 guard to
   `api/jobs.py`.

7. **Run handler tests** — tests 32–33 deleted; new tests added (see below). Full
   suite must pass.

8. **Deploy to dev** — `cdk deploy` (infra) + push handler changes → Lambda deploy.
   Run manual smoke tests.

---

## Testing Plan

> **Deployment requirement:** Layers 1–2 (unit + ASL) run locally with no AWS access.
> Layers 3–4 (Playwright E2E + manual smoke) **require a deployment to dev** — run
> `cdk deploy` for the infra change and push handler/API changes to trigger the Lambda
> deploy CI workflow before executing these layers. Do not run WIP Playwright tests
> against dev until both the state machine and Lambda code are deployed together; a
> partial deploy (ASL only or handlers only) will break reprocess flows.

### Layer 1 — Handler unit tests (moto, ~5 s)

**Deleted:**
- Test 32 (`test_skip_pre_split_when_start_from_later`) — tests a function that no
  longer exists.
- Test 33 (`test_skip_per_photo_wrong_index`) — same.

**Replaced with:**

```python
# test_skip_this_photo — 2 tests
def test_skip_this_photo_wrong_index():
    event = {**make_base_event("u", "j"), "reprocess_photo_index": 1, "photo_index": 2}
    assert skip_this_photo(event) is True

def test_skip_this_photo_correct_index():
    event = {**make_base_event("u", "j"), "reprocess_photo_index": 1, "photo_index": 1}
    assert skip_this_photo(event) is False
```

**New handler invocation tests (6 tests):**
Verify that handlers with `start_from` in the event now run fully — because the skip
decision has moved to the state machine, not the handler:

```python
def test_load_runs_even_with_start_from():
    """load.handler ignores start_from — skip is state machine's job."""
    event = {**make_base_event(user_hash, job_id), "start_from": "normalize"}
    load_handler.handler(event, None)
    item = get_job(user_hash, job_id)
    assert "01_loaded" in item["debug_keys"]  # ran, did not skip
```

One test like this for each of: `load`, `normalize`, `page_detect`, `perspective`,
`photo_detect`, `ai_orient` (6 handlers with the most risk of silent regression).

**`start_from` validation tests (3 new tests in `tests/api/`):**

```python
def test_reprocess_rejects_unknown_step():
    resp = call_reprocess(job_id, from_step="bad_step_name")
    assert resp["statusCode"] == 400
    assert "bad_step_name" in body(resp)["message"]

def test_reprocess_accepts_empty_from_step():
    resp = call_reprocess(job_id, from_step="")
    assert resp["statusCode"] == 200  # full reprocess is valid

def test_reprocess_accepts_all_valid_steps():
    for step in VALID_STEP_NAMES:
        resp = call_reprocess(job_id, from_step=step)
        assert resp["statusCode"] == 200
```

### Layer 2 — State machine ASL tests (CDK synthesis, no AWS)

Test file: `infra/tests/unit/test_sfn_routing.py` (45 tests)

Uses `aws-cdk-lib` `Template.from_stack()` to synthesize the stack in-process and assert
the emitted ASL — no deployment required.

Run from the `infra/` directory using the infra venv:

```bash
cd infra
JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION=1 .venv/bin/pytest tests/unit/test_sfn_routing.py -v
# → 45 passed in ~2s
```

| Test group | What it asserts |
|---|---|
| Existence | All 6 pre-split and 3 per-photo Choice states exist; all Pass (skipped) states exist |
| Guard wiring | Each Choice's `Default` routes to the correct Task; skip `Next` routes to the correct Pass |
| No self-skip | A step's own name is never in its guard's skip condition set |
| Exact conditions | Each guard's `StringEquals` set matches exactly the expected "later steps" |
| Boundary (pre-split) | `start_from=page_detect` skips load+normalize but not page_detect; `start_from=ai_orient` skips all 6 pre-split guards |
| Boundary (per-photo) | `start_from=glare_remove` skips ai_orient but not glare_remove; `start_from=color_restore` skips all 3 per-photo guards |
| Terminal step | `ColorRestore` is a Task with no preceding `SkipColorRestore?` guard |

### Layer 3 — Playwright E2E tests (against dev environment)

Add to the existing Playwright suite (`web/tests/`). These tests verify the indicator
behavior described above — both normal runs and reprocess-from-step.

**Group: Step indicator correctness during reprocess**

| Test ID | Scenario | What it checks |
|---|---|---|
| `T-sfn-ui-01` | Reprocess from `page_detect` | Immediately after reprocess starts: donut shows 1 filled segment, segment 1 pulses; `page_detect` tree row has pulsing dot |
| `T-sfn-ui-02` | Reprocess from `page_detect` | `load` and `normalize` tree rows are `available` (clickable, not disabled) |
| `T-sfn-ui-03` | Reprocess from `page_detect` | `load` and `normalize` rows do NOT have the pulsing dot during processing |
| `T-sfn-ui-04` | Reprocess from `page_detect` | After completion: all processed rows clickable; no row has pulsing dot; donut shows 6 filled segments |
| `T-sfn-ui-05` | Single-photo reprocess (`photo_index=2`) | During processing: only photo 2 per-photo rows show pulsing dot; photo 1 per-photo rows do not |
| `T-sfn-ui-06` | Normal full run | Donut segment count advances in order (0→1→2…→6) as WebSocket step updates arrive |

**Implementation note:** these tests require a job to already be in `complete` state
before triggering reprocess. A test fixture creates and fully processes a job using the
existing `web/.auth/session.json` token. If the session file does not exist, these
tests are skipped (same skip behaviour as current Playwright suite).

### Layer 4 — Automated smoke tests (manually triggered, not CI)

Test file: `web/tests/sfn_smoke.spec.ts` — Playwright tests against the dev environment.
Run manually after deploying to dev; not wired into CI.

```bash
cd web
STAGE=dev TEST_USER_EMAIL=... DEV_FRONTEND_URL=https://dev.sundayalbum.com \
  DEV_API_URL=https://nodcooz758.execute-api.us-west-2.amazonaws.com \
  SESSIONS_TABLE=sa-sessions-dev npx playwright test sfn_smoke.spec.ts
```

| Test ID | Scenario | What it checks |
|---|---|---|
| `T-sfn-smoke-01` | Reprocess from `page_detect` | Execution graph in SFN console shows `SkipLoad` and `SkipNormalize` Pass states (grey); no Lambda log entries for `sa-pipeline-load` or `sa-pipeline-normalize` |
| `T-sfn-smoke-02` | Single-photo reprocess (`reprocess_photo_index=2`, 2-photo job) | Only photo 2 per-photo Lambda logs appear; `finalize` receives `output_keys` for both photos |
| `T-sfn-smoke-03` | Invalid `from_step` via API | 400 response with a descriptive error message |

**Implementation note:** each smoke test creates or reuses a completed job in the dev
environment using the `web/.auth/session-dev.json` token. Tests skip if the session file
is absent.

### Layer 5 — Local pipeline regression (CLI + macOS, post-deploy)

Although this change does not touch `src/pipeline.py`, `src/cli.py`, or `mac-app/`,
run these checks last — after dev deployment is confirmed working — to ensure no
accidental import or dependency has been broken:

```bash
# Full Python test suite (includes pipeline step unit tests)
pytest tests/ -v

# CLI smoke — full run and partial --steps run on a real test image
python -m src.cli process test-images/IMG_three_pics_normal.HEIC --output ./output/ --debug
python -m src.cli process test-images/IMG_cave_normal.HEIC --output ./output/ \
  --steps load,normalize,page_detect

# macOS app — unit tests only (non-disruptive)
xcodebuild test -scheme SundayAlbum -destination 'platform=macOS' \
  -skip-testing:SundayAlbumUITests
```

Expected: all 45 Swift unit tests pass; CLI produces correct output for both full
and partial runs; no import errors from any `handlers/` change leaking into `src/`.

---

## Tradeoffs

| | Current | After |
|---|---|---|
| Step ordering defined in | `handlers/common.py` (Python list) | State machine ASL JSON |
| Handler awareness of workflow | Knows its name and position | Knows nothing about order |
| Adding a step | Edit `common.py` + new handler | Edit ASL JSON + new handler |
| Skip routing visibility | `logging.info("skipping...")` in Lambda logs | Step Functions execution graph |
| State machine states | ~11 Task states | ~33 states (Choice + Pass + Task per step) |
| `reprocess_photo_index` check | Inside `should_skip_per_photo` | `skip_this_photo()` — no ordering knowledge |
| `start_from` validation | None (silent full-pipeline fallback) | 400 in `api/jobs.py` |
| ASL definition | Not in source control (patched via CLI) | `infra/pipeline_state_machine.json` |

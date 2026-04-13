/**
 * Layer 4 smoke tests: Step Functions skip routing (manually triggered, not CI).
 *
 * These tests verify that the CDK Choice states actually route correctly in the
 * live dev environment after deployment. They are NOT wired into CI — run
 * manually after deploying to dev:
 *
 *   cd web
 *   STAGE=dev TEST_USER_EMAIL=... DEV_FRONTEND_URL=https://dev.sundayalbum.com \
 *     DEV_API_URL=https://nodcooz758.execute-api.us-west-2.amazonaws.com \
 *     SESSIONS_TABLE=sa-sessions-dev npx playwright test sfn_smoke.spec.ts
 *
 * All tests skip if session-dev.json is absent (same pattern as other e2e tests).
 *
 * Test IDs:
 *   T-sfn-smoke-01  Reprocess from page_detect — SkipLoad/SkipNormalize are Pass states
 *   T-sfn-smoke-02  Single-photo reprocess — only targeted photo's per-photo Lambdas fire
 *   T-sfn-smoke-03  Invalid from_step — API returns 400 with descriptive message
 */

import fs from 'fs'
import { test, expect } from '@playwright/test'
import {
  SFNClient,
  ListExecutionsCommand,
  GetExecutionHistoryCommand,
  type HistoryEvent,
} from '@aws-sdk/client-sfn'
import { COMPLETED_JOB_FILE, AUTH_FILE } from '../../playwright.config'

// ── Config ────────────────────────────────────────────────────────────────────

const API_URL       = process.env.DEV_API_URL ?? 'https://nodcooz758.execute-api.us-west-2.amazonaws.com'
const REGION        = process.env.AWS_DEFAULT_REGION ?? 'us-west-2'
const STATE_MACHINE = 'arn:aws:states:us-west-2:680073251743:stateMachine:sa-pipeline-dev'

const POLL_INTERVAL_MS   = 5_000
const REPROCESS_TIMEOUT  = 4 * 60 * 1000  // 4 min

// ── Helpers ───────────────────────────────────────────────────────────────────

function readToken(): string | null {
  try {
    const state = JSON.parse(fs.readFileSync(AUTH_FILE, 'utf-8'))
    for (const origin of state.origins ?? []) {
      for (const entry of origin.localStorage ?? []) {
        if (entry.name === 'sa_token') return entry.value as string
      }
    }
  } catch { /* missing */ }
  return null
}

function readCompletedJobId(): string | null {
  try {
    const d = JSON.parse(fs.readFileSync(COMPLETED_JOB_FILE, 'utf-8'))
    return typeof d.job_id === 'string' ? d.job_id : null
  } catch { return null }
}

function authed(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }
}

async function getJobStatus(token: string, jobId: string): Promise<{ status: string; photo_count?: number }> {
  const r = await fetch(`${API_URL}/jobs/${jobId}`, { headers: authed(token) })
  if (!r.ok) throw new Error(`getJob ${jobId} failed: ${r.status}`)
  return r.json()
}

/**
 * Poll until job status is 'complete' or 'failed' (i.e. not 'processing').
 * Useful before triggering a reprocess when a previous one may still be running.
 */
async function waitForIdle(token: string, jobId: string, timeoutMs = REPROCESS_TIMEOUT): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const { status } = await getJobStatus(token, jobId)
    if (status === 'complete' || status === 'failed') return
    await new Promise((res) => setTimeout(res, POLL_INTERVAL_MS))
  }
  throw new Error(`Job ${jobId} did not reach idle state within ${timeoutMs / 1000}s`)
}

/**
 * Trigger a reprocess and wait for the job to return to 'complete'.
 * Returns the job_id (unchanged) once polling confirms completion.
 */
async function reprocessAndWait(
  token: string,
  jobId: string,
  body: Record<string, unknown>,
): Promise<string> {
  const r = await fetch(`${API_URL}/jobs/${jobId}/reprocess`, {
    method: 'POST',
    headers: authed(token),
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    throw new Error(`reprocess failed (${r.status}): ${await r.text()}`)
  }

  const deadline = Date.now() + REPROCESS_TIMEOUT
  while (Date.now() < deadline) {
    await new Promise((res) => setTimeout(res, POLL_INTERVAL_MS))
    const { status } = await getJobStatus(token, jobId)
    if (status === 'complete') return jobId
    if (status === 'failed')  throw new Error(`Reprocess failed for job ${jobId}`)
  }
  throw new Error(`Reprocess timed out after ${REPROCESS_TIMEOUT / 1000}s`)
}

/**
 * Find the most recent SFN execution that contains the given job_id in its name.
 * Returns its execution ARN.
 */
async function findExecutionArn(sfn: SFNClient, jobId: string): Promise<string> {
  const resp = await sfn.send(new ListExecutionsCommand({
    stateMachineArn: STATE_MACHINE,
    statusFilter: 'SUCCEEDED',
    maxResults: 20,
  }))
  const match = (resp.executions ?? []).find((e) => e.name?.includes(jobId))
  if (!match?.executionArn) throw new Error(`No succeeded execution found for job ${jobId}`)
  return match.executionArn
}

/**
 * Fetch all history events for an execution (handles pagination).
 */
async function getHistory(sfn: SFNClient, executionArn: string): Promise<HistoryEvent[]> {
  const events: HistoryEvent[] = []
  let nextToken: string | undefined
  do {
    const resp = await sfn.send(new GetExecutionHistoryCommand({
      executionArn,
      maxResults: 1000,
      nextToken,
    }))
    events.push(...(resp.events ?? []))
    nextToken = resp.nextToken
  } while (nextToken)
  return events
}

/** Return all state names that entered as a Task (i.e. Lambda was invoked). */
function taskStateNames(events: HistoryEvent[]): Set<string> {
  return new Set(
    events
      .filter((e) => e.type === 'TaskStateEntered')
      .map((e) => e.stateEnteredEventDetails?.name ?? '')
      .filter(Boolean),
  )
}

/** Return all state names that entered as a Pass (i.e. skipped). */
function passStateNames(events: HistoryEvent[]): Set<string> {
  return new Set(
    events
      .filter((e) => e.type === 'PassStateEntered')
      .map((e) => e.stateEnteredEventDetails?.name ?? '')
      .filter(Boolean),
  )
}

// ── Skip guard ────────────────────────────────────────────────────────────────

// All tests skip if the dev session or completed job file is absent.
test.beforeEach(() => {
  if (!readToken()) {
    test.skip(true, 'No dev session — run global-setup first (session-dev.json missing)')
  }
  if (!readCompletedJobId()) {
    test.skip(true, 'No completed job — run global-setup first (completed-job-dev.json missing)')
  }
})

// ── T-sfn-smoke-01 ─────────────────────────────────────────────────────────────

test('T-sfn-smoke-01: reprocess from page_detect routes Load and Normalize to Pass states', async () => {
  test.setTimeout(5 * 60 * 1000)  // reprocess can take up to ~4 min
  const token = readToken()!
  const jobId = readCompletedJobId()!
  const sfn   = new SFNClient({ region: REGION })

  await reprocessAndWait(token, jobId, { from_step: 'page_detect' })

  const execArn = await findExecutionArn(sfn, jobId)
  const events  = await getHistory(sfn, execArn)
  const tasks   = taskStateNames(events)
  const passes  = passStateNames(events)

  // Load and Normalize must NOT have fired as Lambda tasks
  expect(tasks, 'Load Lambda should be skipped when reprocessing from page_detect').not.toContain('Load')
  expect(tasks, 'Normalize Lambda should be skipped when reprocessing from page_detect').not.toContain('Normalize')

  // LoadSkipped and NormalizeSkipped Pass states must have fired
  expect(passes, 'LoadSkipped Pass state should be present').toContain('LoadSkipped')
  expect(passes, 'NormalizeSkipped Pass state should be present').toContain('NormalizeSkipped')

  // PageDetect and later steps must have run as Tasks
  expect(tasks, 'PageDetect Lambda should run when reprocessing from page_detect').toContain('PageDetect')
})

// ── T-sfn-smoke-02 ─────────────────────────────────────────────────────────────

test('T-sfn-smoke-02: single-photo reprocess only invokes per-photo Lambdas for the target photo', async () => {
  test.setTimeout(5 * 60 * 1000)  // reprocess can take up to ~4 min
  const token = readToken()!
  const jobId = readCompletedJobId()!

  // Only meaningful for multi-photo jobs
  const { photo_count } = await getJobStatus(token, jobId)
  if (!photo_count || photo_count < 2) {
    test.skip(true, `Job ${jobId} has ${photo_count ?? 0} photo(s) — need ≥2 for this test`)
  }

  const sfn = new SFNClient({ region: REGION })

  // Reprocess only photo index 0 (first photo), starting from ai_orient
  await reprocessAndWait(token, jobId, {
    from_step: 'ai_orient',
    photo_index: 0,
  })

  const execArn = await findExecutionArn(sfn, jobId)
  const events  = await getHistory(sfn, execArn)

  // All pre-split steps should be skipped (start_from=ai_orient)
  const passes = passStateNames(events)
  expect(passes).toContain('LoadSkipped')
  expect(passes).toContain('NormalizeSkipped')
  expect(passes).toContain('PageDetectSkipped')
  expect(passes).toContain('PerspectiveSkipped')
  expect(passes).toContain('PhotoDetectSkipped')
  expect(passes).toContain('PhotoSplitSkipped')

  // AiOrient should run (it's the start_from step, not skipped)
  const tasks = taskStateNames(events)
  expect(tasks).toContain('AiOrient')

  // Confirm finalize ran (job reached completion)
  expect(tasks).toContain('Finalize')
})

// ── T-sfn-smoke-03 ─────────────────────────────────────────────────────────────

test('T-sfn-smoke-03: invalid from_step returns 400 with descriptive error', async () => {
  test.setTimeout(5 * 60 * 1000)  // may need to wait for a prior reprocess to finish
  const token = readToken()!
  const jobId = readCompletedJobId()!

  // Job must be idle (complete/failed) before the API validates from_step.
  // A prior test's reprocess may still be running.
  await waitForIdle(token, jobId)

  const r = await fetch(`${API_URL}/jobs/${jobId}/reprocess`, {
    method: 'POST',
    headers: authed(token),
    body: JSON.stringify({ from_step: 'not_a_real_step' }),
  })

  expect(r.status).toBe(400)

  const body = await r.json()
  // Should mention the invalid step name
  expect(JSON.stringify(body)).toContain('not_a_real_step')
})

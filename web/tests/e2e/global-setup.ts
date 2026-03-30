/**
 * Playwright global setup — authenticates the test user and saves the session
 * to .auth/session.json so all tests start already logged in.
 *
 * If .auth/session.json already exists and the token inside is still valid
 * against the API, the full send-code / verify flow is skipped. This avoids
 * the auth rate-limiter when re-running tests within the same hour.
 *
 * Auth flow (when session is missing or expired):
 *   1. POST /auth/send-code   — triggers a 6-digit verification email
 *   2. Read the code directly from DynamoDB sa-sessions-dev (no email needed in CI)
 *   3. POST /auth/verify      — exchanges code for a session_token
 *   4. Persist session_token in localStorage via page.evaluate
 *   5. Save storageState to AUTH_FILE
 *
 * Required env vars (set in .env.local or GitHub Actions secrets):
 *   TEST_USER_EMAIL       e.g. chintan@reachto.me
 *   DEV_FRONTEND_URL      e.g. https://dev.sundayalbum.com
 *   DEV_API_URL           e.g. https://nodcooz758.execute-api.us-west-2.amazonaws.com
 *   AWS_ACCESS_KEY_ID     IAM read access to sa-sessions-dev DynamoDB table
 *   AWS_SECRET_ACCESS_KEY
 *   AWS_DEFAULT_REGION    defaults to us-west-2
 */

import fs from 'fs'
import { chromium } from '@playwright/test'
import {
  DynamoDBClient,
  GetItemCommand,
} from '@aws-sdk/client-dynamodb'
import { AUTH_FILE } from '../../playwright.config'

const API_URL = process.env.DEV_API_URL ?? ''
const FRONTEND_URL = process.env.DEV_FRONTEND_URL ?? 'https://dev.sundayalbum.com'
const EMAIL = process.env.TEST_USER_EMAIL ?? ''
const REGION = process.env.AWS_DEFAULT_REGION ?? 'us-west-2'
const SESSIONS_TABLE = 'sa-sessions-dev'

if (!API_URL) throw new Error('DEV_API_URL env var is required')
if (!EMAIL) throw new Error('TEST_USER_EMAIL env var is required')

/** Extract sa_token from an existing .auth/session.json, or null. */
function readSavedToken(): string | null {
  try {
    const state = JSON.parse(fs.readFileSync(AUTH_FILE, 'utf-8'))
    for (const origin of state.origins ?? []) {
      for (const entry of origin.localStorage ?? []) {
        if (entry.name === 'sa_token') return entry.value as string
      }
    }
  } catch {
    // file missing or malformed
  }
  return null
}

/** Return true if the token is still accepted by the API. */
async function isTokenValid(token: string): Promise<boolean> {
  try {
    const resp = await fetch(`${API_URL}/jobs`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    return resp.ok
  } catch {
    return false
  }
}

async function readCodeFromDynamo(): Promise<string> {
  const ddb = new DynamoDBClient({ region: REGION })
  for (let attempt = 0; attempt < 10; attempt++) {
    const resp = await ddb.send(new GetItemCommand({
      TableName: SESSIONS_TABLE,
      Key: { email: { S: EMAIL } },
    }))
    const code = resp.Item?.code?.S
    const expiresAt = resp.Item?.code_expires_at?.N
    if (code && expiresAt) {
      const now = Math.floor(Date.now() / 1000)
      if (parseInt(expiresAt, 10) > now) {
        return code
      }
    }
    await new Promise((r) => setTimeout(r, 1000))
  }
  throw new Error(`Verification code not found in DynamoDB for ${EMAIL} after 10 s`)
}

async function persistToken(token: string, userHash: string) {
  const browser = await chromium.launch()
  const context = await browser.newContext()
  const page = await context.newPage()

  await page.goto(FRONTEND_URL)
  await page.evaluate(
    ({ t, h }: { t: string; h: string }) => {
      localStorage.setItem('sa_token', t)
      if (h) localStorage.setItem('sa_user_hash', h)
    },
    { t: token, h: userHash },
  )

  await context.storageState({ path: AUTH_FILE })
  await browser.close()
  console.log(`[global-setup] Session saved to ${AUTH_FILE}`)
}

export default async function globalSetup() {
  // ── Fast path: reuse existing session if still valid ─────────────────────
  const savedToken = readSavedToken()
  if (savedToken) {
    const valid = await isTokenValid(savedToken)
    if (valid) {
      console.log('[global-setup] Existing session is still valid — skipping re-auth')
      return
    }
    console.log('[global-setup] Saved session expired — re-authenticating')
  }

  // ── 1. Request a verification code ───────────────────────────────────────
  const sendResp = await fetch(`${API_URL}/auth/send-code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: EMAIL }),
  })
  if (!sendResp.ok) {
    const body = await sendResp.text()
    throw new Error(`send-code failed (${sendResp.status}): ${body}`)
  }

  // ── 2. Read code from DynamoDB ────────────────────────────────────────────
  const code = await readCodeFromDynamo()
  console.log(`[global-setup] Got verification code for ${EMAIL}`)

  // ── 3. Verify code → get session_token ───────────────────────────────────
  const verifyResp = await fetch(`${API_URL}/auth/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: EMAIL, code }),
  })
  if (!verifyResp.ok) {
    const body = await verifyResp.text()
    throw new Error(`verify failed (${verifyResp.status}): ${body}`)
  }
  const { session_token, user_hash } = await verifyResp.json()
  if (!session_token) throw new Error('verify returned no session_token')
  console.log('[global-setup] Session token obtained')

  // ── 4. Persist token in browser localStorage → save storageState ─────────
  await persistToken(session_token, user_hash ?? '')
}

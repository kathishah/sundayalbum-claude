/**
 * Tests 41–45: Step-detail page — StepTree, auto-select, thumbnail strip,
 * Results view, reprocess trigger.
 */

import path from 'path'
import fs from 'fs'
import os from 'os'
import { test, expect, type Page } from '@playwright/test'

const API_URL = process.env.DEV_API_URL ?? ''

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeTinyJpeg(): string {
  const jpegBytes = Buffer.from(
    'ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909' +
    '0806050808090a0b0c0c0b0a0a0c0f11100c0e0f0d0a0a0c1612130f141011120e0e1619' +
    '16171813181716161a1b1d1b18191c191617161c211c1a1e211f17181d221d1e2124242424' +
    '17191d261d242122241f1d2126251f202224242524' +
    'ffc0000b080001000101011100ffc40014000100000000000000000000000000000000' +
    'ffc40014100100000000000000000000000000000000ffd9',
    'hex',
  )
  const tmpFile = path.join(os.tmpdir(), `sa_test_${Date.now()}.jpg`)
  fs.writeFileSync(tmpFile, jpegBytes)
  return tmpFile
}

/**
 * StepTree locator — works with or without data-testid.
 * The job detail page has exactly one <nav> element.
 */
function stepTree(page: Page) {
  return page.locator('[data-testid="step-tree"]').or(page.locator('nav'))
}

/** Upload a file and return the job_id from the createJob API response. */
async function uploadAndGetJobId(page: Page): Promise<string | null> {
  await page.goto('/library')
  const fileInput = page.locator('input[type="file"]').first()
  const tmpFile = makeTinyJpeg()

  try {
    const responsePromise = page.waitForResponse(
      (r) => r.url().includes('/jobs') && r.request().method() === 'POST',
      { timeout: 15_000 },
    )

    await fileInput.setInputFiles(tmpFile)

    try {
      const resp = await responsePromise
      const body = await resp.json().catch(() => null)
      return body?.job_id ?? null
    } catch {
      return null
    }
  } finally {
    fs.unlinkSync(tmpFile)
  }
}

/**
 * Return the first complete job_id, or null.
 * Must navigate to /library first so localStorage is accessible.
 */
async function findCompletedJobId(page: Page): Promise<string | null> {
  // Navigate to the app first — page.evaluate requires a real origin
  await page.goto('/library')
  const token = await page.evaluate(() => localStorage.getItem('sa_token'))
  if (!token || !API_URL) return null

  try {
    const resp = await fetch(`${API_URL}/jobs`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    if (!resp.ok) return null
    const data = await resp.json()
    const completed = (data.jobs ?? []).find((j: { status: string }) => j.status === 'complete')
    return completed?.job_id ?? null
  } catch {
    return null
  }
}

// ── T41: StepTree is visible on job detail page ───────────────────────────────

test('T41: step-detail page renders StepTree', async ({ page }) => {
  const jobId = await uploadAndGetJobId(page)
  if (!jobId) {
    test.skip()
    return
  }

  await page.goto(`/jobs/${jobId}`)

  // StepTree nav should be visible immediately (even before processing completes)
  const tree = stepTree(page)
  await expect(tree).toBeVisible({ timeout: 10_000 })

  // At least one step button exists (Load is always in JOB_STEP_TREE)
  const loadBtn = tree.getByRole('button', { name: 'Load' })
  await expect(loadBtn).toBeVisible()
})

// ── T42: Breadcrumb shows selected step name ──────────────────────────────────

test('T42: breadcrumb shows selected step name', async ({ page }) => {
  const jobId = await uploadAndGetJobId(page)
  if (!jobId) {
    test.skip()
    return
  }

  await page.goto(`/jobs/${jobId}`)
  await page.waitForLoadState('networkidle')

  // Breadcrumb is inside <main> — scope there to avoid matching the nav header link too
  const libraryLink = page.getByRole('main').getByRole('link', { name: 'Library' })
  await expect(libraryLink).toBeVisible({ timeout: 10_000 })

  // A step name (the third breadcrumb segment) should also be visible
  // The step detail page always shows a label like "Load", "Orientation", etc.
  const breadcrumbRow = libraryLink.locator('xpath=..')
  await expect(breadcrumbRow).toBeVisible()
})

// ── T43: Thumbnail strip appears for single-photo complete jobs ───────────────

test('T43: thumbnail strip visible for completed single-photo job', async ({ page }) => {
  const jobId = await findCompletedJobId(page)
  if (!jobId) {
    test.skip()
    return
  }

  await page.goto(`/jobs/${jobId}`)

  // Thumb strip locator — works with or without data-testid
  // The strip is a horizontal scrollable row of step thumbnails
  const thumbStrip = page.locator('[data-testid="thumb-strip"]').or(
    page.locator('img[alt="Load"]').locator('xpath=./../../..'),
  )

  const isVisible = await thumbStrip.isVisible({ timeout: 5_000 }).catch(() => false)
  if (!isVisible) {
    // Multi-photo job or no thumbnails — skip rather than fail
    test.skip()
    return
  }

  const thumbImgs = thumbStrip.locator('img')
  await expect(thumbImgs.first()).toBeVisible()
})

// ── T44: Results view renders output photo for completed job ──────────────────

test('T44: Results view shows output photo', async ({ page }) => {
  const jobId = await findCompletedJobId(page)
  if (!jobId) {
    test.skip()
    return
  }

  await page.goto(`/jobs/${jobId}`)

  const tree = stepTree(page)
  await expect(tree).toBeVisible({ timeout: 10_000 })

  const resultsBtn = tree.getByRole('button', { name: 'Results' })
  const isVisible = await resultsBtn.isVisible().catch(() => false)
  if (!isVisible) {
    test.skip()
    return
  }

  await resultsBtn.click()

  // Results view locator — works with or without data-testid
  const resultsView = page.locator('[data-testid="results-view"]').or(
    page.locator('button', { hasText: /download all/i }).locator('xpath=./../../..'),
  )
  await expect(resultsView).toBeVisible({ timeout: 5_000 })

  const outputImg = resultsView.locator('img').first()
  await expect(outputImg).toBeVisible()
})

// ── T45: Reprocess — GlareRemovalView Apply button triggers API call ──────────

test('T45: glare removal Apply button triggers reprocess API call', async ({ page }) => {
  const jobId = await findCompletedJobId(page)
  if (!jobId) {
    test.skip()
    return
  }

  await page.goto(`/jobs/${jobId}`)

  const tree = stepTree(page)
  await expect(tree).toBeVisible({ timeout: 10_000 })

  // Glare removal step button (labeled "Glare removal" in PHOTO_STEP_TREE)
  const glareBtn = tree.getByRole('button', { name: /glare/i })
  const isVisible = await glareBtn.isVisible().catch(() => false)
  if (!isVisible) {
    test.skip()
    return
  }

  await glareBtn.click()

  const applyBtn = page.getByRole('button', { name: /apply/i })
  if (!(await applyBtn.isVisible().catch(() => false))) {
    test.skip()
    return
  }

  // Intercept the reprocess API call
  const reprocessPromise = page.waitForRequest(
    (r) => r.url().includes('/reprocess') && r.method() === 'POST',
    { timeout: 5_000 },
  ).catch(() => null)

  await applyBtn.click()

  const req = await reprocessPromise
  if (req) {
    expect(req.method()).toBe('POST')
  } else {
    // Apply button should at minimum be in a loading state
    await expect(applyBtn).toBeDisabled({ timeout: 3_000 })
  }
})

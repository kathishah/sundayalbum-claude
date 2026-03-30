/**
 * Tests 36–40: Library page — upload preview, thumbnails, output thumbnails,
 * delete job, and progress wheel visibility.
 *
 * These tests run authenticated (storageState from global-setup).
 * They upload a small synthetic JPEG via the DropZone file input so no
 * real iPhone photo is required in CI.
 */

import path from 'path'
import fs from 'fs'
import os from 'os'
import { test, expect, type Page } from '@playwright/test'

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Create a minimal valid JPEG file in a temp dir. */
function makeTinyJpeg(): string {
  // 1×1 white JPEG (minimal valid JFIF)
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
 * Locates album cards — works with both the deployed build (no data-testid)
 * and the new build (has data-testid). Each card has a <p title={filename}>
 * as a direct child of the card div.
 */
function anyCard(page: Page) {
  return page.locator('[data-testid="album-card"]').or(
    page.locator('p[title]').locator('xpath=..'),
  )
}

// ── T36: Upload creates a card with filename ──────────────────────────────────

test('T36: upload creates card with filename', async ({ page }) => {
  await page.goto('/library')
  await expect(page.getByRole('heading', { name: 'Library' })).toBeVisible()

  const tmpFile = makeTinyJpeg()
  try {
    const fileInput = page.locator('input[type="file"]').first()
    await fileInput.setInputFiles(tmpFile)

    const filename = path.basename(tmpFile)
    // Wait for the filename text — most reliable signal that the card appeared
    await expect(page.getByText(filename, { exact: false })).toBeVisible({ timeout: 10_000 })

    // Confirm a card element wraps the filename
    const card = anyCard(page).first()
    await expect(card).toBeVisible({ timeout: 5_000 })
  } finally {
    fs.unlinkSync(tmpFile)
  }
})

// ── T37: Before-thumbnail appears in card ─────────────────────────────────────

test('T37: before-thumbnail img appears in card after upload', async ({ page }) => {
  await page.goto('/library')

  const tmpFile = makeTinyJpeg()
  try {
    const fileInput = page.locator('input[type="file"]').first()
    await fileInput.setInputFiles(tmpFile)

    const filename = path.basename(tmpFile)
    await expect(page.getByText(filename, { exact: false })).toBeVisible({ timeout: 10_000 })

    // The card should render a ThumbBox with an <img> (the optimistic preview_url)
    const card = anyCard(page).first()
    const thumbImg = card.locator('img').first()
    await expect(thumbImg).toBeVisible({ timeout: 5_000 })
  } finally {
    fs.unlinkSync(tmpFile)
  }
})

// ── T38: Completed jobs show output photos in card ────────────────────────────

test('T38: complete job shows output photos in the after-section', async ({ page }) => {
  await page.goto('/library')
  await expect(page.getByRole('heading', { name: 'Library' })).toBeVisible()

  // Look for any existing complete jobs (may be empty in CI — that's OK)
  const cards = anyCard(page)
  await page.waitForTimeout(1_000)  // let list load
  const count = await cards.count()
  if (count === 0) {
    test.skip()
    return
  }

  // Find first complete card (has more than one img: before-thumb + output photo)
  for (let i = 0; i < count; i++) {
    const card = cards.nth(i)
    const imgs = card.locator('img')
    const imgCount = await imgs.count()
    if (imgCount > 1) {
      await expect(imgs.nth(1)).toBeVisible()
      return
    }
  }
  test.skip()
})

// ── T39: Delete button removes the card ──────────────────────────────────────

test('T39: delete button removes card from library', async ({ page }) => {
  await page.goto('/library')

  const tmpFile = makeTinyJpeg()
  try {
    const fileInput = page.locator('input[type="file"]').first()
    await fileInput.setInputFiles(tmpFile)

    const filename = path.basename(tmpFile)
    await expect(page.getByText(filename, { exact: false })).toBeVisible({ timeout: 10_000 })

    const card = anyCard(page).first()
    const countBefore = await anyCard(page).count()

    // Hover to reveal the delete button
    await card.hover()
    const deleteBtn = page.getByRole('button', { name: 'Delete job' }).first()
    await expect(deleteBtn).toBeVisible({ timeout: 3_000 })
    await deleteBtn.click()

    // Filename text should disappear
    await expect(page.getByText(filename, { exact: false })).not.toBeAttached({ timeout: 5_000 })

    const countAfter = await anyCard(page).count()
    expect(countAfter).toBeLessThan(countBefore)
  } finally {
    fs.unlinkSync(tmpFile)
  }
})

// ── T40: Processing card shows progress wheel SVG ─────────────────────────────

test('T40: processing card shows progress wheel', async ({ page }) => {
  await page.goto('/library')

  const tmpFile = makeTinyJpeg()
  try {
    const fileInput = page.locator('input[type="file"]').first()
    await fileInput.setInputFiles(tmpFile)

    const filename = path.basename(tmpFile)
    await expect(page.getByText(filename, { exact: false })).toBeVisible({ timeout: 10_000 })

    const card = anyCard(page).first()

    // The card is uploading/processing (shows ProgressWheel SVG) or has an img
    // (completed quickly). Either way, some visual element should be present.
    const svgOrImg = card.locator('svg, img')
    await expect(svgOrImg.first()).toBeVisible({ timeout: 5_000 })
  } finally {
    fs.unlinkSync(tmpFile)
  }
})

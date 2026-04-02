import { defineConfig, devices } from '@playwright/test'
import path from 'path'

/**
 * Playwright E2E configuration.
 *
 * Targets dev.sundayalbum.com (set DEV_FRONTEND_URL to override).
 * Global setup authenticates as the test account and saves a session
 * cookie / localStorage snapshot to .auth/session.json.
 *
 * Required env vars:
 *   TEST_USER_EMAIL      e.g. chintan@reachto.me
 *   DEV_FRONTEND_URL     e.g. https://dev.sundayalbum.com
 *   DEV_API_URL          e.g. https://nodcooz758.execute-api.us-west-2.amazonaws.com
 *   AWS_ACCESS_KEY_ID    IAM key with read access to sa-sessions-dev DynamoDB table
 *   AWS_SECRET_ACCESS_KEY
 */

const BASE_URL = process.env.DEV_FRONTEND_URL ?? 'https://dev.sundayalbum.com'

export const AUTH_FILE = path.join(__dirname, '.auth/session.json')

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,       // tests may share server state; run serially
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI
    ? [['github'], ['html', { open: 'never' }]]
    : [['list'], ['html', { open: 'never' }]],

  use: {
    baseURL: BASE_URL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },

  globalSetup: './tests/e2e/global-setup.ts',

  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        storageState: AUTH_FILE,
      },
    },
  ],
})

export const STEP_PROGRESS: Record<string, number> = {
  load: 0.05,
  normalize: 0.10,
  page_detect: 0.20,
  perspective: 0.25,
  photo_detect: 0.30,
  photo_split: 0.35,
  ai_orient: 0.50,
  glare_remove: 0.70,
  geometry: 0.85,
  color_restore: 0.95,
  done: 1.0,
}

export const STEP_LABELS: Record<string, string> = {
  load: 'Loading',
  normalize: 'Normalising',
  page_detect: 'Finding Page',
  perspective: 'Correcting Perspective',
  photo_detect: 'Detecting Photos',
  photo_split: 'Splitting Photos',
  ai_orient: 'Orienting',
  glare_remove: 'Removing Glare',
  geometry: 'Geometry',
  color_restore: 'Colour Restore',
  done: 'Complete',
}

/**
 * Maps the 10 backend Lambda step names to the 6 visual step indices (0–5)
 * shown in the PipelineProgressWheel pie segments and JobStatusLine.
 *
 * Backend steps not listed independently (normalize, perspective, photo_split,
 * geometry) are collapsed into their adjacent visual step — they don't advance
 * the visible counter on their own.
 */
export const BACKEND_TO_VISUAL: Record<string, number> = {
  load:          0,
  normalize:     0,
  page_detect:   1,
  perspective:   1,
  photo_detect:  2,
  photo_split:   2,
  ai_orient:     3,
  glare_remove:  4,
  geometry:      4,
  color_restore: 5,
  done:          5,
}

/** 6 user-visible step names, indices 0–5, matching macOS PipelineStep enum order */
export const VISUAL_STEP_LABELS: string[] = [
  'Load',
  'Page',
  'Split',
  'Orient',
  'Glare',
  'Color',
]

export const TOTAL_VISUAL_STEPS = 6

/**
 * Labels shown under each debug image in the expanded card strip.
 * Keys match the backend step names returned in debug_urls.
 */
export const DEBUG_STEP_LABELS: Record<string, string> = {
  load:          '1. Load',
  page_detect:   '2. Page',
  photo_detect:  '3. Photos',
  ai_orient:     '4. Orient',
  glare_remove:  '5. Glare',
  color_restore: '6. Color',
}

export const ALLOWED_EXTENSIONS = new Set(['.heic', '.jpg', '.jpeg', '.png'])

export const TOKEN_KEY = 'sa_token'
export const USER_HASH_KEY = 'sa_user_hash'

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  'https://nodcooz758.execute-api.us-west-2.amazonaws.com'

export const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ??
  'wss://0rqvs3ydf9.execute-api.us-west-2.amazonaws.com/$default'

export const POLLING_INTERVAL_MS = 4000

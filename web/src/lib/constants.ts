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

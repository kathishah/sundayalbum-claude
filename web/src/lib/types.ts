export type JobStatus = 'uploading' | 'processing' | 'complete' | 'failed'

export interface Job {
  job_id: string
  status: JobStatus
  current_step: string
  step_detail: string
  input_filename: string
  input_stem: string
  photo_count: number
  created_at: string
  updated_at: string
  error_message: string
  processing_time: number
  output_keys: string[]
  output_urls?: string[]
  upload_url?: string
}

export interface StepUpdate {
  type: 'step_update'
  jobId: string
  status: JobStatus
  step: string
  detail: string
  progress: number
  photoCount: number
}

export interface ApiKeyStatus {
  has_anthropic_key: boolean
  has_openai_key: boolean
}

export interface AuthResponse {
  session_token: string
  user_hash: string
  expires_at: string
}

export interface CreateJobResponse {
  job_id: string
  upload_url: string
  upload_key: string
  expires_in: number
}

export interface JobsResponse {
  jobs: Job[]
  count: number
}

export interface StartJobResponse {
  status: string
  execution_arn: string
}

export type AccessMode = 'MOCK' | 'DEMO' | 'ADMIN'

export type IncidentStatus = 'OPEN' | 'TRACKING' | 'RESOLVED' | 'CLOSED'

export interface Incident {
  id: string
  ref: string
  title: string
  status: IncidentStatus
  created_at: string
  updated_at: string
  description: string
  last_seen_location: string
  last_seen_at: string
  person_count: number
  alert_count: number
  pending_alert_count: number
  is_paused: boolean
}

export interface Person {
  id: string
  name: string
  age: number
  gender: string
  description: string
  incident_id: string
  enrolled_at: string
  source_image_path?: string
  extra_photo_paths?: string[]
}

export interface Sighting {
  id: string
  incident_id: string
  person_id: string
  person_name: string
  confidence: number
  camera_id: string
  source_name: string
  timestamp: string
  status: 'PENDING' | 'CONFIRMED' | 'REJECTED'
  frame_index: number
  snapshot_path?: string
}

export interface Camera {
  id: string
  name: string
  location: string
  status: 'ACTIVE' | 'INACTIVE' | 'ERROR'
  fps?: number
}

export interface ActivityEvent {
  id: string
  type:
    | 'CASE_CREATED'
    | 'PERSON_ENROLLED'
    | 'TRACKING_STARTED'
    | 'SIGHTING_DETECTED'
    | 'ALERT_VERIFIED'
    | 'CASE_CLOSED'
    | 'COMMENT_ADDED'
  incident_ref?: string
  message: string
  timestamp: string
}

export interface TimelineEntry {
  id: string
  type:
    | 'CASE_CREATED'
    | 'PERSON_ENROLLED'
    | 'EMBEDDINGS_GENERATED'
    | 'TRACKING_STARTED'
    | 'SIGHTING_DETECTED'
    | 'ALERT_VERIFIED'
    | 'COMMENT_ADDED'
    | 'CASE_CLOSED'
    | 'TRACKING_PAUSED'
  timestamp: string
  message: string
  sighting?: Sighting
  comment?: string
}

export interface SystemMetrics {
  fps: number
  detector_latency_ms: number
  gpu_status: 'OK' | 'DEGRADED' | 'UNAVAILABLE'
  hardware_backend_type: number
  queue_depth: number
  identity_switch_rate: number
  active_tracks: number
  cpu_percent: number
  memory_mb: number
  stable_matches: number
  validator_rejection_rate: number
  confirmation_rate: number
  uptime_seconds: number
}

export interface SparkPoint {
  t: string
  v: number
}

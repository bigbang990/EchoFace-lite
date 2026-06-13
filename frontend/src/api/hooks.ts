import { useCallback, useEffect, useRef, useState } from 'react'
import { useAppStore } from '../store/appStore'
import { createApiClient } from './client'
import type {
  Incident,
  Person,
  Sighting,
  Camera,
  ActivityEvent,
  TimelineEntry,
  SystemMetrics,
  SparkPoint,
} from '../types'
import {
  mockIncidents,
  mockPersonsByIncident,
  mockSightingsByIncident,
  mockTimelinesByIncident,
  mockCameras,
  mockActivityFeed,
  mockSystemMetrics,
  mockFpsHistory,
} from '../mock/data'

// ── normalizers ───────────────────────────────────────────────────────────────

type Raw = Record<string, unknown>

function normalizeStatus(raw: unknown): Incident['status'] {
  switch (String(raw ?? '').toLowerCase()) {
    case 'active':
    case 'tracking':  return 'TRACKING'
    case 'resolved':  return 'RESOLVED'
    case 'closed':    return 'CLOSED'
    default:          return 'OPEN'
  }
}

function normalizeSightingStatus(raw: unknown): Sighting['status'] {
  switch (String(raw ?? '').toLowerCase()) {
    case 'confirmed': return 'CONFIRMED'
    case 'rejected':  return 'REJECTED'
    default:          return 'PENDING'
  }
}

function normalizeIncident(raw: Raw, idx: number): Incident {
  return {
    id: String(raw.id ?? ''),
    ref: String(raw.ref ?? `INC-${String(idx + 1).padStart(3, '0')}`),
    title: String(raw.title ?? 'Unknown case'),
    status: normalizeStatus(raw.status),
    created_at: String(raw.created_at ?? new Date().toISOString()),
    updated_at: String(raw.updated_at ?? new Date().toISOString()),
    description: String(raw.description ?? ''),
    last_seen_location: String(raw.last_seen_location ?? ''),
    last_seen_at: String(raw.last_seen_at ?? new Date().toISOString()),
    person_count: Number(raw.person_count ?? 0),
    alert_count: Number(raw.alert_count ?? raw.sighting_count ?? 0),
  }
}

function normalizePerson(raw: Raw, incidentId: string): Person {
  return {
    id: String(raw.id ?? ''),
    name: String(raw.display_name ?? raw.name ?? ''),
    age: Number(raw.age ?? 0),
    gender: String(raw.gender ?? ''),
    description: String(raw.notes ?? raw.description ?? ''),
    incident_id: incidentId,
    enrolled_at: String(raw.enrolled_at ?? raw.created_at ?? new Date().toISOString()),
    source_image_path: raw.source_image_path ? String(raw.source_image_path) : undefined,
  }
}

function normalizeSighting(raw: Raw): Sighting {
  return {
    id: String(raw.id ?? ''),
    incident_id: String(raw.incident_id ?? ''),
    person_id: String(raw.person_id ?? ''),
    person_name: String(raw.person_name ?? 'Unknown'),
    confidence: Number(raw.confidence ?? raw.match_confidence ?? 0),
    camera_id: String(raw.camera_id ?? ''),
    source_name: String(raw.source_name ?? raw.camera_id ?? 'Unknown source'),
    timestamp: String(raw.timestamp ?? raw.detected_at ?? raw.created_at ?? new Date().toISOString()),
    status: normalizeSightingStatus(raw.status),
    frame_index: Number(raw.frame_index ?? 0),
  }
}

// ── timeline builder ──────────────────────────────────────────────────────────

function buildTimeline(
  incident: Incident,
  persons: Person[],
  sightings: Sighting[]
): TimelineEntry[] {
  const entries: TimelineEntry[] = [
    {
      id: `created-${incident.id}`,
      type: 'CASE_CREATED',
      timestamp: incident.created_at,
      message: `Case ${incident.ref} opened — ${incident.title}`,
    },
  ]

  for (const p of persons) {
    entries.push({
      id: `enrolled-${p.id}`,
      type: 'PERSON_ENROLLED',
      timestamp: p.enrolled_at,
      message: `Person enrolled: ${p.name}`,
    })
    entries.push({
      id: `embed-${p.id}`,
      type: 'EMBEDDINGS_GENERATED',
      timestamp: new Date(new Date(p.enrolled_at).getTime() + 22_000).toISOString(),
      message: 'Face embeddings generated — tracking profile active',
    })
  }

  if (['TRACKING', 'RESOLVED', 'CLOSED'].includes(incident.status)) {
    entries.push({
      id: `tracking-${incident.id}`,
      type: 'TRACKING_STARTED',
      timestamp: incident.updated_at,
      message: 'Tracking pipeline activated',
    })
  }

  for (const s of sightings) {
    entries.push({
      id: `sighting-${s.id}`,
      type: 'SIGHTING_DETECTED',
      timestamp: s.timestamp,
      message: `Alert: potential match detected — ${Math.round(s.confidence * 100)}% confidence`,
      sighting: s,
    })
    if (s.status === 'CONFIRMED') {
      entries.push({
        id: `confirmed-${s.id}`,
        type: 'ALERT_VERIFIED',
        timestamp: new Date(new Date(s.timestamp).getTime() + 360_000).toISOString(),
        message: 'Alert confirmed — identity match verified',
      })
    }
  }

  if (incident.status === 'RESOLVED' || incident.status === 'CLOSED') {
    entries.push({
      id: `closed-${incident.id}`,
      type: 'CASE_CLOSED',
      timestamp: incident.updated_at,
      message: `Case ${incident.ref} resolved and closed`,
    })
  }

  return entries.sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  )
}

// ── activity feed (pure function, not a hook) ─────────────────────────────────

export function deriveActivityFeed(incidents: Incident[]): ActivityEvent[] {
  const events: ActivityEvent[] = []
  for (const inc of incidents) {
    events.push({
      id: `c-${inc.id}`,
      type: 'CASE_CREATED',
      incident_ref: inc.ref,
      message: `Case ${inc.ref} opened — ${inc.title}`,
      timestamp: inc.created_at,
    })
    if (inc.alert_count > 0) {
      events.push({
        id: `a-${inc.id}`,
        type: 'SIGHTING_DETECTED',
        incident_ref: inc.ref,
        message: `Alert: potential match detected on ${inc.ref}`,
        timestamp: inc.updated_at,
      })
    }
    if (inc.status === 'RESOLVED' || inc.status === 'CLOSED') {
      events.push({
        id: `r-${inc.id}`,
        type: 'CASE_CLOSED',
        incident_ref: inc.ref,
        message: `Case ${inc.ref} resolved and closed`,
        timestamp: inc.updated_at,
      })
    }
  }
  return events
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
    .slice(0, 12)
}

// ── module-level caches (survive component remounts / navigation) ─────────────

let _incidentsCache: Incident[] = []

// ── useIncidents ──────────────────────────────────────────────────────────────

export function useIncidents() {
  const { accessMode, incUrl } = useAppStore()
  const [data, setData] = useState<Incident[]>(_incidentsCache)
  const [loading, setLoading] = useState(_incidentsCache.length === 0)
  const [error, setError] = useState<string | null>(null)
  const hasLoadedRef = useRef(false)

  const load = useCallback(async () => {
    if (accessMode === 'MOCK') {
      setData(mockIncidents)
      setLoading(false)
      hasLoadedRef.current = true
      return
    }
    try {
      // background polls are silent — only the first fetch shows the spinner
      if (!hasLoadedRef.current) setLoading(true)
      const raw = await createApiClient(incUrl).get<Raw[]>('/incidents')
      const normalized = raw.map((r, i) => normalizeIncident(r, i))
      _incidentsCache = normalized
      setData(normalized)
      setError(null)
      hasLoadedRef.current = true
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [accessMode, incUrl])

  useEffect(() => {
    hasLoadedRef.current = false
    load()
    if (accessMode === 'ADMIN') {
      const t = setInterval(load, 10_000)
      return () => clearInterval(t)
    }
  }, [load, accessMode])

  return { data, loading, error, refetch: load }
}

// ── useIncidentDetail ─────────────────────────────────────────────────────────

export function useIncidentDetail(id: string | undefined) {
  const { accessMode, incUrl } = useAppStore()
  const [incident, setIncident] = useState<Incident | null>(null)
  const [persons, setPersons] = useState<Person[]>([])
  const [sightings, setSightings] = useState<Sighting[]>([])
  const [timeline, setTimeline] = useState<TimelineEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // Track whether initial data has been loaded — background polls skip the spinner
  const hasLoadedRef = useRef(false)

  const load = useCallback(async () => {
    if (!id) return

    if (accessMode === 'MOCK') {
      const inc = mockIncidents.find((i) => i.id === id) ?? null
      setIncident(inc)
      setPersons(mockPersonsByIncident[id] ?? [])
      setSightings(mockSightingsByIncident[id] ?? [])
      setTimeline(mockTimelinesByIncident[id] ?? [])
      setLoading(false)
      hasLoadedRef.current = true
      return
    }

    try {
      // Only show the loading overlay on first fetch — subsequent polls are silent
      if (!hasLoadedRef.current) setLoading(true)
      const client = createApiClient(incUrl)
      const [rawInc, rawPersons, rawSightings] = await Promise.all([
        client.get<Raw>(`/incidents/${id}`),
        client.get<Raw[]>(`/incidents/${id}/persons`),
        client.get<Raw[]>(`/incidents/${id}/sightings`),
      ])
      const inc = normalizeIncident(rawInc, 0)
      const ps = rawPersons.map((p) => normalizePerson(p, id))
      const ss = rawSightings.map(normalizeSighting)
      setIncident(inc)
      setPersons(ps)
      setSightings(ss)
      setTimeline(buildTimeline(inc, ps, ss))
      setError(null)
      hasLoadedRef.current = true
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [id, accessMode, incUrl])

  useEffect(() => {
    // Reset on id/mode change so the loading overlay shows for the new case
    hasLoadedRef.current = false
    setLoading(true)
    setIncident(null)
    load()
    if (accessMode === 'ADMIN') {
      const t = setInterval(load, 8_000)
      return () => clearInterval(t)
    }
  }, [load, accessMode])

  return { incident, persons, sightings, timeline, loading, error, refetch: load }
}

// ── useSystemMetrics ──────────────────────────────────────────────────────────

export function useSystemMetrics() {
  const { accessMode, backendUrl } = useAppStore()
  const [data, setData] = useState<SystemMetrics>(mockSystemMetrics)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [fpsHistory, setFpsHistory] = useState<SparkPoint[]>(mockFpsHistory)
  const lastHistoryTs = useRef(0)

  const fetch = useCallback(async () => {
    if (accessMode === 'MOCK') {
      setData(mockSystemMetrics)
      setFpsHistory(mockFpsHistory)
      setLoading(false)
      return
    }
    try {
      const raw = await createApiClient(backendUrl).get<Raw>('/observability/metrics')
      const m: SystemMetrics = {
        fps: Number(raw.fps ?? raw.average_processing_fps ?? 0),
        detector_latency_ms: Number(raw.detector_latency_ms ?? raw.detector_runtime_ms ?? 0),
        gpu_status:
          (raw.gpu_status as SystemMetrics['gpu_status']) ??
          (Number(raw.hardware_backend_type) === 1 ? 'OK' : 'UNAVAILABLE'),
        hardware_backend_type: Number(raw.hardware_backend_type ?? 0),
        queue_depth: Number(raw.queue_depth ?? 0),
        identity_switch_rate: Number(raw.identity_switch_rate ?? 0),
        active_tracks: Number(raw.active_tracks ?? 0),
        cpu_percent: Number(raw.cpu_percent ?? 0),
        memory_mb: Number(raw.memory_mb ?? 0),
        stable_matches: Number(raw.stable_matches ?? 0),
        validator_rejection_rate: Number(raw.validator_rejection_rate ?? 0),
        confirmation_rate: Number(raw.confirmation_rate ?? 0),
        uptime_seconds: Number(raw.uptime_seconds ?? 0),
      }
      setData(m)
      setError(null)
      const now = Date.now()
      if (now - lastHistoryTs.current > 60_000) {
        lastHistoryTs.current = now
        const t = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        setFpsHistory((prev) => [...prev.slice(-23), { t, v: Math.round(m.fps) }])
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [accessMode, backendUrl])

  useEffect(() => {
    fetch()
    if (accessMode === 'ADMIN') {
      const t = setInterval(fetch, 3_000)
      return () => clearInterval(t)
    }
  }, [fetch, accessMode])

  return { data, loading, error, fpsHistory }
}

// ── useCameras ────────────────────────────────────────────────────────────────

export function useCameras() {
  const { accessMode, backendUrl } = useAppStore()
  const [data, setData] = useState<Camera[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (accessMode === 'MOCK') {
      setData(mockCameras)
      setLoading(false)
      return
    }
    createApiClient(backendUrl)
      .get<Raw[]>('/cameras')
      .then((raw) => {
        setData(
          raw.map((c) => ({
            id: String(c.id ?? ''),
            name: String(c.name ?? ''),
            location: String(c.location ?? ''),
            status: (c.status as Camera['status']) ?? 'INACTIVE',
            fps: c.fps != null ? Number(c.fps) : undefined,
          }))
        )
        setError(null)
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [accessMode, backendUrl])

  return { data, loading, error }
}

// ── useVideoJob ───────────────────────────────────────────────────────────────

export interface VideoJobProgress {
  status: 'queued' | 'processing' | 'completed' | 'failed'
  total_frames: number
  processed_frames: number
  alerts_created: number
  avg_fps: number
  processing_duration_seconds: number
  total_faces_detected: number
  total_faces_rejected: number
  blur_rejections: number
  error_message?: string
}

export function useVideoJob(jobId: string | null) {
  const { backendUrl } = useAppStore()
  const [progress, setProgress] = useState<VideoJobProgress | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!jobId) return
    const client = createApiClient(backendUrl)
    const backendBase = backendUrl.replace(/\/api\/v1\/?$/, '')
    let stopped = false
    let previewFetched = false

    const poll = async () => {
      if (stopped) return
      try {
        const s = await client.get<Raw>(`/videos/processing-status/${jobId}`)
        if (stopped) return
        setProgress({
          status: String(s.status ?? 'queued') as VideoJobProgress['status'],
          total_frames: Number(s.total_frames ?? 0),
          processed_frames: Number(s.processed_frames ?? 0),
          alerts_created: Number(s.alerts_created ?? 0),
          avg_fps: Number(s.avg_fps ?? 0),
          processing_duration_seconds: Number(s.processing_duration_seconds ?? 0),
          total_faces_detected: Number(s.total_faces_detected ?? 0),
          total_faces_rejected: Number(s.total_faces_rejected ?? 0),
          blur_rejections: Number(s.blur_rejections ?? 0),
          error_message: s.error_message ? String(s.error_message) : undefined,
        })
        setError(null)

        if (s.status === 'completed' && !previewFetched) {
          previewFetched = true
          try {
            const p = await client.get<Raw>(`/videos/processing-preview/${jobId}`)
            const pUrl = String(p.preview_url ?? p.preview_path ?? '')
            if (pUrl && !stopped) {
              // ensure leading slash so backendBase + path concatenates correctly
              const rel = pUrl.startsWith('/') ? pUrl : `/${pUrl}`
              setPreviewUrl(pUrl.startsWith('http') ? pUrl : `${backendBase}${rel}`)
            }
          } catch { /* preview is optional */ }
        }

        if (s.status === 'completed' || s.status === 'failed') {
          clearInterval(t)
        }
      } catch (e) {
        if (!stopped) setError((e as Error).message)
      }
    }

    poll()
    const t = setInterval(poll, 2_000)
    return () => {
      stopped = true
      clearInterval(t)
    }
  }, [jobId, backendUrl])

  return { progress, previewUrl, error }
}

// ── useHealthCheck ────────────────────────────────────────────────────────────

export function useHealthCheck(url: string) {
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking')

  const check = useCallback(async () => {
    setStatus('checking')
    try {
      await createApiClient(url).get('/health')
      setStatus('online')
    } catch {
      setStatus('offline')
    }
  }, [url])

  useEffect(() => {
    check()
  }, [check])

  return { status, recheck: check }
}

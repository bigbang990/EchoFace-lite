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

export const mockIncidents: Incident[] = [
  {
    id: 'a1b2c3d4-0001-4e5f-8a9b-c0d1e2f30001',
    ref: 'INC-001',
    title: 'Missing: Sarah Chen',
    status: 'TRACKING',
    created_at: '2026-06-13T09:00:00Z',
    updated_at: '2026-06-13T10:45:00Z',
    description:
      'Sarah Chen, 24, was last seen near Whitechapel Market on the evening of 12 June. She was wearing a blue jacket and carrying a grey backpack. Family reported her missing at 23:00.',
    last_seen_location: 'Whitechapel Market, London E1',
    last_seen_at: '2026-06-12T21:30:00Z',
    person_count: 1,
    alert_count: 1,
  },
  {
    id: 'a1b2c3d4-0002-4e5f-8a9b-c0d1e2f30002',
    ref: 'INC-002',
    title: 'Missing: Marcus Rodriguez',
    status: 'OPEN',
    created_at: '2026-06-13T11:20:00Z',
    updated_at: '2026-06-13T11:35:00Z',
    description:
      'Marcus Rodriguez, 31, failed to return home from his evening run. Last seen on Waterloo Bridge around 20:15 on 12 June. Running attire, red headphones.',
    last_seen_location: 'Waterloo Bridge, London SE1',
    last_seen_at: '2026-06-12T20:15:00Z',
    person_count: 1,
    alert_count: 0,
  },
  {
    id: 'a1b2c3d4-0003-4e5f-8a9b-c0d1e2f30003',
    ref: 'INC-003',
    title: 'Missing: Priya Sharma',
    status: 'RESOLVED',
    created_at: '2026-06-11T14:00:00Z',
    updated_at: '2026-06-12T16:30:00Z',
    description:
      'Priya Sharma, 19, first-year student reported missing from university campus. Located safe via CCTV match at King\'s Cross station. Case resolved.',
    last_seen_location: 'UCL Main Campus, Gower Street, London',
    last_seen_at: '2026-06-11T13:00:00Z',
    person_count: 1,
    alert_count: 2,
  },
]

export const mockPersonsByIncident: Record<string, Person[]> = {
  'a1b2c3d4-0001-4e5f-8a9b-c0d1e2f30001': [
    {
      id: 'p0001-0000-0000-0000-000000000001',
      name: 'Sarah Chen',
      age: 24,
      gender: 'Female',
      description: 'Blue jacket, grey backpack. East Asian appearance, shoulder-length black hair.',
      incident_id: 'a1b2c3d4-0001-4e5f-8a9b-c0d1e2f30001',
      enrolled_at: '2026-06-13T09:05:00Z',
    },
  ],
  'a1b2c3d4-0002-4e5f-8a9b-c0d1e2f30002': [
    {
      id: 'p0002-0000-0000-0000-000000000001',
      name: 'Marcus Rodriguez',
      age: 31,
      gender: 'Male',
      description: 'Red wireless headphones, dark running gear. Latino appearance, short curly hair.',
      incident_id: 'a1b2c3d4-0002-4e5f-8a9b-c0d1e2f30002',
      enrolled_at: '2026-06-13T11:35:00Z',
    },
  ],
  'a1b2c3d4-0003-4e5f-8a9b-c0d1e2f30003': [
    {
      id: 'p0003-0000-0000-0000-000000000001',
      name: 'Priya Sharma',
      age: 19,
      gender: 'Female',
      description: 'University backpack, casual clothes. South Asian appearance, long dark hair in ponytail.',
      incident_id: 'a1b2c3d4-0003-4e5f-8a9b-c0d1e2f30003',
      enrolled_at: '2026-06-11T14:10:00Z',
    },
  ],
}

const pendingSighting: Sighting = {
  id: 's0001-0000-0000-0000-000000000001',
  incident_id: 'a1b2c3d4-0001-4e5f-8a9b-c0d1e2f30001',
  person_id: 'p0001-0000-0000-0000-000000000001',
  person_name: 'Sarah Chen',
  confidence: 0.923,
  camera_id: 'cam-001',
  source_name: 'E1_Market_CCTV_01.mp4',
  timestamp: '2026-06-13T10:23:14Z',
  status: 'PENDING',
  frame_index: 18432,
}

export const mockSightingsByIncident: Record<string, Sighting[]> = {
  'a1b2c3d4-0001-4e5f-8a9b-c0d1e2f30001': [pendingSighting],
  'a1b2c3d4-0002-4e5f-8a9b-c0d1e2f30002': [],
  'a1b2c3d4-0003-4e5f-8a9b-c0d1e2f30003': [
    {
      id: 's0003-0000-0000-0000-000000000001',
      incident_id: 'a1b2c3d4-0003-4e5f-8a9b-c0d1e2f30003',
      person_id: 'p0003-0000-0000-0000-000000000001',
      person_name: 'Priya Sharma',
      confidence: 0.961,
      camera_id: 'cam-kx-04',
      source_name: "King's_Cross_Gate_B.mp4",
      timestamp: '2026-06-12T15:44:22Z',
      status: 'CONFIRMED',
      frame_index: 41280,
    },
    {
      id: 's0003-0000-0000-0000-000000000002',
      incident_id: 'a1b2c3d4-0003-4e5f-8a9b-c0d1e2f30003',
      person_id: 'p0003-0000-0000-0000-000000000001',
      person_name: 'Priya Sharma',
      confidence: 0.887,
      camera_id: 'cam-kx-07',
      source_name: "King's_Cross_Platform_3.mp4",
      timestamp: '2026-06-12T15:52:08Z',
      status: 'CONFIRMED',
      frame_index: 44916,
    },
  ],
}

export const mockTimelinesByIncident: Record<string, TimelineEntry[]> = {
  'a1b2c3d4-0001-4e5f-8a9b-c0d1e2f30001': [
    {
      id: 'tl-001-001',
      type: 'CASE_CREATED',
      timestamp: '2026-06-13T09:00:00Z',
      message: 'Case INC-001 opened — Missing: Sarah Chen',
    },
    {
      id: 'tl-001-002',
      type: 'PERSON_ENROLLED',
      timestamp: '2026-06-13T09:05:22Z',
      message: 'Person enrolled: Sarah Chen — 3 reference photos uploaded',
    },
    {
      id: 'tl-001-003',
      type: 'EMBEDDINGS_GENERATED',
      timestamp: '2026-06-13T09:05:41Z',
      message: 'Face embeddings generated — 3/3 photos processed, tracking profile active',
    },
    {
      id: 'tl-001-004',
      type: 'TRACKING_STARTED',
      timestamp: '2026-06-13T09:10:05Z',
      message: 'Tracking activated on source: E1_Market_CCTV_01.mp4',
    },
    {
      id: 'tl-001-005',
      type: 'SIGHTING_DETECTED',
      timestamp: '2026-06-13T10:23:14Z',
      message: 'Alert: potential match detected — 92.3% confidence',
      sighting: pendingSighting,
    },
    {
      id: 'tl-001-006',
      type: 'COMMENT_ADDED',
      timestamp: '2026-06-13T10:45:00Z',
      message: 'Operator note: "Match appears to be SW corridor near exit. Forwarding to field unit for verification."',
      comment: 'Match appears to be SW corridor near exit. Forwarding to field unit for verification.',
    },
  ],
  'a1b2c3d4-0002-4e5f-8a9b-c0d1e2f30002': [
    {
      id: 'tl-002-001',
      type: 'CASE_CREATED',
      timestamp: '2026-06-13T11:20:00Z',
      message: 'Case INC-002 opened — Missing: Marcus Rodriguez',
    },
    {
      id: 'tl-002-002',
      type: 'PERSON_ENROLLED',
      timestamp: '2026-06-13T11:35:14Z',
      message: 'Person enrolled: Marcus Rodriguez — 2 reference photos uploaded',
    },
    {
      id: 'tl-002-003',
      type: 'EMBEDDINGS_GENERATED',
      timestamp: '2026-06-13T11:35:28Z',
      message: 'Face embeddings generated — 2/2 photos processed, awaiting source video',
    },
  ],
  'a1b2c3d4-0003-4e5f-8a9b-c0d1e2f30003': [
    {
      id: 'tl-003-001',
      type: 'CASE_CREATED',
      timestamp: '2026-06-11T14:00:00Z',
      message: 'Case INC-003 opened — Missing: Priya Sharma',
    },
    {
      id: 'tl-003-002',
      type: 'PERSON_ENROLLED',
      timestamp: '2026-06-11T14:10:33Z',
      message: 'Person enrolled: Priya Sharma — 4 reference photos uploaded',
    },
    {
      id: 'tl-003-003',
      type: 'EMBEDDINGS_GENERATED',
      timestamp: '2026-06-11T14:10:52Z',
      message: "Face embeddings generated — 4/4 photos processed",
    },
    {
      id: 'tl-003-004',
      type: 'TRACKING_STARTED',
      timestamp: '2026-06-11T14:15:00Z',
      message: "Tracking activated on 2 CCTV sources: King's Cross Gate B, Platform 3",
    },
    {
      id: 'tl-003-005',
      type: 'SIGHTING_DETECTED',
      timestamp: '2026-06-12T15:44:22Z',
      message: "Alert: potential match at King's Cross Gate B — 96.1% confidence",
      sighting: {
        id: 's0003-0000-0000-0000-000000000001',
        incident_id: 'a1b2c3d4-0003-4e5f-8a9b-c0d1e2f30003',
        person_id: 'p0003-0000-0000-0000-000000000001',
        person_name: 'Priya Sharma',
        confidence: 0.961,
        camera_id: 'cam-kx-04',
        source_name: "King's_Cross_Gate_B.mp4",
        timestamp: '2026-06-12T15:44:22Z',
        status: 'CONFIRMED',
        frame_index: 41280,
      },
    },
    {
      id: 'tl-003-006',
      type: 'ALERT_VERIFIED',
      timestamp: '2026-06-12T15:50:00Z',
      message: 'Alert confirmed — identity match verified by operator',
    },
    {
      id: 'tl-003-007',
      type: 'SIGHTING_DETECTED',
      timestamp: '2026-06-12T15:52:08Z',
      message: "Alert: second match at King's Cross Platform 3 — 88.7% confidence",
      sighting: {
        id: 's0003-0000-0000-0000-000000000002',
        incident_id: 'a1b2c3d4-0003-4e5f-8a9b-c0d1e2f30003',
        person_id: 'p0003-0000-0000-0000-000000000001',
        person_name: 'Priya Sharma',
        confidence: 0.887,
        camera_id: 'cam-kx-07',
        source_name: "King's_Cross_Platform_3.mp4",
        timestamp: '2026-06-12T15:52:08Z',
        status: 'CONFIRMED',
        frame_index: 44916,
      },
    },
    {
      id: 'tl-003-008',
      type: 'ALERT_VERIFIED',
      timestamp: '2026-06-12T15:55:00Z',
      message: 'Second alert confirmed — location relayed to Metropolitan Police',
    },
    {
      id: 'tl-003-009',
      type: 'CASE_CLOSED',
      timestamp: '2026-06-12T16:30:00Z',
      message: 'Case resolved — subject located safe at King\'s Cross station. Case INC-003 closed.',
    },
  ],
}

export const mockCameras: Camera[] = [
  { id: 'cam-001', name: 'E1 Market CCTV 01', location: 'Whitechapel Market, E1', status: 'ACTIVE', fps: 25 },
  { id: 'cam-002', name: 'Waterloo Bridge North', location: 'Waterloo Bridge, SE1', status: 'INACTIVE' },
  { id: 'cam-kx-04', name: "King's Cross Gate B", location: "King's Cross Station, N1C", status: 'INACTIVE' },
]

export const mockActivityFeed: ActivityEvent[] = [
  {
    id: 'act-001',
    type: 'SIGHTING_DETECTED',
    incident_ref: 'INC-001',
    message: 'Alert: potential match for Sarah Chen detected at E1 Market CCTV — 92% confidence',
    timestamp: '2026-06-13T10:23:14Z',
  },
  {
    id: 'act-002',
    type: 'COMMENT_ADDED',
    incident_ref: 'INC-001',
    message: 'Operator comment added on INC-001',
    timestamp: '2026-06-13T10:45:00Z',
  },
  {
    id: 'act-003',
    type: 'PERSON_ENROLLED',
    incident_ref: 'INC-002',
    message: 'Person enrolled: Marcus Rodriguez (INC-002)',
    timestamp: '2026-06-13T11:35:14Z',
  },
  {
    id: 'act-004',
    type: 'CASE_CREATED',
    incident_ref: 'INC-002',
    message: 'New case created: INC-002 Missing: Marcus Rodriguez',
    timestamp: '2026-06-13T11:20:00Z',
  },
  {
    id: 'act-005',
    type: 'TRACKING_STARTED',
    incident_ref: 'INC-001',
    message: 'Tracking engine activated for INC-001 on E1_Market_CCTV_01.mp4',
    timestamp: '2026-06-13T09:10:05Z',
  },
  {
    id: 'act-006',
    type: 'CASE_CREATED',
    incident_ref: 'INC-001',
    message: 'New case created: INC-001 Missing: Sarah Chen',
    timestamp: '2026-06-13T09:00:00Z',
  },
  {
    id: 'act-007',
    type: 'ALERT_VERIFIED',
    incident_ref: 'INC-003',
    message: "Alert confirmed for INC-003 — Priya Sharma located at King's Cross",
    timestamp: '2026-06-12T16:30:00Z',
  },
  {
    id: 'act-008',
    type: 'CASE_CLOSED',
    incident_ref: 'INC-003',
    message: 'Case INC-003 resolved and closed — subject located safe',
    timestamp: '2026-06-12T16:30:00Z',
  },
]

export const mockSystemMetrics: SystemMetrics = {
  fps: 81.4,
  detector_latency_ms: 13.2,
  gpu_status: 'OK',
  hardware_backend_type: 1,
  queue_depth: 3,
  identity_switch_rate: 0,
  active_tracks: 7,
  cpu_percent: 34.2,
  memory_mb: 2418,
  stable_matches: 50,
  validator_rejection_rate: 0.08,
  confirmation_rate: 0.94,
  uptime_seconds: 86412,
}

export const mockFpsHistory: SparkPoint[] = Array.from({ length: 24 }, (_, i) => ({
  t: `${String(i).padStart(2, '0')}:00`,
  v: Math.round(72 + Math.random() * 18),
}))

export function getIncidentById(id: string) {
  return mockIncidents.find((inc) => inc.id === id) ?? null
}

export function nextIncidentRef() {
  return `INC-${String(mockIncidents.length + 1).padStart(3, '0')}`
}

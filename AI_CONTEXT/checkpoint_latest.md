## Checkpoint — 2026-06-13 — Phase A: INC API consolidated to single-server

### Phase
Frontend v1.3 — single-server setup; alerts pipeline wired end-to-end

### Status
complete — all routes (engine + INC) served from one uvicorn process on :8000;
inc_server.py kept intact for future Phase B standalone use;
colab_start.py simplified to single ngrok tunnel on :8000;
video detections create Sighting rows so they appear in the case timeline

### What was built
A standalone React + TypeScript frontend in `frontend/` — dark, professional
investigation platform aesthetic matching the spec.

### Stack
- Vite 5 + React 18 + TypeScript
- Tailwind CSS 3 (dark theme, custom gray-950, JetBrains Mono font)
- Framer Motion 11 (animations: AccessGate shake, Timeline stagger, ProcessingSequence)
- Zustand 4 (accessMode + activeCaseId store)
- React Router 6 (client-side routing)
- Recharts 2 (SystemHealth FPS sparkline)
- Lucide React (icons)

### Access gate codes
- `MOCK`  → MOCK mode  (all screens, static mock data, violet badge)
- `DEMO`  → DEMO mode  (all screens, real API, cyan badge)
- `ADMIN` → ADMIN mode (all screens + System Health, real-time polling, amber badge)

### Files (frontend/)
- package.json, vite.config.ts, tailwind.config.js, postcss.config.js,
  tsconfig.json, index.html, public/favicon.svg
- src/index.css, src/main.tsx, src/App.tsx
- src/types/index.ts          — Incident, Person (+ source_image_path), Sighting,
                                 Camera, ActivityEvent, TimelineEntry, SystemMetrics, SparkPoint
- src/store/appStore.ts       — accessMode, activeCaseId, activeJobId, backendName, backendUrl,
                                 BACKENDS registry (Local CPU + Colab GPU)
- src/mock/data.ts            — 3 incidents (INC-001 TRACKING, INC-002 OPEN,
                                 INC-003 RESOLVED), persons, sightings, timelines,
                                 activity feed, metrics, fps history
- src/api/client.ts           — thin fetch wrapper: createApiClient(baseUrl)
- src/api/hooks.ts            — useIncidents, useIncidentDetail, useSystemMetrics,
                                 useCameras, useVideoJob, useHealthCheck, deriveActivityFeed,
                                 normalizers (display_name/notes/source_image_path), buildTimeline
- src/components/AccessGate.tsx        — MOCK / DEMO / ADMIN codes, shake animation,
                                          mode label on success, favicon.svg logo
- src/components/Layout.tsx            — sidebar + MOCK/DEMO/ADMIN badge + backend btn
- src/components/BackendPanel.tsx      — slide-over backend registry with health check,
                                          custom URL input
- src/components/StatusIndicator.tsx   — dot + label + ONLINE/OFFLINE/DEGRADED badge
- src/components/Timeline.tsx          — stagger animation, expandable alert cards
- src/components/ProcessingSequence.tsx — STABLE — do not modify timing
- src/pages/Overview.tsx       — useIncidents + useSystemMetrics + useCameras hooks;
                                   ADMIN: 4 live telemetry tiles + "LIVE · 3s" badge
- src/pages/CreateCase.tsx     — 3-step form → real 4-step API progress tracker →
                                   navigate to created case (DEMO/ADMIN) or success
                                   screen (MOCK). Photo rejection warning + agent
                                   confirmation dialog. personId from pData.person.id
- src/pages/Operations.tsx     — real video upload via POST /videos/upload-and-process,
                                   persistent job in Zustand, useVideoJob polling,
                                   preview video on completion
- src/pages/CaseList.tsx       — useIncidents hook; loading/error states
- src/pages/CaseWorkspace.tsx  — useIncidentDetail hook; 3-column layout; PersonAvatar
                                   shows enrolled photo (source_image_path via
                                   /data/uploads/ static mount); close/pause buttons
                                   send PATCH /incidents/{id}/status to backend
- src/pages/SystemHealth.tsx   — useSystemMetrics (3s polling in ADMIN); CSV + JSON export

### API hooks — MOCK mode returns mock data, DEMO/ADMIN hit real FastAPI
- useIncidents()        → incidents[], ADMIN polls every 10s
- useIncidentDetail(id) → incident, persons[], sightings[], timeline[], ADMIN polls 8s
- useSystemMetrics()    → SystemMetrics + fpsHistory SparkPoint[], ADMIN polls 3s
- useCameras()          → Camera[]
- useHealthCheck(url)   → 'checking' | 'online' | 'offline'

### Backend wiring — no component changes needed
When backend is running at http://127.0.0.1:8000 (or Colab ngrok URL):
- Select backend in BackendPanel (gear icon in sidebar, hidden in MOCK mode)
- Enter DEMO or ADMIN at access gate
- All pages auto-fetch from /api/v1/incidents, /api/v1/incidents/:id/persons,
  /api/v1/incidents/:id/sightings, /api/v1/metrics

### Mock data shape matches real API
Incident { id (UUID), ref, title, status, created_at, updated_at,
           description, last_seen_location, last_seen_at, person_count, alert_count }
Person   { id, name, age, gender, description, incident_id, enrolled_at,
           source_image_path? }
Sighting { id, incident_id, person_id, person_name, confidence, camera_id,
           source_name, timestamp, status, frame_index }

### Critical enrollment bug (FIXED)
Backend PersonEnrollOut = { person: PersonOut, deduplicated: bool }.
The person ID is at `pData.person.id`, NOT `pData.id`.
The old code read `pData.id` (undefined) → personId = '' → link skipped →
person existed in DB but not linked to any incident → gallery was empty →
"No enrolled persons in gallery" on video scan.

### Backend schemas (key for API wiring)
- POST /persons → PersonEnrollOut { person: PersonOut, deduplicated: bool }
  PersonOut { id, display_name, notes, source_image_path, source_image_hash, created_at }
- POST /persons/{id}/photos → PersonEnrollMultiOut { person, photos_accepted, photos_rejected, rejection_reasons[] }
- PATCH /incidents/{id}/status → body: { status: "open" | "active" | "closed" }
- source_image_path format: "data/uploads/{uuid}.ext" — served at {backendBase}/data/uploads/{uuid}.ext

### Static mounts in main.py
- /data/previews → settings.resolved_previews_dir()
- /data/debug/rejected_faces → settings.resolved_rejected_faces_dir()
- /data/uploads → settings.resolved_uploads_dir()  ← added Phase 3

### Embedding lifecycle
Gallery = persons linked to OPEN incidents. Closing an incident (PATCH status: "closed")
removes its persons from the gallery automatically. Reopening (status: "open") re-enables.
No separate embedding delete step needed — controlled entirely by incident status.

### To start
```
cd frontend
npm install
npm run dev
# open http://localhost:5173
# enter DEMO or ADMIN at the access gate
```
Node.js 18+ must be installed (nodejs.org).

### GPU baseline metrics (still valid from Phase 8B)
- hardware_backend_type: 1 (GPU)
- stable_matches: 50
- identity_switch_rate: 0
- average_processing_fps: 81
- detector_runtime_ms: 13.2ms

### Previous phases on this branch
- Phase 8B: incident-person link endpoints
- Phase 8C: cameras.py + incidents.py routers, 7 new schemas
- Phase 8A: ExperimentCoordinator extracted
- Phase 7B: session lifecycle isolation, multi-photo enrollment

### Architecture — single server

```
EchoFace :8000  — all routes: engine + INC (incidents, persons, sightings)
                  /data/uploads static mount for enrolled person photos
                  SQLite/PostgreSQL database
```

Start commands:
```
# terminal 1 — single server (all routes)
uvicorn ecoface_lite.api.main:app --port 8000 --reload

# terminal 2 — frontend
cd frontend && npm run dev
```

Colab: `python scripts/colab_start.py` — one process, one ngrok tunnel.
Set both `backendUrl` AND `incUrl` in BackendPanel to the same ngrok URL.

Future Phase B: `inc_server.py` is ready to run standalone on :8001 with
a second ngrok tunnel. See AI_CONTEXT/roadmap.md for Phase B plan.

### Alerts pipeline (fixed)
`video_service.py` now: DetectionEvent → `session.flush()` → query
`incident_persons` → create `Sighting(incident_id, detection_id)`.
`GET /incidents/{id}/sightings` eager-loads detection+person, returns
enriched `SightingOut` with person_name, confidence, source_name, frame_index.
Frontend `normalizeSighting` uses `raw.created_at` as timestamp fallback.

### Next target
- Phase B: INC server writes sightings via its own endpoint (engine POSTs to INC API
  instead of writing to DB directly) — full service separation
- Same-person photo verification during enrollment
- Confirm/reject sighting wired to real API

### Branch
phase8-pipeline-decompose

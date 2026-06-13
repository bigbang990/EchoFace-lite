## Checkpoint — 2026-06-13 — Frontend Phase 2 complete (backend-wired)

### Phase
Frontend v1.1 — Backend API wiring + real-time ADMIN mode + metrics export

### Status
complete — all pages wired to API hooks; dev server (`cd frontend && npm run dev`)

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
- src/types/index.ts          — Incident, Person, Sighting, Camera, ActivityEvent,
                                 TimelineEntry, SystemMetrics, SparkPoint
- src/store/appStore.ts       — accessMode, activeCaseId, backendName, backendUrl,
                                 BACKENDS registry (Local CPU + Colab GPU)
- src/mock/data.ts            — 3 incidents (INC-001 TRACKING, INC-002 OPEN,
                                 INC-003 RESOLVED), persons, sightings, timelines,
                                 activity feed, metrics, fps history
- src/api/client.ts           — thin fetch wrapper: createApiClient(baseUrl)
- src/api/hooks.ts            — useIncidents, useIncidentDetail, useSystemMetrics,
                                 useCameras, useHealthCheck, deriveActivityFeed,
                                 normalizers, buildTimeline
- src/components/AccessGate.tsx        — MOCK / DEMO / ADMIN codes, shake animation,
                                          mode label on success, favicon.svg logo
- src/components/Layout.tsx            — sidebar + MOCK/DEMO/ADMIN badge + backend btn
- src/components/BackendPanel.tsx      — slide-over backend registry (mirrors backend_registry.py)
                                          with health check per entry, custom URL input
- src/components/StatusIndicator.tsx   — dot + label + ONLINE/OFFLINE/DEGRADED badge
- src/components/Timeline.tsx          — stagger animation, expandable alert cards
- src/components/ProcessingSequence.tsx — STABLE — do not modify timing
- src/pages/Overview.tsx       — useIncidents + useSystemMetrics + useCameras hooks;
                                   ADMIN: 4 live telemetry tiles + "LIVE · 3s" badge
- src/pages/CreateCase.tsx     — 3-step form → ProcessingSequence → success screen
- src/pages/Operations.tsx     — drag/drop video upload, live mock stat ticker
- src/pages/CaseList.tsx       — useIncidents hook; loading/error states
- src/pages/CaseWorkspace.tsx  — useIncidentDetail hook; 3-column layout; local state
                                   for confirm/reject/comment/status actions
- src/pages/SystemHealth.tsx   — useSystemMetrics (3s polling in ADMIN); CSV + JSON
                                   export buttons (URL.createObjectURL); LIVE badge

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
Person   { id, name, age, gender, description, incident_id, enrolled_at }
Sighting { id, incident_id, person_id, person_name, confidence, camera_id,
           source_name, timestamp, status, frame_index }

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

### Next target
Wire frontend to real FastAPI backend — replace mock/data.ts with
fetch/SWR calls to /api/v1/incidents, /api/v1/incidents/:id/persons,
/api/v1/incidents/:id/sightings, /api/v1/observability/*

### Branch
phase8-pipeline-decompose

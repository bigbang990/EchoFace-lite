# Checkpoint — 2026-06-14 — UI: Administration panel + VSL source selector

## Phase
Frontend UI — Administration panel, Operations source selector, Live Feed stub
Branch: `vsl-phase3-multi-source`
All prior VSL phases (1–5) intact and verified.

## Regression baseline metrics (30/30 pass — unchanged)
identity_switch_rate: 0.000
stable_matches: green
confirmation_rate: green
validator_rejection_rate: green
bbox_jitter: green
Test suite: 30 tests, 0 failed

---

## UI changes this session

### Task 1 — Administration tab unlocked (Layout.tsx)
- `ShieldOff` disabled block removed; replaced with active NavLink `/administration`
- `Settings` icon replaces `ShieldOff` in import list
- Tab appears in ADMIN and MOCK sidebar under ADMIN section

### Task 2 — Administration page (new: frontend/src/pages/Administration.tsx)
Two-panel layout:
- **Sites & Zones**: GET/POST/DELETE /sites, GET/POST/DELETE /zones
  Expandable site rows, zone rows with camera count + online count
  Inline add-site and add-zone forms (Enter to save, Escape to cancel)
  MOCK mode: pre-seeded data (HQ Building, Warehouse A, 3 zones)
- **Camera Sources**: GET/POST/DELETE /cameras, GET /cameras/health-summary
  Health summary bar (online/offline/reconnecting)
  Camera table with name, zone, type, status badge, delete button
  Register Camera modal: name, source_type, stream_url, site, zone, direction,
    trust_level, overlap_group, retention_days
    Zone select auto-populates after site is chosen (GET /sites/{id}/zones)

### Task 3 — Operations source selector (Operations.tsx)
- Source type tabs: File Upload | Registered Camera | RTSP URL
  Only visible when no job is running and not in running mock state
- File Upload: existing behavior 100% unchanged
- Registered Camera panel: dropdown from useCameras(), selected camera card,
  Activate button → POST /cameras/{id}/start-tracking (backend stub)
- RTSP URL panel: text input, Test button → POST /cameras/test-rtsp,
  shows ✓ Connected / ✗ Connection failed, Activate button
- Live Feed button always in header → window.open('/live-feed', ...)

### Task 4 — LiveFeed.tsx stub (new: frontend/src/pages/LiveFeed.tsx)
- Full-screen dark page, no sidebar, no auth required
- Header: ECHOFACE · Live Tracking Feed · [✕ close window]
- Body: Radio icon placeholder + "No active feed" message
- Footer: ● No active session | FPS: -- | Faces: --
- Route /live-feed added to App.tsx OUTSIDE <Layout> (no AccessGate)

### Task 5 — Overview camera health (Overview.tsx)
- Replaced useCameras() + activeCams with useCameraHealthSummary() (30s poll)
- Camera Sources StatusIndicator replaced with inline JSX showing:
  ● {total} registered  ● {online} online  ◌ {offline} offline
  + [Manage →] button navigating to /administration

### Task 6 — AlertDetail camera enrichment (AlertDetail.tsx)
- Added backendUrl to useAppStore destructure
- Added createApiClient import
- cameraDetail state (useState) + useEffect after sighting useMemo
  → GET /cameras/{camera_id} on each alert load (ADMIN/DEMO only)
- CAMERA / SOURCE cell shows:
  enriched: "{name} · {zone_name}" / "{site_name} · {direction} · Trust: {level} | {TYPE}"
  fallback: raw source_name / camera_id (unchanged)

### New hook (hooks.ts)
useCameraHealthSummary():
  - GET /cameras/health-summary, polls every 30s
  - MOCK returns { total:3, online:2, offline:1, reconnecting:0, unknown:0 }
  - Exports CameraHealthSummary interface

### App.tsx restructure
- /live-feed route is first, outside auth + layout
- Remaining routes wrapped in conditional: !accessMode → AccessGate, else Layout

### index.css
- Added @layer components { .field-input { ... } } Tailwind component utility
  Used in RegisterCameraModal form fields

## TypeScript
- tsc --noEmit: 0 errors
- Runtime: 0 console errors verified in browser

## Files changed
frontend/src/pages/Administration.tsx  (new)
frontend/src/pages/LiveFeed.tsx         (new)
frontend/src/api/hooks.ts               (append: useCameraHealthSummary)
frontend/src/components/Layout.tsx      (Administration tab unlocked)
frontend/src/App.tsx                    (live-feed route + administration route)
frontend/src/pages/Operations.tsx       (source selector + Live Feed button)
frontend/src/pages/Overview.tsx         (camera health panel)
frontend/src/pages/AlertDetail.tsx      (camera enrichment)
frontend/src/index.css                  (field-input utility)

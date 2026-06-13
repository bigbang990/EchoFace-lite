# Checkpoint — 2026-06-13 — Phase A session 3: evidence display + persistence fixes

## Phase
Frontend v1.4 — sighting status persistence, face-crop evidence, alert counts fixed

## Changes this session

### Backend
- `models.py` — `Sighting.status` column added (VARCHAR 32, default `"pending"`)
- `session.py` — schema patch adds `status` column to existing sightings tables on startup
- `schemas.py` — `SightingOut.status` field; `SightingStatusUpdate` schema
- `incidents.py` — `_count_persons()` + `_count_sightings()` direct SQL count helpers replace broken `selectinload` on many-to-many; `_build_sightings_out()` helper; `PATCH /{id}/sightings/{sid}` endpoint for confirm/reject
- `main.py` — `/data/snapshots` static mount added (face crops were never served)

### Frontend
- `types/index.ts` — `Sighting.snapshot_path?: string`
- `hooks.ts` — `normalizeSighting` includes `snapshot_path`
- `Timeline.tsx` — `AlertCard` shows face-crop `<img>` when `snapshot_path` available; accepts `backendBase` prop
- `CaseWorkspace.tsx` — confirm/reject calls PATCH API then refetches; comments persist to localStorage `echoface_notes_{id}`; collapsible ENROLLED PHOTOS section; case info shows alert count
- `SystemHealth.tsx` — CPU mode (type=0) shows "CPU BACKEND: ACTIVE" (green)
- `Operations.tsx` — preview is now primary (above job details); image preview auto-refreshes every 2s while running; metrics reset on file selection; `LivePreviewImage` component for jpg/png detection

## Regression baseline
Not run this session (frontend/schema changes only, no pipeline logic touched).
Run before merge: `python -m pytest tests/identity_stress_suite.py -v`

## Key known-good metrics (from last run)
identity_switch_rate: 0.000
stable_matches: (varies by video)
confirmation_rate: (varies)
validator_rejection_rate: (varies)

# Checkpoint — 2026-06-14 — Phase A session 6: forensic alert detail page

## Phase
Frontend v1.7 — forensic alert detail page + enriched alert cards

## Changes this session (session 6)

### Forensic Alert Detail page (NEW)
- `frontend/src/pages/AlertDetail.tsx` — NEW full-page forensic view at `/cases/:id/alerts/:sightingId`
  - Left panel: detected face crop (large), all snapshots filmstrip, enrolled reference photos — all open in ImageZoomModal
  - Center panel: person identity block, animated confidence bar with tier label, 6-tile detail grid (camera/source, location, detected-at, frame index, alert ID, case ref), operator notes (session-local)
  - Right panel: detection history for this person (total/confirmed/rejected counts + chronological list with status badges; click to navigate between sightings), video clip placeholder
  - Header: breadcrumb (INC-XXX / Alert #N), PENDING badge, CONFIRM MATCH / REJECT buttons, Refresh
- `frontend/src/App.tsx` — added route `/cases/:id/alerts/:sightingId`
- `frontend/src/components/Timeline.tsx` — AlertCard enriched: camera icon + source name, clock icon + timestamp; new "VIEW FULL REPORT →" link; added `incidentId` prop threaded through Timeline → AlertCard
- `frontend/src/pages/CaseWorkspace.tsx` — passes `incidentId={String(incident.id)}` to `<Timeline>`
- `.claude/launch.json` — fixed `runtimeExecutable` to absolute node path for preview tool

## Changes in sessions 4 + 5 (prior)

### Issue 1 — Pending alert count desync (FIXED)
- `frontend/src/pages/Overview.tsx:24` — now sums `pending_alert_count` (was `alert_count`)

### Issue 2 — ImageZoomModal navigation + zoom/pan (FIXED)
- `frontend/src/components/ImageZoomModal.tsx` — full rewrite; scroll zoom 1x–4x; drag-pan; prev/next; keyboard ArrowLeft/ArrowRight/Escape; per-image fade; backdrop-click-to-close with drag detection
- `frontend/src/components/Timeline.tsx` — modal lifted to Timeline level; all snapshots as navigation array
- `frontend/src/pages/CaseWorkspace.tsx` — updated to new `images[]` API

### Issue 3 — Enrollment photos not displaying (FIXED)
- `frontend/src/pages/CaseWorkspace.tsx` — `EnrolledPhoto` component with "NO IMAGE" error state; backslash path normalization; PersonAvatar background container
- `frontend/src/types/index.ts` — added `extra_photo_paths?: string[]` to `Person`
- `frontend/src/api/hooks.ts` — `normalizePerson` now maps `extra_photo_paths`; gallery collects all photos across all persons

### Issue 4 — Alert snapshots full frame (FIXED)
- `ecoface_lite/services/video_service.py` — snapshot save changed from `packet.bgr` (full frame) to cropped face from `inference_frame` using `m.face.bbox` with 15% padding
- `ecoface_lite/db/models.py` — added `extra_photo_paths: Mapped[str | None]` column on Person
- `ecoface_lite/db/session.py` — schema patch: `ALTER TABLE persons ADD COLUMN extra_photo_paths TEXT`
- `ecoface_lite/api/schemas.py` — `PersonOut.extra_photo_paths` field + `field_validator` to decode JSON string

### Issue 5 — Live preview updates every ~25s (FIXED)
- `ecoface_lite/ai_engine/visualization.py:37` — `should_write()` now uses `video_preview_interval` (was `tracking_overlay_interval` — wrong setting)
- `ecoface_lite/core/config.py` — `video_preview_interval` default changed from 5 → 1; preview writes every processed frame

### Multi-photo enrollment persistence (FIXED)
- `ecoface_lite/services/person_service.py` — `add_photos_to_person()` now saves image files to uploads dir and appends paths to `person.extra_photo_paths` JSON column

## Regression suite result — 2026-06-14
All 14 tests pass: `python -m pytest tests/test_temporal_identity.py tests/test_temporal_modules.py tests/test_tracking.py -v`

## Regression baseline metrics (14/14 pass)
identity_switch_rate: 0.000
stable_matches: green
confirmation_rate: green
validator_rejection_rate: green
bbox_jitter: green

## Files touched (pipeline-adjacent — regression required before merge)
- `ecoface_lite/services/video_service.py` (face crop change)
- `ecoface_lite/db/models.py` (extra_photo_paths column)
- `ecoface_lite/ai_engine/visualization.py` (preview interval setting)
- `ecoface_lite/core/config.py` (video_preview_interval default)

## Branch
`phase8-pipeline-decompose` — do NOT merge to main before running stress suite

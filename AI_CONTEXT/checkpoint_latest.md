# Checkpoint — 2026-06-14 — Phase 8.7 complete: Enrollment Conflict Detection

## Phase
Phase 8.7 — Enrollment Conflict Detection (complete)

## Changes this session (Phase 8.6 — Case Lifecycle Integrity)

### Fix 1 — Pipeline gallery gate (video_service.py)
- `inc_rows` query at the per-frame match loop now JOINs `Incident` and filters
  `.where(Incident.status == "open").where(Incident.is_paused == False)`
- Previously the gallery load was gated but the per-incident alert dispatch was not —
  a person linked to both an open and a closed incident would fire `record_match()`
  for the closed incident. This is the root cause of "Alerts Firing on Closed Incidents".

### Fix 2 — Confidence floor verification (no code change)
- `FrameMatch.confidence` is the ArcFace recognizer score; detector confidence is
  separate in `FaceDebugTrace.detector_confidence`. The floor is comparing the right value.
- Alert #9 at 48% was created before Phase 8.5 was deployed (pre-fix artifact).

### Fix 3 — Case closure modal + DB (backend + frontend)
**DB (`ecoface_lite/db/models.py`):**
- 5 new columns on `Incident`: `closing_reason VARCHAR(64)`, `closing_summary TEXT`,
  `closed_by VARCHAR(128)`, `closed_at DATETIME`, `evidence_paths TEXT` (JSON list)

**Migrations (`ecoface_lite/db/session.py`):**
- 5 `ALTER TABLE incidents ADD COLUMN` patches

**Schemas (`ecoface_lite/api/schemas.py`):**
- `IncidentCloseRequest`: `reason` (enum pattern), `summary` (required), `closed_by` (optional)
- `IncidentOut`: new closure fields + `evidence_paths: list[str]` with JSON `@field_validator`
- `AlertOut`: `incident_status: str | None` enriched field

**Alert engine (`ecoface_lite/services/alert_session_engine.py`):**
- `evict_incident(incident_id)` — removes all sessions for a closing incident from
  the in-memory registry without requiring a DB call

**Incidents router (`ecoface_lite/api/routers/incidents.py`):**
- `POST /incidents/{id}/close` — requires reason + summary; atomically:
  sets status="closed", writes closure metadata, bulk-closes open alerts via SQL UPDATE,
  evicts sessions from the alert engine registry
- `POST /incidents/{id}/evidence` — multipart file upload; saves to
  `data/evidence/{incident_id}/`, appends paths to `evidence_paths` JSON column
- `_incident_out()` passes all 5 closure fields through

**Frontend (`frontend/src/components/CaseCloseModal.tsx`) — NEW:**
- Modal with: reason dropdown (required), summary textarea (required, 4000 chars),
  closed_by text (optional), evidence file picker (optional, multi-file)
- On submit: uploads evidence files first (if any), then POST /incidents/{id}/close
- MOCK mode: simulates 600ms delay then calls onClosed()

**Frontend (`frontend/src/pages/CaseWorkspace.tsx`):**
- "Resolve & Close" button now opens `CaseCloseModal` instead of calling `patchStatus` directly
- `handleCaseClosed()`: closes modal, sets local status, appends timeline entry, refetches

### Fix 4 — Read-only enforcement on closed cases (backend + frontend)
**Backend (`ecoface_lite/api/routers/alerts.py`):**
- `append_alert_note`: loads `Alert.incident` via `selectinload`; returns 409 if
  `alert.incident.status == "closed"` — "Case is closed — notes cannot be appended"
- `update_alert_status`: same guard — "Case is closed — alert status cannot be changed"
- `get_alert`: loads `Alert.incident` and passes `incident_status` to `_alert_out`
- `_alert_out`: accepts and passes through `incident_status` parameter

**Frontend (`frontend/src/api/hooks.ts`):**
- `AlertApiData`: added `incident_status: string | null`

**Frontend (`frontend/src/pages/AlertDetail.tsx`):**
- `caseClosed = incident?.status === 'CLOSED' || alertData?.incident_status === 'closed'`
- CONFIRM MATCH / REJECT buttons hidden when `caseClosed`
- "CASE CLOSED — READ ONLY" chip shown in header instead
- Note input and submit button replaced with "Case closed — notes locked" text

## Changes this session (Phase 8.7 — Enrollment Conflict Detection)

### Fix 1 — Pre-enrollment identity check (backend)
- `ecoface_lite/core/config.py` — `enrollment_conflict_threshold: float = 0.65` (env: `ENROLLMENT_CONFLICT_THRESHOLD`)
- `ecoface_lite/services/person_service.py`:
  - `EnrollmentConflictError` dataclass exception: fields `person_id`, `person_name`, `incident_id`,
    `incident_ref`, `incident_title`, `incident_status`, `incident_opened_at`, `similarity`
  - `_check_identity_conflict(session, new_embedding, threshold)`: JOINs FaceEmbedding → Person →
    incident_persons → Incident; filters open non-paused incidents; computes ArcFace cosine sim
    (dot product of L2-normalized vectors); raises `EnrollmentConflictError` if `best_sim >= threshold`
  - `create_person_from_image()`: added `skip_conflict_check: bool = False`; conflict check fires
    after embedding generation unless `skip_conflict_check=True`
- `ecoface_lite/api/routers/persons.py`:
  - `create_person` endpoint: added `force_create: bool = Form(default=False)` parameter
  - Passes `skip_conflict_check=force_create` to `create_person_from_image()`
  - Catches `EnrollmentConflictError` → HTTP 409 with structured dict detail:
    `conflict`, `person_id`, `person_name`, `incident_id`, `incident_ref`, `incident_title`,
    `incident_status`, `incident_opened_at` (ISO string), `similarity` (4 dp)

### Fix 2 — Gallery eviction verification (no code change)
- Finding: no persistent in-memory face embedding gallery exists.
  `load_gallery()` and `load_named_gallery()` in `video_service.py` both query the DB fresh on
  every video job / live-test call. The OPEN-incident filter was already present in the gallery
  loader; there is nothing to evict.
- The alert session engine IS in-memory and IS correctly evicted by `evict_incident()` (Phase 8.6)
  before the HTTP response returns. Ordering: DB commit → engine evict → HTTP 200.

### Fix 3 — Active Case Conflict UX in enrollment flow (frontend)
- `frontend/src/pages/CreateCase.tsx`:
  - `EnrollmentConflict` interface (mirrors 409 detail dict)
  - States: `enrollConflict: EnrollmentConflict | null`, `conflictConfirmText: string`
  - `runProcessing(force = false)`: clears conflict on fresh run; appends `force_create=true` form
    field when `force=true`; on HTTP 409 parses `errData.detail` and sets `enrollConflict`
  - "Active Case Conflict" card (replaces generic photo warning): shows matched person name +
    incident ref + similarity % + incident title + status + opened date
  - Three CTA options:
    - "View Case" → `navigate('/cases/{incident_id}')`
    - "Add Photos to Existing" → `navigate('/cases/{incident_id}')`
    - "Create Anyway" section with typed confirmation (`conflictConfirmText === "CREATE DUPLICATE"`)
      that enables a button calling `runProcessing(true)`

## Changes this session (Phase 8.5 cleanup sprint)

### Fix 2 — Reference photo validation gate
- `ecoface_lite/ai_engine/pipeline.py` — `count_enrollment_faces()` added; does not alter state
- `ecoface_lite/services/person_service.py` — `_validate_enrollment_image()` helper;
  called before `enroll_reference_embedding` in both `create_person_from_image()` and
  `add_photos_to_person()`; 0 faces → 400 "No face detected"; >1 face → 400 "Multiple faces detected"

### Fix 1 — Confidence floor
- `ecoface_lite/core/config.py` — `alert_min_confidence_floor: float = 0.65` (ALERT_MIN_CONFIDENCE_FLOOR)
- `ecoface_lite/services/alert_session_engine.py` — below-floor: write Sighting with alert_id=None, return (None, sighting)

### Fix 3 — Operator notes persistence
- `ecoface_lite/db/models.py` — `operator_notes TEXT` column on Alert
- `ecoface_lite/db/session.py` — schema patch: `ALTER TABLE alerts ADD COLUMN operator_notes TEXT`
- `ecoface_lite/api/schemas.py` — `AlertNoteCreate`, `AlertOut.operator_notes` added
- `ecoface_lite/api/routers/alerts.py` — `POST /alerts/{id}/notes` append-only with UTC timestamp
  format: `[YYYY-MM-DD HH:MM:SS UTC] note text\n`

### Fix 4 — Detection history wired to real sighting count (frontend only)
- `frontend/src/api/hooks.ts` — `AlertApiData` interface + `useAlert()` hook
- `frontend/src/pages/AlertDetail.tsx` — TOTAL/VALID/LOW counts from `alertData`;
  VALID = sightings with confidence ≥ 0.65, LOW = confidence < 0.65;
  `addNote()` async → POST /api/v1/alerts/{id}/notes, refetches after success;
  notes seeded from `alertData.operator_notes` on load

### TypeScript build fix
- `frontend/src/api/client.ts:12` — typed `extraHeaders` as `Record<string, string>` to
  eliminate the `{ 'ngrok-skip-browser-warning'?: undefined }` union that broke `HeadersInit`

## Changes this session (Phase 8 implementation)

### New file: `ecoface_lite/services/alert_session_engine.py`
- `AlertSessionEngine` class with `asyncio.Lock`-guarded in-memory registry
- Registry key: `(incident_id, person_id, camera_id)` — one session per presence per camera
- `record_match()` — opens new Alert on first match or after gap/zone-change, appends Sighting every call
- `rebuild_from_db()` — restores open sessions from DB on restart (lookback window configurable)
- `close_all()` — graceful shutdown hook
- `get_alert_session_engine()` — process-lifetime singleton

### New file: `ecoface_lite/api/routers/alerts.py`
- `GET /api/v1/incidents/{id}/alerts` — list alerts with optional status/level/source filters
- `GET /api/v1/alerts/{id}` — single alert with full sighting list
- `PATCH /api/v1/alerts/{id}/status` — operator status update
- `PATCH /api/v1/alerts/{id}/level` — operator level promotion (Phase 11 ladder)

### DB models — `ecoface_lite/db/models.py`
- `Alert` model added: incident_id, person_id, camera_id, zone_id (Phase 10),
  status, level (Phase 11), source (VSL Phase 4), first_seen_at, last_seen_at,
  sighting_count, timestamps
- `Sighting` updated: added alert_id, person_id, confidence, frame_index,
  snapshot_path, source columns
- `Incident.alerts` relationship added

### DB migration — `ecoface_lite/db/session.py`
- `CREATE TABLE IF NOT EXISTS alerts` with all Phase 8 columns
- `ALTER TABLE sightings ADD COLUMN` for all 6 new Sighting fields
- Indexes on alerts.incident_id, alerts.person_id, sightings.alert_id

### Config — `ecoface_lite/core/config.py`
- `alert_session_gap_seconds: int = 60` (ALERT_SESSION_GAP_SECONDS)
- `alert_session_rebuild_minutes: int = 10` (ALERT_SESSION_REBUILD_MINUTES)

### `ecoface_lite/services/video_service.py`
- `get_alert_session_engine` imported and wired into `process_prerecorded_video`
- Per-frame loop now calls `engine.record_match()` per incident per match
- DetectionEvent still created as audit trail; Sighting created by engine with alert_id link
- Frame-dedupe logic renamed to `last_sighting_frame_by_person` — controls write frequency,
  not alert creation (alert session handles presence continuity)
- `alerts` counter now tracks `new_sessions` (sighting_count == 1)

### `ecoface_lite/api/main.py`
- `alerts` router imported and included at `/api/v1`
- `rebuild_from_db()` called in lifespan after `init_db()` — registry warm on startup

### `ecoface_lite/api/schemas.py`
- `AlertOut` added with all Phase 8 fields + future-scoped zone_id, level, source
- `AlertStatusUpdate`, `AlertLevelUpdate` added

## Changes this session (session 7)

### hardware_backend_type inversion bug (FIXED)
- `frontend/src/api/hooks.ts` — `useSystemMetrics` normalizer now reads from
  `raw.averages.hardware_backend_type` (correct nested path from `metrics.export()`);
  uses `effective_runtime_config.backend_type` string enum as primary GPU source
  (`"COLAB_GPU"` / `"REMOTE_GPU"` → isGpu=true → hwType=1 → shows GPU label);
  all other metrics (fps, latency, identity_switch_rate etc.) also fixed to read
  from `raw.averages.*` / `raw.rates.*` with root-level fallback
- `frontend/src/api/client.ts` — `.trim()` on baseUrl + `ngrok-skip-browser-warning`
  header added to bypass ngrok free-tier HTML interstitial
- `frontend/src/store/appStore.ts` — `.trim()` on `setBackend`/`setIncUrl`;
  updated Colab GPU ngrok URL to `a84a-136-118-99-101.ngrok-free.app`

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

# Checkpoint — 2026-06-14 — VSL Phase 1 complete: Source Abstraction Foundation

## Phase
VSL Phase 1 — Source Abstraction Foundation (complete)
Branch: `vsl-phase1-source-abstraction`

## Regression baseline metrics (14/14 pass)
identity_switch_rate: 0.000
stable_matches: green
confirmation_rate: green
validator_rejection_rate: green
bbox_jitter: green

## Changes this session (VSL Phase 1)

### New file: `ecoface_lite/input_sources/base.py`
- `BaseVideoSource` ABC — the contract every video source must satisfy
- `Frame` dataclass: index, bgr, captured_at (datetime), source_id (str)
- `CameraMetadata` dataclass: source_id, name, source_type, stream_url, zone, location, fps, width, height
- `HealthStatus` dataclass: source_id, status (SourceStatus), last_seen, error
- `SourceType` enum: FILE / RTSP / ANDROID (Phase 3) / NVR (Phase 5)
- `SourceStatus` enum: ONLINE / OFFLINE / RECONNECTING / UNKNOWN
- Interface: connect() → bool, disconnect() → None, get_frame() → Frame | None,
  get_metadata() → CameraMetadata, health_check() → HealthStatus,
  get_historical_stream(start, end) → NotImplementedError until VSL Phase 4

### Modified: `ecoface_lite/input_sources/video_file.py`
- `VideoFileSource` now implements `BaseVideoSource` (multiple inheritance; VideoSource kept)
- Added: connect(), disconnect(), get_frame(), get_metadata(), health_check()
- `frames()` iterator preserved unchanged — existing pipeline callers unaffected
- Optional params: source_id, name, zone, location

### New file: `ecoface_lite/input_sources/rtsp_source.py`
- `RTSPSource` implementing `BaseVideoSource`
- Drop-stale-frames policy: always serves latest frame, never queues
- Auto-reconnect: `reconnect_with_backoff(max_attempts=5)` — exponential 2s → 30s cap
- Covers Hikvision, Dahua, Android IP Webcam RTSP URL patterns

### New file: `ecoface_lite/input_sources/source_registry.py`
- `SourceRegistry` — bridges Camera DB rows to concrete BaseVideoSource instances
- `build_source(camera)` dispatches on `camera.source_type` (file → VideoFileSource, rtsp/android → RTSPSource)
- `register()` persists new Camera row and returns it
- `list_cameras()` / `get_camera()` async DB helpers
- `get_source_registry()` — process-lifetime singleton

### Modified: `ecoface_lite/db/models.py`
- Camera: +source_type VARCHAR(32) DEFAULT 'file', +zone VARCHAR(255),
  +status VARCHAR(32) DEFAULT 'unknown', +last_seen DATETIME

### Modified: `ecoface_lite/db/session.py`
- 4 new `ALTER TABLE cameras ADD COLUMN` migration patches

### Modified: `ecoface_lite/api/schemas.py`
- `CameraOut`: added source_type, zone, status, last_seen
- `CameraCreate`: added source_type (file|rtsp|android, pattern-validated), zone
- `CameraHealthUpdate` (NEW): status + last_seen — used by health monitor patch endpoint

### Modified: `ecoface_lite/api/routers/cameras.py`
- create_camera: passes source_type + zone
- `PATCH /{id}/health`: updates status/last_seen — called by VSL Phase 2 health monitor
- `POST /{id}/test-connect`: builds source, attempts live connect, persists result, returns JSON

## Architecture state

```
input_sources/
  base.py          ← BaseVideoSource ABC (VSL contract)
  video_file.py    ← VideoFileSource (file + legacy frames())
  rtsp_source.py   ← RTSPSource (RTSP/Android)
  source_registry.py ← SourceRegistry singleton
  __init__.py      ← exports all VSL Phase 1 symbols
```

## VSL Phase 2 prerequisites (next)
- Location hierarchy: Country → State → District → Site → Zone → Camera
- Health monitor async background task (polls each source, calls PATCH /{id}/health)
- Dashboard panel: Connected / Online / Offline / Warning counts
- Zone-aware alert routing (zone field now on Camera, needs FK in Phase 2)

## Prior phases preserved
See prior checkpoint history — Phase 8 + 8.5–8.7 changes remain in full effect
on `phase8-lifecycle-enrollment`. VSL Phase 1 branched from there.

## Branch
`vsl-phase1-source-abstraction` — regression gate passed, safe to merge to main
after stress suite (per CLAUDE.md rules, do NOT skip)

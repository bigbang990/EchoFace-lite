# Checkpoint — 2026-06-14 — VSL Phases 3 + 4 complete

## Phase
VSL Phase 4 — Historical Footage Access (complete)
Branch: `vsl-phase2-location-health`
All prior phases (1, 2, 3) complete and verified on same branch.

## Regression baseline metrics (30/30 pass)
identity_switch_rate: 0.000
stable_matches: green
confirmation_rate: green
validator_rejection_rate: green
bbox_jitter: green
Test suite: 30 tests, 0 failed

## Bug fixed this session
`main.py` shutdown handler caught `Exception` after `await _health_task` — but
`asyncio.CancelledError` is a `BaseException` in Python 3.8+, not `Exception`.
Fix: changed `except Exception` → `except BaseException` in lifespan shutdown.
Impact: `test_health` was failing with `CancelledError` on every TestClient teardown.
Now 30/30 green.

---

## VSL Phase 3 changes (AndroidCameraSource + MultiSourceScheduler)

### New file — `ecoface_lite/input_sources/android_source.py`
- `AndroidCameraSource(RTSPSource)`: thin subclass, `SourceType.ANDROID`,
  initial backoff 1s (Android RTSP restarts faster than IP cameras)
- Constructor matches RTSPSource interface, passes `source_type=SourceType.ANDROID`

### Modified — `ecoface_lite/input_sources/source_registry.py`
- `build_source()` dispatches on `source_type` string:
  `"android"` → `AndroidCameraSource`, `"rtsp"` → `RTSPSource`, else → `VideoFileSource`
- Singleton: `get_source_registry()` returns module-level `SourceRegistry` instance

### New file — `ecoface_lite/services/multi_source_scheduler.py`
- `MultiSourceScheduler(sources: list[BaseVideoSource])`
- `start() → dict[int, bool]`: calls `connect()` once per source (ByteTrack safe)
- `stop() → None`: calls `disconnect()` once per source (ByteTrack safe)
- `get_next_frame() → Frame | None`: round-robin, skips sources on exception,
  returns None only when all sources exhausted this round
- `_FpsTracker`: rolling deque window (configurable size) for per-source FPS
- Source isolation: one source returning None or raising never stops the loop

### Config — `ecoface_lite/core/config.py`
- `use_vsl_frame_path: bool = False` (USE_VSL_FRAME_PATH feature flag)
  → NEVER True until 3× consecutive green GPU regression runs on Colab T4
- `scheduler_fps_window: int = 30` (rolling FPS window size)

### Exports — `ecoface_lite/input_sources/__init__.py`
- Added `AndroidCameraSource` to `__all__`

---

## VSL Phase 4 changes (Historical Footage Access)

### Modified — `ecoface_lite/input_sources/video_file.py`
- `VideoFileSource.__init__`: added `video_epoch: datetime | None = None` param
  → anchor for wall-clock → frame-index mapping in historical stream
- `get_historical_stream(start_time, end_time) → Generator[Frame, None, None]`:
  - Epoch resolution: `video_epoch` if set, else file `mtime`
  - Seeks to `start_frame` via `CAP_PROP_POS_FRAMES`
  - Yields frames with wall-clock `captured_at` timestamps
  - Opens/releases its own `cv2.VideoCapture` (leaves `self._cap` for `get_frame()` intact)
  - Guards: logs and returns early if window is outside video range
- `supports_historical` property override: `True` (file sources support seek)

### New file — `ecoface_lite/services/historical_search.py`
- `_run_historical_search(*, job_id, incident_id, video_path, start_time, end_time,
  source_id, video_epoch, frame_skip, settings)`:
  - Background asyncio coroutine (launched via `asyncio.create_task()`)
  - Loads gallery from DB, aborts with `mark_failed` if empty
  - Iterates `VideoFileSource.get_historical_stream()`
  - `pipeline.process_frame()` per frame
  - Creates `DetectionEvent` + `Sighting(source="historical", alert_id=None)` per match
  - `await asyncio.sleep(0)` each frame — yields to event loop, live pipeline unaffected
  - Progress update every 50 frames via `processing_status_service.set_progress()`
  - `mark_completed` / `mark_failed` at end
- `submit_historical_search(*, incident_id, video_path, start_time, end_time,
  video_epoch=None, frame_skip=1, source_id="historical", job_id=None) → str`:
  - Accepts optional `job_id` (caller pre-created the status row) or auto-generates one
  - Returns `job_id` for polling

### Modified — `ecoface_lite/api/schemas.py`
- `HistoricalSearchRequest`: video_path (str), start_time (datetime), end_time (datetime),
  video_epoch (datetime|None, default None), frame_skip (int, ge=1, default 1)

### New file — `ecoface_lite/api/routers/historical.py`
- `POST /incidents/{incident_id}/historical-search` (202, AsyncVideoJobResponse):
  - Validates incident exists + not closed
  - `safe_video_path()` validates path under VIDEOS_DIR
  - Pre-creates `ProcessingStatus` row with shared `job_id`
  - Calls `submit_historical_search(..., job_id=job_id)` — single job_id, no mismatch
  - Returns `{job_id, status: "queued", status_url: /api/v1/videos/processing-status/{job_id}}`
- `GET /incidents/{incident_id}/historical-sightings`:
  - Queries `Sighting WHERE incident_id=X AND source="historical"`
  - Ordered by `frame_index`
  - Returns id, person_id, confidence, frame_index, snapshot_path, status, source, created_at

### Modified — `ecoface_lite/api/main.py`
- Imports `historical` router
- Registers at `/api/v1` (between incidents and alerts)
- Shutdown fix: `except BaseException` instead of `except Exception` for health task cancel

---

## Architecture state (VSL Phases 1–4 complete)

```
Input sources:
  base.py              → BaseVideoSource ABC (connect/disconnect/get_frame/get_metadata/health_check)
  video_file.py        → VideoFileSource (frames() legacy + get_frame() + get_historical_stream())
  rtsp_source.py       → RTSPSource (exponential backoff: 2s→4s→8s→16s→30s cap)
  android_source.py    → AndroidCameraSource (RTSPSource subclass, 1s initial backoff)
  source_registry.py   → SourceRegistry.build_source() dispatches on source_type

Multi-source scheduling:
  services/multi_source_scheduler.py → MultiSourceScheduler (round-robin, source isolation)
  connect() once in start(), disconnect() once in stop() — ByteTrack state preserved
  USE_VSL_FRAME_PATH=False → single-source pipeline still uses frames() iterator

Health monitoring (Phase 2):
  services/health_monitor.py → standalone asyncio.Task "health_monitor"
  polls every 60s; writes cameras.status/last_seen; NOT in frame loop

Location hierarchy (Phase 2):
  sites → zones → cameras (FK chain, SET NULL on zone delete)

Historical search (Phase 4):
  services/historical_search.py → asyncio background task per job
  VideoFileSource.get_historical_stream() → seeks by wall-clock window
  Sighting(source="historical", alert_id=None) → never in live alert feed
  ProcessingStatus row polled via GET /api/v1/videos/processing-status/{job_id}

API surface (complete through Phase 4):
  GET  /api/v1/sites
  POST /api/v1/sites
  GET  /api/v1/sites/{id}
  GET  /api/v1/sites/{id}/zones
  DELETE /api/v1/sites/{id}
  GET  /api/v1/zones
  POST /api/v1/zones
  GET  /api/v1/zones/{id}
  DELETE /api/v1/zones/{id}
  GET  /api/v1/cameras/health-summary
  POST /api/v1/incidents/{id}/historical-search   [Phase 4]
  GET  /api/v1/incidents/{id}/historical-sightings [Phase 4]
```

## VSL Phase 5 (deferred — post-dissertation)
NVR/DVR integration: RTSPSource.supports_historical override, NVR-specific auth,
DVR segment fetching. No code written. Deferred per roadmap.

## Hard stops still in force
- `frames()` iterator must remain callable forever (never delete)
- `USE_VSL_FRAME_PATH=False` default — flip only after 3× green GPU regression runs
- Health monitor never called from frame loop
- `get_frame()` single-source pipeline switch deferred to Phase 3 scheduler activation

## Prior phases preserved
VSL Phase 1 (base abstraction) intact.
VSL Phase 2 (location hierarchy + health monitor) intact.
VSL Phase 3 (Android source + multi-source scheduler) intact.
Phase 8 + 8.5–8.7 (alert session engine) intact.

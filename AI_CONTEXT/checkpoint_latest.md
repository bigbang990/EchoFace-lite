## Checkpoint — 2026-06-11 — Phase 7B session lifecycle isolation complete

### Phase
7B session lifecycle isolation

### Status
complete

### Bug fixed
MetricsRegistry counter leak across runs — metrics.reset() now called at job start.

### New behaviour
Each video job gets a UUID via new_session_id() before metrics.reset().
clear_session_id() fires in a finally block — runs on success AND error.
GET /observability/metrics now includes "session_id" field.

### Files changed (Phase 7B part 2)
- ecoface_lite/services/video_service.py — session start/end hooks in run_async_video_job
- ecoface_lite/api/routers/observability.py — added session_id to metrics response

### Lifecycle order (in run_async_video_job)
1. new_session_id() → store UUID
2. metrics.reset() → clear all counters
3. logger.info("Session started ...")
4. try: process_prerecorded_video(...)
5. except HTTPException / except Exception → mark_failed
6. finally: clear_session_id() + logger.info("Session ended ...")

### Previous phase (7B multi-photo enrollment — also complete)
- POST /persons/{person_id}/photos — max 5 photos, sha256 dedupe, returns accepted/rejected
- ecoface_lite/services/person_service.py, ecoface_lite/api/routers/persons.py

### GPU baseline metrics (still valid)
- hardware_backend_type: 1 (GPU)
- total_faces_detected: 687
- detector_rejection_rate: 0.347
- recall_per_resolution: 0.65
- face_visibility_ratio: 0.95
- alerts_per_video: 3
- stable_matches: 50
- identity_switch_rate: 0
- average_processing_fps: 81
- detector_runtime_ms: 13.2ms

### State
- Working: full GPU pipeline + multi-photo enrollment + session isolation
- Next: run regression suite, then merge to main as v0.7.0

### Branch
phase6-detector-abstraction

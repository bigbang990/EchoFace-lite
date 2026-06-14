# Checkpoint — 2026-06-15 — video_service.py stream URL support (VSL Phase 3)

## Phase
VSL Phase 3 — multi-source stream URL routing
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

## Changes this session

### video_service.py — stream URL support

**Problem:** `video_relative_path` was always treated as a filesystem path.
Stream URLs (`http://`, `https://`, `rtsp://`) failed with "Video file not found" because
`_safe_video_path()` would resolve the URL against `VIDEOS_DIR`, and `is_file()` returned False.

**Fix (video_service.py only):**

**`_is_stream_url(path: str) -> bool`** — new module-level helper added after `_safe_video_path`.
Returns True if path starts with `http://`, `https://`, or `rtsp://`.

**`process_prerecorded_video()`** — branched before `_safe_video_path()`:
- Stream URL path: opens `cv2.VideoCapture(url)` directly, checks `isOpened()`,
  raises HTTP 422 if connection fails. Frame iteration uses an inline generator
  (`_make_frame_iter`) that yields `FramePacket` objects compatible with the
  existing loop body. `_cap.release()` is in the generator's `finally` block.
- File path: unchanged — `_safe_video_path()` + `is_file()` check as before.
- `video_path` is set to the URL string for stream paths (used only for logging/return).
- `VideoFileSource` is NOT used for stream URLs — `path.resolve()` in VideoFileSource
  mangles URL strings on Windows (pathlib collapses `//` to `/`).

**`run_async_video_job()`** — branched before `safe_video_path()`:
- Stream URL path: skips `safe_video_path()`, `is_file()`, and `count_emitted_frames()`
  (frame count unknown for live streams). Sets `total = 0`, proceeds directly to
  `set_total_frames_and_running` with `max(0, 1) = 1`.
- File path: unchanged — existing validation + `count_emitted_frames()` as before.
- `process_prerecorded_video()` is called with `video_relative_path` in both paths;
  it handles the stream/file distinction internally.

**No changes to:** `processing.py`, `schemas.py`, any router, detection/recognition/alert
pipeline, `_safe_video_path()`, `VideoFileSource`.

**Import added:** `FramePacket` imported alongside `VideoFileSource` in
`process_prerecorded_video()` local imports block.

---

## Previous session changes (still valid)

### cameras.py — test-connect endpoint
android/rtsp source types: URL format validation only, no cv2.VideoCapture
(Colab cannot reach local-network IPs).

### AndroidCameraSource (ecoface_lite/input_sources/android_source.py)
Standalone BaseVideoSource for MJPEG HTTP (IP Webcam app).
See prior checkpoint for design details.

### Frontend
- hooks.ts: `useCameras()` maps `c.label ?? c.name` to `name` field
- Administration.tsx: Promise.allSettled for independent cameras + health fetches;
  local CameraRow interface uses `label`; debug console.log on load
- Operations.tsx: local URL validator for RTSP Test button (no API call);
  camera/rtsp startTracking branches POST to `/videos/process/async`

## TypeScript
- tsc --noEmit: 0 errors (last verified previous session)

## Files changed this session
ecoface_lite/services/video_service.py  (stream URL branching — 4-step fix)

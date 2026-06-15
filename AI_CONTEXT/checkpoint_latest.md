# Checkpoint — 2026-06-16 — preview no-cache fix + live feed implementation

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

### Bug 1 — Operations page preview never updates during processing

**Root cause:** `app.mount("/data/previews", StaticFiles(...))` sends ETag + Last-Modified
headers. The file on disk was being overwritten by VideoPreviewWriter every frame, but
StaticFiles served a cached ETag — all 198 polling requests got 304 Not Modified.

**Fix 1a — processing.py:** New endpoint `GET /api/v1/videos/preview-image/{job_id}`.
Uses `FileResponse` with `Cache-Control: no-store, no-cache, must-revalidate` headers.
Reads the file fresh on every request, bypassing StaticFiles entirely for previews.

**Fix 1b — hooks.ts:** `useVideoJob` now sets `previewUrl` to
`${backendBase}/api/v1/videos/preview-image/${jobId}` (no-cache endpoint)
instead of the former `${backendBase}/data/previews/${jobId}/latest.jpg` (StaticFiles).

The `LivePreviewImage` component already appends `?t={tick}` for cache-busting;
the no-cache headers on the server make this redundant but harmless.

### Bug 2 — Live Feed popup shows "No active feed" always

**Root cause:** `LiveFeed.tsx` was a pure static placeholder — no hooks, no API calls,
no way to know which job was active.

**Fix 2a — Operations.tsx:** `window.open` now passes `?job={activeJobId}` when a job
is active: `/live-feed?job=<uuid>`. If no job is active, opens with no param (placeholder).

**Fix 2b — LiveFeed.tsx:** Full implementation replacing the static placeholder:
- Reads `?job=` from URL search params (`useSearchParams`)
- Gets `backendUrl` from Zustand appStore (same localStorage as Operations window)
- Polls `GET /api/v1/videos/processing-status/{jobId}` every 2 s for status/FPS/faces/alerts
- Polls `GET /api/v1/videos/preview-image/{jobId}?t={tick}` every 2 s for the annotated frame
- Shows spinner until first frame loads (`onLoad` callback gates visibility)
- Footer status bar: live dot + status label + FPS + face count + alert count + % complete
- Shows placeholder when no `?job=` param

### Build
`tsc && vite build` — 0 TypeScript errors. Pre-existing chunk-size warning only.

---

## Active threshold config (local .env / config.py defaults)

| Field | Value |
|---|---|
| `MATCH_CONFIDENCE_THRESHOLD` | 0.68 |
| `VALIDATOR_MIN_DETECTOR_CONFIDENCE` | 0.70 |
| `ALERT_MIN_CONFIDENCE_FLOOR` | 0.72 |
| `ENABLE_EMERGENCY_RECALL_MODE` | False |
| `ENABLE_ADAPTIVE_LOAD_GOVERNANCE` | False |

Emergency relaxation floor in pipeline.py: `target_min_conf=0.45`, `target_min_cutoff=0.50`.

---

## Previous session changes (still valid)

### embedder.py — re-detection fallback with det_score guard
Buffalo_l sometimes returns DetectedFace with embedding=None. Re-detect, but reject
if best.det_score < 0.70 (non-face object guard).

### video_service.py — stream URL support
Stream URLs (`http://`, `https://`, `rtsp://`) bypass `_safe_video_path()`.

### cameras.py / AndroidCameraSource
RTSP test-connect, AndroidCameraSource for MJPEG HTTP (IP Webcam).

## Files changed this session
ecoface_lite/api/routers/processing.py   (preview-image no-cache endpoint)
frontend/src/api/hooks.ts                (previewUrl → no-cache endpoint)
frontend/src/pages/Operations.tsx        (window.open passes ?job=)
frontend/src/pages/LiveFeed.tsx          (full implementation replacing placeholder)

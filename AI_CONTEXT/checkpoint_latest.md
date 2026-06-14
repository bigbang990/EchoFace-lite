# Checkpoint — 2026-06-14 — AndroidCameraSource (VSL Phase 3)

## Phase
VSL Phase 3 — AndroidCameraSource standalone implementation
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

## Backend change this session

### AndroidCameraSource rewrite (ecoface_lite/input_sources/android_source.py)

Previous implementation was a thin subclass of RTSPSource targeting RTSP endpoints.
Replaced with a full standalone BaseVideoSource implementation targeting the MJPEG
HTTP endpoint (`http://<phone-ip>:8080/video`) of the Android IP Webcam app.

**Key design decisions:**

**Persistent connection**
- `connect()` opens `cv2.VideoCapture(stream_url)` once and holds it
- `get_frame()` calls `grab()` / `retrieve()` on the same persistent object
- Cap object is never opened/closed per frame

**Stale-frame flush**
- IP Webcam maintains an internal MJPEG buffer
- Every `get_frame()` call calls `grab()` first (discards buffered frame) then
  `retrieve()` (decodes the newest frame)
- Prevents multi-second frame lag at typical 15–30 fps phone capture rates

**Failure tracking + inline reconnect**
- `consecutive_failures` counter increments on every failed `grab()` or `retrieve()`
- Counter resets to 0 on any successful read
- Threshold: 3 consecutive failures triggers `_reconnect()`
- Backoff sequence: 5 s → 10 s → 30 s (index capped at 2, i.e. 30 s max)
- `_reconnect_attempts` tracks which backoff slot to use, resets on successful `connect()`

**HealthStatus mapping**
SourceStatus has no WARNING variant; UNKNOWN is used as the intermediate state:
- `consecutive_failures == 0`            → ONLINE
- `consecutive_failures 1–2`             → UNKNOWN (degraded, reconnect not triggered)
- `consecutive_failures >= 3`            → RECONNECTING
- `_cap is None`                         → OFFLINE

**Capability properties**
- `supports_live` → True
- `supports_historical` → False
- `supports_ptz` → False
- `get_historical_stream()` → raises NotImplementedError

**Tested URL format**
http://192.168.1.21:8080/video  (MJPEG from IP Webcam app)

**Source type:** SourceType.ANDROID

**No files touched other than android_source.py**
- `source_registry.py` already had the ANDROID case and the import — unchanged
- `__init__.py` already exported AndroidCameraSource — unchanged
- `base.py`, `rtsp_source.py`, `video_file.py` — not touched

---

## Previous session UI changes (still valid)

### Administration panel (frontend/src/pages/Administration.tsx)
- RegisterCameraModal: fixed 422 error — POST body now sends `label` (not `name`)

### Other UI changes from previous session
See git log for Administration.tsx, LiveFeed.tsx, Operations.tsx, Overview.tsx,
AlertDetail.tsx, Layout.tsx, App.tsx, hooks.ts, index.css.

## TypeScript
- tsc --noEmit: 0 errors (last verified previous session)

## Files changed this session
ecoface_lite/input_sources/android_source.py  (rewritten — standalone BaseVideoSource)
frontend/src/pages/Administration.tsx          (label field fix in RegisterCameraModal)

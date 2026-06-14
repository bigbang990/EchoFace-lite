# Checkpoint — 2026-06-14 — VSL Phase 5 complete: NVR/DVR Integration

## Phase
VSL Phase 5 — NVR/DVR Enterprise Integration (complete)
Branch: `vsl-phase2-location-health`
All prior VSL phases (1–4) intact and verified.

## Regression baseline metrics (30/30 pass)
identity_switch_rate: 0.000
stable_matches: green
confirmation_rate: green
validator_rejection_rate: green
bbox_jitter: green
Test suite: 30 tests, 0 failed

---

## VSL Phase 5 changes

### Enum — `ecoface_lite/input_sources/base.py`
- `SourceType.DVR = "dvr"` added (NVR was already present from Phase 1 stub)
  Both NVR and DVR are now first-class source types.

### DB model — `ecoface_lite/db/models.py`
5 new columns on `cameras`:
| Column | Type | Purpose |
|---|---|---|
| `onvif_host` | VARCHAR(255) | NVR IP or hostname |
| `onvif_port` | INTEGER | ONVIF service port (default 80) |
| `onvif_username` | VARCHAR(128) | ONVIF login |
| `onvif_password_enc` | TEXT | base64-encoded — never in config/env |
| `dvr_clip_dir` | VARCHAR(1024) | operator drop directory for DVR exports |

### DB migrations — `ecoface_lite/db/session.py`
5 ALTER TABLE patches added to `_sqlite_apply_schema_patches()`:
  `onvif_host`, `onvif_port`, `onvif_username`, `onvif_password_enc`, `dvr_clip_dir`

### New file — `ecoface_lite/input_sources/nvr_source.py`

**NVRSource(RTSPSource)**:
- Live path: inherited RTSPSource.get_frame() — no changes to live pipeline
- Historical: ONVIF GetReplayUri() → time-windowed RTSP URL → OpenCV VideoCapture
  Same technique as VideoFileSource — frames yielded until EOF or end_time
- `get_device_info()`: ONVIF GetDeviceInformation (model, firmware, serial)
- `discover(timeout_seconds)` classmethod: WS-Discovery probe, returns candidate list
  NEVER auto-registers — operator-triggered only via GET /cameras/discover-onvif
- `supports_historical = True`
- `onvif-zeep` import is lazy (inside method bodies only); module loads without it
  ImportError at call time includes `pip install onvif-zeep` instruction

**DVRSource(RTSPSource)**:
- Live path: inherited RTSPSource.get_frame()
- Historical: scans `dvr_clip_dir` for video files (.mp4/.avi/.mkv etc.),
  picks best match by mtime proximity to requested start_time,
  delegates to `VideoFileSource.get_historical_stream()` — zero new packages
- `_find_clip(start, end)`: window match ±24h, fallback to closest mtime
- `supports_historical = True`
- Raises FileNotFoundError with clear message if no clips found

**Helpers** (module-level):
- `_encode_password(plaintext) -> str`: base64 for storage
- `_decode_password(enc) -> str`: base64 decode, falls through if not base64

### Modified — `ecoface_lite/input_sources/source_registry.py`
- Imports `NVRSource`, `DVRSource`
- `build_source()` dispatch extended:
  - `"nvr"` → `NVRSource(onvif_host, onvif_port, onvif_username, onvif_password_enc)`
  - `"dvr"` → `DVRSource(dvr_clip_dir)`
  - Both validate required fields with clear ValueError messages before construction

### Modified — `ecoface_lite/input_sources/__init__.py`
- `NVRSource`, `DVRSource` added to imports and `__all__`

### Modified — `ecoface_lite/api/schemas.py`
- `CameraOut`: added `onvif_host`, `onvif_port`, `onvif_username`, `dvr_clip_dir`
  (`onvif_password_enc` is NEVER returned — password stays in DB only)
- `CameraCreate`: added same fields + `onvif_password` (plaintext input, encoded on write);
  `source_type` pattern extended to `^(file|rtsp|android|nvr|dvr)$`
- New schemas:
  - `ONVIFDeviceInfo`: manufacturer, model, firmware_version, serial_number, hardware_id, host, port
  - `ONVIFDiscoveryResult`: xaddrs, types, scopes (one per discovered device)
  - `NVRCredentialsUpdate`: PATCH body for credential updates only

### New file — `ecoface_lite/api/routers/nvr.py`
- `POST /cameras/{id}/nvr/test-onvif` → `ONVIFDeviceInfo`
  Connects ONVIF, returns device info. 501 if onvif-zeep absent, 502 on connection fail.
- `PATCH /cameras/{id}/nvr/credentials` → `{updated: true, camera_id: N}`
  Encodes password before storage. PATCH semantics — only present fields updated.
- `GET /cameras/discover-onvif?timeout=5.0` → `list[ONVIFDiscoveryResult]`
  WS-Discovery scan. 501 if onvif-zeep absent. Opt-in, never auto-registers.

### Modified — `ecoface_lite/api/routers/cameras.py`
- `POST /cameras`: persists all 5 NVR/DVR fields; `onvif_password` encoded to
  `onvif_password_enc` at write time (plaintext never stored)

### Modified — `ecoface_lite/api/main.py`
- Imports `nvr` router; registers at `/api/v1`

---

## Architecture state (VSL Phases 1–5 complete)

```
Input sources:
  base.py              → BaseVideoSource ABC
  video_file.py        → VideoFileSource (file + historical)
  rtsp_source.py       → RTSPSource (live, exponential backoff)
  android_source.py    → AndroidCameraSource (RTSPSource subclass)
  nvr_source.py        → NVRSource (ONVIF live+historical), DVRSource (RTSP+clip dir)
  source_registry.py   → dispatches file/rtsp/android/nvr/dvr

NVR/DVR API:
  POST /api/v1/cameras/{id}/nvr/test-onvif      (ONVIF device info)
  PATCH /api/v1/cameras/{id}/nvr/credentials    (update ONVIF auth)
  GET  /api/v1/cameras/discover-onvif           (WS-Discovery, opt-in)

Historical search (Phase 4):
  VideoFileSource  → frame seek by epoch anchor
  NVRSource        → ONVIF GetReplayUri -> RTSP playback
  DVRSource        → operator-exported clip -> VideoFileSource delegation

onvif-zeep dependency:
  NOT installed by default. Required for:
    - NVRSource.get_historical_stream()
    - NVRSource.get_device_info()
    - NVRSource.discover()
  Install: pip install onvif-zeep
  All three raise ImportError with the install command if absent.
  DVRSource requires NO additional packages.
```

## Smoke test results (VSL Phase 5 verification — 10/10)
1. SourceType.NVR + SourceType.DVR defined
2. NVRSource instantiated — supports_live=True, supports_historical=True, supports_ptz=False
3. DVRSource instantiated — supports_historical=True
4. NVRSource.get_historical_stream raises ImportError with pip hint (no onvif-zeep)
5. DVRSource.get_historical_stream raises FileNotFoundError on empty clip dir
6. NVRSource.discover raises ImportError with pip hint (no onvif-zeep)
7. SourceRegistry dispatches nvr->NVRSource, dvr->DVRSource
8. NVR router has all 3 routes (test-onvif, credentials, discover-onvif)
9. Password base64 encode/decode round-trip correct
10. CameraCreate accepts source_type=nvr with ONVIF fields

## VSL roadmap complete
All 5 VSL phases implemented:
  Phase 1: BaseVideoSource abstraction, VideoFileSource, RTSPSource, SourceRegistry
  Phase 2: Location hierarchy (Site->Zone->Camera), health monitor background task
  Phase 3: AndroidCameraSource, MultiSourceScheduler, USE_VSL_FRAME_PATH flag
  Phase 4: Historical footage search, get_historical_stream, historical sightings API
  Phase 5: NVRSource (ONVIF), DVRSource (clip dir), NVR management API

## Next roadmap item
INC API Phase B — engine calls INC HTTP (true separation)
or Intelligence Layer Phase 10 — Cross-Camera Intelligence
(per roadmap.md)

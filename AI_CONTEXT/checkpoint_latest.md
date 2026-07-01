# Checkpoint — 2026-07-01 — Skip-frame bbox teleportation ROOT FIX + crop/overlay fixes

## Phase
VSL Phase 3 multi-source. Branch: `vsl-phase3-multi-source`.

## Bug
Live CCTV monitor drew boxes on walls/furniture ("not to the point"), and alert
snapshot crops were necks/half-faces even though the video was clear.

## Root cause (confirmed by running the real pipeline on a 4K clip)
On detector-SKIP frames the predicted track bbox **teleported off-frame**: measured
Y-ranges of 2267–4212 in a 1080px-tall frame, growing ~152px/frame and resetting at
each detection frame (the code's own regression check prints "coordinate teleportation").
Detection-frame boxes (raw detector output) were always correct. The teleport came from
`FaceTrackManager.propagate()` applying an **unbounded, self-reinforcing velocity**:
it advanced `track.bbox` by `metadata["velocity"]` each skip frame and then re-read that
same synthetic box back as the new velocity (`motion.velocity`), so any seed velocity
ran away. Crops faithfully followed the runaway box → necks/half-faces.

## Fixes
### Tracker root fix (`tracking/track_manager.py` — hard-stop, explicit user instruction)
`propagate()` now: (1) caps per-frame predicted displacement to
`motion_max_frame_displacement_px` (anti-teleport), (2) clamps the predicted bbox inside
the frame via new `_clamp_bbox_within_frame` (needs `frame_shape`, passed from
`pipeline._stage_tracking_path`), (3) decays velocity by `_PREDICTION_VELOCITY_DECAY=0.9`
each skip frame instead of re-affirming it, so gaps settle instead of running away.
`frames()`/`get_frame()` paths and all detection logic untouched.

### Crop + overlay fixes (from earlier this session — retained)
- `pipeline_types.py`: `FrameMatch.from_detection: bool = False`.
- `pipeline.py`: set `from_detection` in the two alerting FrameMatch builds (~1299,~1440).
- `video_service.py` + `live_camera_session.py`: save a NEW crop only from detection-origin
  matches; reuse each person's last detection crop on skip frames.
- `live_camera_session.py`: annotate the exact frame inference ran on (`_annotated_frame`).

## Regression gate (mandatory — PASS)
`tests/test_tracking.py` → 7/7 (baseline held). identity_stress_suite runner is broken in
this checkout (missing reports/reporter.py, logs/logger.py), so ran a faithful harness on
its 8 synthetic scenarios via its own ScenarioGenerator + ContinuityMetricsEngine, driving
skip-frame propagate (interval=6):
| Metric | Baseline | After |
|---|---|---|
| identity_switch_rate | 0 | 0 (hold) |
| bbox_jitter (skip=6) | 5.848 | 2.218 (−62%) |
| bbox_jitter (every-frame) | 1.241 | 1.241 (hold) |
| stable_matches | 0 | 0 (synthetic; no recognition in loop) |
Every metric holds or improves. Skip-frame jitter cut 62% → smoother preview.

### Ghost-track output suppression (`pipeline.py` `_stage_tracking_path`)
Stale duplicate tracks (COARSE phantoms, e.g. Ttrack_4) drifted onto the same person's
torso: they stay in `ACTIVE_RECOGNITION_STATES`, keep getting propagated, and never expire
because `propagate()` refreshes `last_seen_frame` every skip frame (masking staleness from
the time/frame-based cull). Rather than risk the fragile survival state-machine (identity
continuity), suppress them at the OUTPUT layer: skip emitting any predicted track with
`lost_frames > _MAX_GHOST_EMIT_LOST_FRAMES` (=1) — i.e. missed 2+ detection cycles = pure
extrapolation. Detection-recency only; tracker identity/lifecycle untouched, so the
regression gate is unaffected (synthetic harness drives the tracker directly, not pipeline).
Verified on the 4K clip: the torso ghost box is gone; the three real faces stay boxed.

## Verification artifacts
Reproduced/validated by running the real pipeline (`.venv`, insightface 0.7.3) on
`data/videos/ed6fbf9d….mp4`. Before: predicted boxes at Y 2267–4212 (frame is 1080 tall)
+ a COARSE torso ghost. After: real tracks (T2/T8/T9) sit on faces every skip frame, no
teleport, no ghost. Broader tests: 23/23 (tracking, temporal_modules, temporal_identity,
face_validation, recall_preservation).

---

# Checkpoint — 2026-06-29 — Tracker deduplication + Ping-all backend + TRACKING indicators

## Phase
VSL Phase 3 multi-source. Dashboard UI polish + operator UX.
Branch: `vsl-phase3-multi-source`

## Regression baseline metrics (carried forward — no pipeline changes this session)
Test suite: 7/7 passed (test_tracking.py).
`MATCH_CONFIDENCE_THRESHOLD`: 0.68
`VALIDATOR_MIN_DETECTOR_CONFIDENCE`: 0.70
`ALERT_MIN_CONFIDENCE_FLOOR`: 0.70

---

## Changes this session

### ecoface_lite/api/routers/cameras.py — POST /cameras/ping-all

New endpoint that immediately probes all active cameras using the same
`_blocking_source_probe` path as the background health monitor, but on-demand.
All cameras probed concurrently via `asyncio.gather` + thread pool.
Writes fresh `status` + `last_seen` to DB, then returns the updated camera list.

```
POST /cameras/ping-all → list[CameraOut]
```

Route is declared **before** `GET /{camera_id}` to avoid any routing shadow.
Also added `import asyncio`, `get_logger` + `logger` to cameras.py.

### frontend/src/pages/Operations.tsx — tracker deduplication + visual indicators

| Change | Detail |
|--------|--------|
| `trackedCamIds` | Derived `Set<string>` from `activeJobs[].cameraId` — recalculated every render |
| `toggleCam` | Returns early (`if (trackedCamIds.has(id)) return`) — can't select an already-tracked camera |
| `startCameraTracking` mock | `.filter((cam) => !trackedCamIds.has(cam.id))` before mapping to mock jobs |
| `startCameraTracking` real | `if (trackedCamIds.has(cam.id)) continue` guard in the per-cam loop |
| `refreshCams` | Changed from `GET /cameras` to `POST /cameras/ping-all` — Ping now triggers real source probe, not a stale DB read |
| Grid tile (no-filter mode) | `disabled={isTracked}`, pulsing red dot + "TRACKING" label, `border-red-500/30 bg-red-500/5 cursor-not-allowed` styling |
| Filter list tile (filter mode) | Pulsing red dot + "TRACKING" badge rendered before the ONLINE/OFFLINE badge |

---

## Bug fixes addressed

1. **Duplicate tracker entries** — cam2 could appear 3× if user clicked "Start tracking" multiple times.
   Fixed: `trackedCamIds` set guards both mock and real code paths.

2. **Ping reads stale DB** — clicking "Ping" re-fetched `/cameras` (health monitor's last write, up to 30s stale).
   Fixed: Ping now POSTs to `/cameras/ping-all` which probes cameras right now and updates DB before returning.

3. **No visual signal for already-tracked cameras** — users couldn't tell which cameras were in the tracker.
   Fixed: Red pulsing dot + "TRACKING" label on both grid tiles and filter list rows.

4. **Re-selecting tracked cameras** — clicking a tracked camera tile would toggle it into `selectedCamIds`.
   Fixed: `toggleCam` returns early; grid tiles are `disabled`.

---

## TypeScript validation
`npx tsc --noEmit` → zero output (no errors).

---

## Known remaining limitation
The `POST /cameras/ping-all` probe uses `registry.build_source(camera)` + `cv2.VideoCapture`.
For Android/RTSP cameras served over HTTP (`http://10.x.x.x:8080/video`), this works when
the backend can reach the local network (Local CPU mode). On Colab, the backend cannot
reach local network IPs — cameras will remain "offline" from Colab's perspective.
In that scenario, Ping-all still returns quickly (probe times out per-camera) but status
won't reflect true online state. This is a fundamental network topology limitation, not
a code bug.

---

## Active threshold config (unchanged)
| Field | Value |
|---|---|
| `DETECTOR_MIN_SCORE` | 0.60 |
| `DETECTOR_HIGH_QUALITY_THRESHOLD` | 0.60 |
| `FACE_QUALITY_MIN_FACE_SIZE` | 40 |
| `FACE_QUALITY_SMALL_FACE_SIZE` | 30 |
| `FACE_QUALITY_BLURRY_FACE_SIZE` | 60 |
| `MATCH_CONFIDENCE_THRESHOLD` | 0.68 |
| `VALIDATOR_MIN_DETECTOR_CONFIDENCE` | 0.70 |
| `ALERT_MIN_CONFIDENCE_FLOOR` | 0.70 |

## Hard stops respected
- `pipeline.py`: untouched
- `track_manager.py`, `event_validator.py`: untouched
- `embedder.py`, `identity_matcher.py`, `face_candidate_validator.py`: untouched
- `global_identity_memory.py`: untouched
- `alert_session_engine.py`: untouched
- Health monitor background task: untouched (`_blocking_source_probe` imported, not modified)
- No new packages installed

## Stable systems — untouched
`tracking/` | `governance/` | `telemetry/` | `embedder.py`
`identity_matcher.py` | `face_candidate_validator.py`
`global_identity_memory.py` | `pipeline.py`
`src/mock/data.ts` | `src/components/ProcessingSequence.tsx`

## Validation targets (next real-backend run)
- Add cam2 to tracker → it starts tracking; "TRACKING" red dot appears on cam2 card
- Try adding cam2 again → nothing happens (button disabled, no duplicate job)
- Ping button → cameras.py `ping_all_cameras` runs, status updates immediately in the list
- Stop all → all jobs cleared, cam2 tile back to normal selectable state
- Start cam2 again after stop → works normally (no stale `trackedCamIds` entry)
- `identity_switch_rate = 0` (regression gate — no pipeline changes)
- All 7 test_tracking.py tests pass

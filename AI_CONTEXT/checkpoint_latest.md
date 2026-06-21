# Checkpoint — 2026-06-21 — confirmation queue profiler

## Phase
VSL Phase 3 — multi-source stream URL routing
Branch: `vsl-phase3-multi-source`
All prior VSL phases (1–5) intact and verified.

## Regression baseline metrics
Test suite: 7/7 passed (test_tracking.py — re-verified this session).

---

## Changes this session

### Confirmation queue profiler (this session)

New toggleable monkey-patch instrumentation for confirmation-queue methods.
Zero runtime cost when disabled — only imported if `ENABLE_CONFIRMATION_PROFILING=true`.

**New files:**

| File | Purpose |
|---|---|
| `ecoface_lite/diagnostics/__init__.py` | Empty — makes `diagnostics` a proper Python package |
| `ecoface_lite/diagnostics/confirmation_profiler.py` | `install_confirmation_profiler(cls, dump_path, dump_interval_calls=200)` |

**Modified files:**

| File | Change |
|---|---|
| `ecoface_lite/api/main.py` | `import os` added; 5-line profiler activation block in `lifespan()` after `await init_db()` |

**Env vars:**

| Var | Default | Effect |
|---|---|---|
| `ENABLE_CONFIRMATION_PROFILING` | `""` (off) | Set `"true"` to activate profiler |
| `CONFIRMATION_PROFILE_DUMP_PATH` | `data/logs/confirmation_profile.json` | Atomic-write dump target |

**Profiler behaviour:**
- Wraps `_admit_or_queue_pending`, `_decay_pending`, `_best_match` on `FaceTrackManager` class
- `_admit_or_queue_pending`: records `duration_ms`, `pending_len_at_entry`, `spurious_match_count` (bbox_iou ≥ temporal_min_track_iou AND centroid distance > 40px)
- `_decay_pending`: records `duration_ms`, `pending_len_at_entry`
- `_best_match`: records `duration_ms`
- Stats accumulate for full process lifetime (never cleared after dump)
- Atomic dump every `dump_interval_calls` total calls via `.tmp` + `os.replace`
- Idempotent: module-level `_profiler_installed` flag + `cls._profiler_installed` attribute check
- Defensive: wrapper exceptions → `logger.warning()`, fall through to original method

**Hard stops respected:**
- No logic changes to `_admit_or_queue_pending`, `_decay_pending`, `_best_match`
- No changes to `RecognitionPipeline`, alert engine, DB models
- No new dependencies

**Not yet tested on Colab GPU — next step is a soak run with `ENABLE_CONFIRMATION_PROFILING=true`**

---

---

## Changes this session

### Instrumentation — decode call count + resize timing (this session)

**Bottleneck audit context (measured data):**
- video_decode_duration_ms = 21.7ms/call; ~358 emitted frames assumed 1:1 with reads → ~7.77s estimated
- Remaining unaccounted gap: ~10s of 34.93s total
- `_resize_for_inference()` confirmed OUTSIDE `total_frame_processing_duration` boundary → genuine gap contributor

**New metrics added:**

| Metric | Type | File:line | Notes |
|---|---|---|---|
| `video_decode_call_count` | single observe in `finally` block | `video_file.py:frames()` | Total `cap.read()` calls per job incl. skipped frames + EOF; reveals true decode call count when `video_frame_skip > 1` |
| `resize_for_inference_duration_ms` | per-emitted-frame observe | `video_service.py:229` (before `process_frame()`) | OUTSIDE `total_frame_processing_duration` — not double-counted |

**Key finding confirmed:** `_resize_for_inference()` at `video_service.py:229` is called BEFORE `pipeline.process_frame()` at line 230. `total_frame_processing_duration` timer wraps `_process_frame_staged()` INSIDE `process_frame()` at `pipeline.py:282-283`. Therefore resize is an unaccounted gap contributor.

**`video_decode_call_count` design note:** Counter increments on every `cap.read()` (including skipped frames and EOF read). With `video_frame_skip=2` and 358 emitted frames, expected count = 717 (716 raw reads + 1 EOF). If count = 359, skip=1 was confirmed. Single `metrics.observe("video_decode_call_count", float(_decode_calls))` in `finally` block → `averages["video_decode_call_count"]` = exact count (1 sample).

---

### Instrumentation — video decode + job setup timing

**Bottleneck audit context (measured data, do not re-investigate):**
- AI pipeline (`_process_frame_staged`): ~16.2s (43% of 37.3s job)
- Preview writer (`preview_generation_time`): ~1.7s (5%) — cleared
- Overlay render: ~0.15s — negligible
- DB commits (`db_commit_duration_ms`): ~0.04s — negligible
- Unaccounted gap: ~19.2s (51%) — suspected video decode and/or job setup

**New metrics added (all auto-exposed via `/api/v1/observability/metrics` → `averages`):**

| Metric | Type | Where |
|---|---|---|
| `video_open_duration_ms` | per-job single observe | `video_file.py:frames()` around `cv2.VideoCapture()` |
| `video_decode_duration_ms` | per-frame observe (~888 samples/job incl. final failed read) | `video_file.py:frames()` around `cap.read()` |
| `job_setup_duration_ms` | per-job single observe | `video_service.py:process_prerecorded_video()`, from local imports to `started_at` |

**Files changed:**

| File | Change |
|---|---|
| `ecoface_lite/input_sources/video_file.py` | Added `from time import perf_counter` + `from ecoface_lite.core.metrics import metrics`; wrap `cv2.VideoCapture()` and `cap.read()` in `frames()` |
| `ecoface_lite/services/video_service.py` | `_setup_t0 = perf_counter()` before setup phase; `metrics.observe("job_setup_duration_ms", ...)` reuses `started_at` as end marker |

**Pre-run hypothesis for 19.2s gap:**
- `video_decode_duration_ms` is the primary suspect. 887 H264 reads × 5–20ms each = 4–18s on CPU; on Drive-mounted storage, the JPEG-compressed `frame_XXXXXX.jpg` persistence (preview writer) is cleared at 1.7s, so this is a pure decode cost.
- `job_setup_duration_ms` covers gallery load (2 SQL queries + numpy blob deserialization) + `VideoPreviewWriter(mkdir)`. Expected < 500ms.
- `video_open_duration_ms` expected < 100ms (container header parse).
- Note: `count_emitted_frames()` in `run_async_video_job` also opens VideoCapture once (NOT timed — it's a pre-flight call separate from the `frames()` iterator). Its cap is opened, reads `CAP_PROP_FRAME_COUNT`, and releases immediately — should be < 50ms.

**Key architectural finding (HARD STOP — do not fix this session):**
- `get_recognition_pipeline()` at `bootstrap.py:230-235` IS correctly a process-wide singleton — model weights load once. NOT a per-job cost.
- `load_gallery(session)` at `video_service.py:212` IS called per job inside `process_prerecorded_video`. Gallery is NOT cached between jobs. For small galleries: negligible. For large galleries: could grow.

---

### FIX 1 — Correct misleading Avg FPS label

**Root cause (confirmed via audit):**
- `averages["average_processing_fps"]` ≈ 37 FPS — dominated by 887 per-frame observations of `1/total_frame_processing_duration` inside `pipeline.py:295`. Measures AI pipeline throughput only; excludes video decode, disk I/O, DB writes.
- True wall-clock throughput = `avg_fps` from `VideoJobDiagnostics` = frames / total job wall-clock ≈ 0.81 FPS. Was only written to DB, never exposed via MetricsRegistry.

**What was changed:**
- `video_service.py` — added `metrics.observe("avg_fps_wall_clock", emitted_count / duration)` once per job at end-of-job (alongside existing `average_processing_fps` observe). Single observation → `averages["avg_fps_wall_clock"]` = exact wall-clock value.
- `dashboard/app.py:682–686` — changed from 4-column to 5-column row; "Avg FPS" renamed to "Avg FPS (wall-clock)" reading `avg_fps_wall_clock`; added "AI Pipeline FPS" reading `average_processing_fps`. Both have tooltip help text explaining the distinction.
- `frontend/src/types/index.ts` — added `ai_fps: number` to `SystemMetrics` interface.
- `frontend/src/mock/data.ts` — `fps` mock changed to 0.81 (wall-clock); `ai_fps: 37.4` added.
- `frontend/src/api/hooks.ts` — `fps` now normalizes `avgs.avg_fps_wall_clock`; `ai_fps` normalizes `avgs.average_processing_fps`. FPS history sparkline tracks `m.ai_fps` (AI pipeline FPS — more stable, more useful for monitoring).
- `frontend/src/pages/SystemHealth.tsx` — grid changed to `grid-cols-5`; "AVG FPS" tile now shows wall-clock with detail "end-to-end wall-clock"; new "AI FPS" tile added showing `m.ai_fps`. FPS history chart header relabeled "AI PIPELINE FPS HISTORY"; sparkline summary shows `m.ai_fps`.

---

### FIX 2 — Reduce debug-crop I/O cost

**Root cause (confirmed via audit):**
- `_save_rejected_debug_crops()` called on every 10th frame (interval=10). With ~887 frames: ~88 sampled frames × ~11 yellow boxes = ~968 crops × 2 disk writes (cv2.imwrite + Path.write_text) = ~1,936 synchronous I/O ops on Colab Drive. This is the dominant cost in the ~1076s gap between AI time (~24s) and wall-clock job time (~1100s).

**Math:**
- Target: 50–100 crops per job. At ~11 yellow/frame: need ~5–9 sampled frames.
- Interval=100: 887 ÷ 100 ≈ 8 sampled frames × 11 = **~88 crops, ~176 disk writes** — within target range.
- Hard cap 200: guards against pathologically crowded videos (e.g., 50+ yellow faces/frame) regardless of interval.
- Expected speedup: ~1,936 → ~176 disk writes = **~11× fewer I/O ops** per job.

**What was changed:**
- `ecoface_lite/core/config.py:382` — `rejected_face_snapshot_interval` default changed from `10` to `100` (overridable via `REJECTED_FACE_SNAPSHOT_INTERVAL` env var).
- `ecoface_lite/services/video_service.py` — added module-level `_job_crop_counts: dict[str | None, int] = {}` and `_CROP_SAVE_HARD_CAP = 200`. In `_save_rejected_debug_crops`: added function-level early return when `_job_crop_counts[job_id] >= 200`, and inner per-face `break` guard; increment `_job_crop_counts[job_id]` after each successful write pair.

### Files changed (this session)

| File | Change |
|---|---|
| `ecoface_lite/core/config.py` | `rejected_face_snapshot_interval` default: 10 → 100 |
| `ecoface_lite/services/video_service.py` | Module-level cap dict + constant; hard cap in `_save_rejected_debug_crops`; `avg_fps_wall_clock` observe at end-of-job |
| `dashboard/app.py` | 4-col → 5-col; "Avg FPS" → wall-clock; "AI Pipeline FPS" added |
| `frontend/src/types/index.ts` | `ai_fps: number` added to `SystemMetrics` |
| `frontend/src/mock/data.ts` | `fps` → 0.81, `ai_fps: 37.4` added |
| `frontend/src/api/hooks.ts` | `fps` → `avg_fps_wall_clock`; `ai_fps` → `average_processing_fps`; history tracks `ai_fps` |
| `frontend/src/pages/SystemHealth.tsx` | `grid-cols-4` → `grid-cols-5`; "AI FPS" tile added; chart relabeled |

### Hard stops respected
- `pipeline.py`, `EventValidator`, box color logic, detection thresholds: untouched
- `GOVERNANCE_MAX_CANDIDATE_QUEUE_SIZE`: untouched
- No async/threading refactor of crop-saving path

---

## Previous session — 3 new candidate queue health metrics (still valid)

### Task — 3 new candidate queue health metrics

**What was added (read-only instrumentation, no logic changed):**

| Metric | Type | Where read |
|---|---|---|
| `max_queue_size_seen` | `metrics.observe` → `averages` | `/observability/metrics`, `/debug/stream-metrics` |
| `avg_queue_size_seen` | `metrics.observe` → `averages` | `/observability/metrics`, `/debug/stream-metrics` |
| `queue_full_duration_ms` | `metrics.increment` → `counters` | `/observability/metrics`, `/debug/stream-metrics` |

**Definition:**
- `max_queue_size_seen` — running peak of `candidate_queue_size` since session start, reset on `metrics.reset()`.
- `avg_queue_size_seen` — running mean of `candidate_queue_size` across all `_check_congestion` calls (mirrors the averaging semantics already used for `candidate_queue_size`).
- `queue_full_duration_ms` — cumulative milliseconds the queue has been at or above `governance_max_candidate_queue_size`. Uses entry-time tracking: increments by the inter-call delta each time the queue is still full on successive calls to `_check_congestion`.

**Saturation threshold used:** `self._cfg.governance_max_candidate_queue_size` (config default 25, Colab SERVER_ENV sets 15).

### Files changed

| File | Change |
|---|---|
| `ecoface_lite/ai_engine/tracking/track_manager.py` | 2 new instance fields in `__init__`; 18 lines added to `_check_congestion` |
| `ecoface_lite/api/routers/stream.py` | import `metrics`; 3 new keys in `GET /debug/stream-metrics` response |
| `dashboard/app.py` | "Candidate Queue Health" panel (3 `st.metric` widgets) added to Observability tab |
| `frontend/src/types/index.ts` | 3 new fields on `SystemMetrics` interface |
| `frontend/src/mock/data.ts` | 3 new fields in `mockSystemMetrics` |
| `frontend/src/api/hooks.ts` | normalize `ctrs.queue_full_duration_ms` + 2 `avgs` fields in `useSystemMetrics` |
| `frontend/src/pages/SystemHealth.tsx` | "CANDIDATE QUEUE HEALTH" panel (3 tiles) inserted above PLATFORM CONFIG |

### Hard stops respected
- `governance_max_candidate_queue_size` threshold: read-only reference, not modified
- `RecognitionPipeline`, `ByteTrack`, `EventValidator`: untouched
- `detection_optimizer.py`, `bootstrap.py`: untouched
- `candidate_queue_drops` / `candidate_ingestion_rejections`: untouched
- `load_shedding_active` trigger and naming: untouched

---

## Previous session — event_validator confidence gate (still valid)
`ecoface_lite/ai_engine/event_validator.py` confidence floor gate (3 lines).
`tests/test_event_validator.py` regression test.
See prior checkpoint detail if needed — not modified this session.

## Previous session — live MJPEG streaming (still valid)
`ecoface_lite/services/live_camera_session.py`, `ecoface_lite/api/routers/stream.py`,
stream router registered in `main.py`, `LiveFeed.tsx` minimal `<img>` replacement.

## Active threshold config (.env / config.py defaults)
| Field | Value |
|---|---|
| `MATCH_CONFIDENCE_THRESHOLD` | 0.68 |
| `VALIDATOR_MIN_DETECTOR_CONFIDENCE` | 0.70 |
| `ALERT_MIN_CONFIDENCE_FLOOR` | 0.72 |
| `GOVERNANCE_MAX_CANDIDATE_QUEUE_SIZE` | 32 (default) / 15 (Colab SERVER_ENV) |
| `ENABLE_EMERGENCY_RECALL_MODE` | False |
| `ENABLE_ADAPTIVE_LOAD_GOVERNANCE` | False |

## Resolution-cap experiment (Jun 2026) — CLOSED, negative result
DETECTOR_MAX_INPUT_PIXELS swept 409600 → 2073600 (5x) on crowd video 
(498be8fd80624e69ac7c6441a6c43b2d.mp4, 150 frames). avg_face_size flat 
at 51.5px implied width across entire range — zero effect. 
DETECTOR_MAX_INPUT_PIXELS is NOT the lever for small-face rejections.

NEXT: check settings.video_inference_width in config.py and 
_resize_for_inference() in video_service.py — this earlier resize step 
runs BEFORE the detector pipeline and may be the actual resolution 
ceiling, independent of DETECTOR_MAX_INPUT_PIXELS. One grep, not a 
re-sweep.

Track-state cross-job leak (negative avg_track_lifetime): FIXED via 
reset_session() in pipeline.py + track_manager.py + video_service.py. 
Verified clean across 3-video soak sequence in same session.

Standalone pipeline construction (bypassing FastAPI startup) runs 
detector on CPU, not GPU — ~300x slower than production. Valid for 
face-size/rejection-count comparisons, NOT valid for speed/FPS numbers.
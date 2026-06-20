# Checkpoint — 2026-06-20 — FPS label fix + debug-crop I/O reduction

## Phase
VSL Phase 3 — multi-source stream URL routing
Branch: `vsl-phase3-multi-source`
All prior VSL phases (1–5) intact and verified.

## Regression baseline metrics
Test suite: 31/31 passed (re-verified this session after both fixes).

---

## Changes this session

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

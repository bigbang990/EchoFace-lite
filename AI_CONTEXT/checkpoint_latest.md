# Checkpoint — 2026-06-20 — candidate queue health metrics

## Phase
VSL Phase 3 — multi-source stream URL routing
Branch: `vsl-phase3-multi-source`
All prior VSL phases (1–5) intact and verified.

## Regression baseline metrics
Test suite: 31/31 passed (carried forward from prior session — no new tests added this session; new metrics are additive observe/increment calls with no branching logic to test separately).

---

## Changes this session

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

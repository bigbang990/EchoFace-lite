# EchoFace Lite — roadmap

## Completed phases
- Phase 2A–2C: detection, embeddings, matching, tracking,
  continuity stabilization, bbox stabilization,
  identity stress suite, real video validation
- Phase 2D Part 1: CPU-aware adaptive interval + governance fixes
- Phase 2C.3B: real video validation pipeline
- Phase 5: platform bootstrap, ghost track hard kill,
  validator confidence floors, timing telemetry fix,
  detector abstraction layer (SCRFD + YOLOv8 providers)
- Phase 2A detection upgrade: multiscale fusion and observability
- Phase 6: YOLOv8-face detector abstraction — MERGED to main v0.6.0
  17ms GPU inference, 183 stable matches, 2 alerts, 0 identity switches
  torch.load patch for PyTorch 2.6 weights_only. Production verified on T4.

## Active
Single-server consolidation complete — all routes on :8000; inc_server.py kept for
Phase B standalone use. Colab: one process, one ngrok tunnel.
See AI_CONTEXT/checkpoint_latest.md for full architecture state.

## INC API phases

### Phase A — ✅ DONE + consolidated
- `ecoface_lite/api/inc_server.py` — INC router definitions (NOT standalone currently)
- All routes merged into `ecoface_lite/api/main:app` on :8000
- Frontend: `incUrl` in store, BackendPanel INC section, all incident/person calls
  routed to `incUrl`; for single-server point both `backendUrl` and `incUrl` at :8000
- Colab: `python scripts/colab_start.py` — one process, one ngrok tunnel

### Phase B — engine calls INC HTTP (true separation)
Goal: engine never writes incident/sighting data directly; INC is the DB master.

Changes needed:
1. `ecoface_lite/core/config.py` — add `INC_API_URL: str | None = None`
2. `ecoface_lite/api/inc_server.py` — add `POST /api/v1/gallery` endpoint
   that returns `[{person_id, embedding_b64}]` for all OPEN incident persons
3. `ecoface_lite/services/video_service.py` — `load_gallery()` calls
   `GET {INC_API_URL}/gallery` when INC_API_URL is set (falls back to local DB)
4. `ecoface_lite/services/video_service.py` — instead of creating Sighting in DB,
   call `POST {INC_API_URL}/incidents/{id}/sightings`
5. `ecoface_lite/api/inc_server.py` — add `POST /api/v1/embed` proxy
   (INC calls this on enrollment; engine generates the 512-dim vector and returns it)
   This makes enrollment work from any host without InsightFace locally.

Result: engine can run on Colab GPU; INC can run on cheap cloud/local. No shared filesystem.

### Phase C — public viewer (SSE stream)
1. `ecoface_lite/api/inc_server.py` — add `GET /api/v1/incidents/{id}/stream` (SSE)
   - subscriber list per incident_id
   - when a sighting is POSTed (Phase B), broadcast to all subscribers
2. Simple React page at `/view/{caseRef}` — no auth, just case ref
   - connects to SSE stream, shows live timeline updates
   - can be shared publicly ("watch case INC-004 live")
3. CORS on INC server already `allow_origins=["*"]` — ready for public access

## Intelligence Layer — Phases 8–11

### Phase 8 — Alert Session Engine
*The noise problem, solved.*
Branch: `phase8-pipeline-decompose` (active)

Goals:
- `active_alert_registry` in-memory dict keyed by `(incident_id, camera_id)`
- Every recognition match appends a sighting, updates `last_seen_at`
- Alert created only on first match of a new session
- Session closes after configurable gap threshold (default 60s)
- New alert opens if person reappears after gap or crosses zone boundary

Data introduced: `Alert` (one per session, operator-facing) · `Sighting` (one per frame match)

Key issue: on restart, query DB for open alerts within last N minutes and rebuild registry.

Regression gate:
- Existing identity stress suite passes unchanged
- New metric: `alerts_per_continuous_presence` must equal 1

---

### Phase 9 — Forensic Dashboard + Case UI
*The operator sees the system for the first time.*

Goals:
- Live alert feed — active sessions with person, location, first/last seen, sighting count
- Alert detail view — full sighting timeline with snapshots and per-frame confidence
- Case management UI — link alerts to incidents, all alerts under one incident
- Camera health panel — online/offline, frame rate, last heartbeat
- VSL registration panel — add video file, RTSP, Android IP camera sources

Regression gate:
- Phase 8 alert session behavior intact
- Dashboard loads without error on CPU and Colab

---

### Phase 10 — Cross-Camera Intelligence
*The system understands movement, not just presence.*

Goals:
- Zone-based alert clustering — same person, same zone, within time window → one alert
- Movement tracker — same person, different zones → movement timeline event
- Confidence fusion — multiple weak sightings across cameras → fused score
  Formula: `P(combined) = 1 - Π(1 - Pᵢ)` — independent probability fusion, dissertation-defensible
- `EchoFace Intelligence Layer` component formally introduced in architecture

Data introduced: `MovementEvent` · `FusedConfidence`

Regression gate:
- Phase 8 session behavior unchanged
- Single-camera alerts work without movement context

---

### Phase 11 — Intelligence Layer Hardening + Commercial API
*The system becomes a product.*

Goals:
- Alert Correlation Engine formalized as standalone component
- Alert levels: Sighting → Candidate → Verified → Critical (configurable per incident type)
- REST API endpoints for third-party integration (Face++ pattern)
- Incident/case system as primary API surface — not raw face matches
- All manual operator promotions/demotions logged with operator ID and timestamp

Regression gate:
- Full identity stress suite passes
- All Phase 8–10 behaviors verified under multi-incident load

---

## Video Source Layer — see AI_CONTEXT/vsl_roadmap.md

```
VSL Phase 1  →  abstraction foundation, file + RTSP sources
VSL Phase 2  →  location hierarchy, health monitoring
VSL Phase 3  →  android camera, multi-source pipeline
VSL Phase 4  →  historical footage, search pipeline
VSL Phase 5  →  NVR/DVR, enterprise integration (post-dissertation)
```

VSL Phase 1 runs in parallel with or just after Phase 8 — `BaseVideoSource` is the
prerequisite for multi-source work in Phase 3 and historical search in Phase 4.

---

## Pending
- Phase 7: resolve capped_detector_resolution in detection_optimizer.py
  (separate from settings flag, hard-coded in optimizer logic)
- Phase 7: decouple face_app from YOLO path in bootstrap
  (InsightFace loads unnecessarily on YOLO provider)
- Phase 2D Part 2: Detection Truthfulness Validation Framework
  (raw vs validator-passed detections, small-face acquisition curves, crowd recall)
- Phase 2D backlog: profile softening — validator cutoff reduction for
  LEFT_PROFILE/RIGHT_PROFILE (pose bucket not exposed at governance eval site)
- Dashboard GPU/CPU toggle
- Academic: mid-semester presentation, final presentation

## Explicitly NOT in scope
- New AI models beyond current detector swap
- Governance/telemetry/embedding/identity rewrites
- Dashboard redesign (until core model stable)

## Branch index
- main — stable, v0.6.0 tagged
- phase6-detector-abstraction — merged to main
- phase6-colab-gate-test — source branch for Phase 6 work (same commits)
- phase5-colab-ready — remote only, merged work
- phase3-async-stabilization, phase4-gpu-ready — remote, prior phases
- archive/* — poisoned/failed experiments, do not rebase
- experiment-resolution-cap — remote only

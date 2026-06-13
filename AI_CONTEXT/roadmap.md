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
INC API split — Phase A done; Option 1 (Colab dual-server) unblocks hybrid immediately.
See AI_CONTEXT/checkpoint_latest.md for full architecture state.

## INC API phases

### Phase A — ✅ DONE
- `ecoface_lite/api/inc_server.py` — second entry point, INC routers only, same DB
- Frontend: `incUrl` in store, BackendPanel INC section, all incident/person calls
  routed to `incUrl`, video/metrics stay on `backendUrl`
- For Colab hybrid: run both servers, expose via 2 ngrok tunnels
  → `python scripts/colab_start.py` handles setup

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
- CCTV stream ingestion
- Incident/Case system (future — see decisions.md)
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

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
- Phase 6: detector abstraction layer — SCRFD + YOLOv8 providers,
  DETECTOR_PROVIDER env var, detect() implemented, Colab verified.
  117.9 FPS on T4. Merged to phase6-detector-abstraction 2026-06-08.

## Active
(nothing — awaiting next task)

## Pending
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
- main — stable
- phase6-detector-abstraction — Phase 6 complete, pending merge to main
- phase6-colab-gate-test — source branch for Phase 6 work (same commits)
- phase5-colab-ready — remote only, merged work
- phase3-async-stabilization, phase4-gpu-ready — remote, prior phases
- archive/* — poisoned/failed experiments, do not rebase
- experiment-resolution-cap — remote only

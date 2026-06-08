## Checkpoint — 2026-06-09 — Phase 6 MERGED v0.6.0

### Done
Phase 6 merged to main. Tagged v0.6.0.
YOLOv8-face GPU detector production verified:
  detector_runtime_ms: 16.8ms
  stable_matches: 183
  alerts_per_video: 2
  identity_switch_rate: 0
  average_processing_fps: 56

### Known debt
1. ghost_survival > 18s in crowd scenes (starvation override)
   — Phase 5 known issue, tracked
2. capped_detector_resolution fixed at 480px
   — detection_optimizer.py:112, separate from settings flag
   — Phase 7 target
3. face_app loads on YOLO path unnecessarily
   — Phase 7 target

### State
- Working: full pipeline on main, YOLOv8 GPU verified
- Next session: Phase 7 — detection_optimizer resolution cap fix,
  then Phase 2D Part 2 Detection Truthfulness Validation Framework

### Branch
main — v0.6.0 tagged

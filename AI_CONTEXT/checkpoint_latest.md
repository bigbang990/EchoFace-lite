## Checkpoint — 2026-06-09 — Phase 7 Task 1 DONE

### Done
Fixed DETECTOR_RESOLUTION_CAP_ENABLED in
detection_optimizer.py._select_detector_size.
Cap was unconditional (line 267 had no flag check).
Now respects settings.detector_resolution_cap_enabled.
When False: returns original size unchanged.
When True: applies gpu_detector_resolution ceiling as before.

### Files changed
- ecoface_lite/ai_engine/detection_optimizer.py
  _select_detector_size: added flag check before min() cap

### Regression gate result
15 tests pass. 9 pre-existing failures (Phase 6 merge debt,
unchanged before/after). Zero new failures introduced.

### Pre-existing test failures to fix in Phase 7 (backlog)
9 failures in governance/recall/tracking suites.
Must be resolved before Phase 8 starts.

### State
- Working: resolution cap now controllable via env var
  DETECTOR_RESOLUTION_CAP_ENABLED=0 → full resolution
  DETECTOR_RESOLUTION_CAP_ENABLED=1 → gpu_res ceiling
- Next task: Phase 7 Task 2 — decouple face_app
  from YOLO path in bootstrap.py

### Branch
phase7-resolution-cap-fix — not yet on main

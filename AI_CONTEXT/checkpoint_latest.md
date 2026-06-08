## Checkpoint — 2026-06-09 — Phase 7 resolution cap fix

### Done
Fixed DETECTOR_RESOLUTION_CAP_ENABLED flag not respected in _select_detector_size.
The GPU ceiling (min(size, gpu_res)) was unconditional — applied regardless of the flag.
Added guard: when flag=False, return raw adaptive size without the gpu_res ceiling.
Fix: detection_optimizer.py _select_detector_size, 3-line insertion.

Pre-existing test failures (not introduced by this change):
  test_adaptive_governance::test_adaptive_detector_interval
  test_recall_preservation::test_threshold_hysteresis
  test_tracking (4 failures)
  — all fail identically with and without the change, from phase6 merge debt.

### State
- Working: phase7-resolution-cap-fix branch, fix committed
- Next: decouple face_app from YOLO path in bootstrap.py
  then Phase 2D Part 2 Detection Truthfulness Validation Framework
- Pre-existing test debt: 9 failures across governance/recall/tracking suites
  (separate task — do not conflate with this fix)

### Branch
phase7-resolution-cap-fix — not yet on main

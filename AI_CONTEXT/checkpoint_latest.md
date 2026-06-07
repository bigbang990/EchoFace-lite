## Checkpoint — 2026-06-08 — torch.load patch approach

### Done
Replaced unbounded add_safe_globals allowlist with a scoped torch.load patch
in YOLOv8FaceDetector.__init__. Patch forces weights_only=False for the
YOLO() call only, then restores the original torch.load in a finally block.
No add_safe_globals calls remain. Only yolov8_detector.py was modified.

### State
- Working: phase6-detector-abstraction, torch.load patch applied
- Blocked on: re-run Colab cell to confirm no UnpicklingError, then real video test
- Next task: confirm 422 on enroll, run real video through pipeline.
  Targets: detector_runtime_ms < 50ms,
           validator_rejection_rate < 0.30,
           identity_switch_rate = 0,
           stable_matches > 35

### Branch
phase6-detector-abstraction — not yet on main

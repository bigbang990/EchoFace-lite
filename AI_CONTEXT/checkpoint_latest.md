## Checkpoint — 2026-06-08 — YOLOv8 weights_only fix

### Done
Fixed PyTorch 2.6 weights_only UnpicklingError in YOLOv8FaceDetector.__init__.
PoseModel + DetectionModel (+ Conv, C2f, SPPF, Detect, Pose) added to
torch.serialization.add_safe_globals() before YOLO load.
Only yolov8_detector.py was modified — surgical, one-function change.

### State
- Working: phase6-detector-abstraction, weights_only fix applied
- Blocked on: real video telemetry test on Colab GPU before merging to main
- Next task: re-run Colab cell, verify pipeline alive, run real video test.
  Targets: detector_runtime_ms < 50ms,
           validator_rejection_rate < 0.30,
           identity_switch_rate = 0,
           stable_matches > 35

### Branch
phase6-detector-abstraction — not yet on main

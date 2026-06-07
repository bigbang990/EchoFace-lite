## Checkpoint — 2026-06-08 — Phase 6 Phase 3 DONE

### Done
YOLOv8FaceDetector.detect() implemented.
Format conversion: xyxy boxes + 5-point keypoints
→ BoundingBox + FaceLandmarks + DetectedFace.
Landmark order: [left_eye, right_eye, nose,
left_mouth, right_mouth] — matches FaceLandmarks
convention confirmed in detector.py audit.
Interface complete: all 3 abstract methods implemented.

### Files changed
- ecoface_lite/ai_engine/detection/detectors/
  yolov8_detector.py — detect() implemented

### Verify result
Local verify: import of YOLOv8FaceDetector succeeded.
Instantiation fails with ModuleNotFoundError (torch) —
expected; lazy imports in __init__ are Colab-only.
Logic verified by inspection.

### State
- Working: full detector implementation, interface complete
- torch/ultralytics: lazy-imported inside __init__ — module
  importable without them; instantiation requires Colab
- Known issue: face_app loads unconditionally in bootstrap
  even on YOLO path — InsightFace weights load unnecessarily.
  Acceptable now. Fix in Phase 7 when embedder is decoupled.
- Next task: Phase 6 Phase 4 — Colab end-to-end test
  and final commit to phase6-detector-abstraction branch

### Branch
phase6-colab-gate-test

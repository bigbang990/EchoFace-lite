## Checkpoint — 2026-06-08 — Phase 6 Phase 2 DONE

### Done
Detector abstraction scaffold complete.
DETECTOR_PROVIDER env var wired in bootstrap.py.
YOLOv8FaceDetector stub created — import verified.
Both SCRFD and YOLO paths import cleanly.

### Files changed
- ecoface_lite/core/platform_bootstrap.py — "detector_provider": "scrfd" added
  to both GPU and CPU branches
- ecoface_lite/ai_engine/bootstrap.py — provider selection block replaces
  hard-coded InsightFaceDetector construction
- ecoface_lite/ai_engine/detection/detectors/yolov8_detector.py — stub (new)

### Design note
face_app = _create_face_analysis(settings) sits before the if/else so the
embedder (unchanged, line 115) always has it. YOLO path does not use face_app
for the detector. Phase 3 will revisit embedder wiring if needed.

### State
- Working: scaffold verified, both paths import OK
- SCRFD path: fully functional (no behaviour change)
- YOLO path: stub — detect() raises NotImplementedError; Phase 3 pending
- Blocked on: nothing
- Next task: Phase 6 Phase 3 — real detect() implementation
  (format conversion, landmark mapping, FaceLandmarks construction)

### Branch
phase6-colab-gate-test

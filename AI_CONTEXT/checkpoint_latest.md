## Checkpoint — 2026-06-08 — Phase 6 COMPLETE

### Done
Full detector abstraction layer implemented and merged.
SCRFD and YOLOv8-face are both working providers.
DETECTOR_PROVIDER env var selects at runtime.
117.9 FPS on T4 GPU confirmed (Gate D).
Pushed to phase6-detector-abstraction.

### Files changed (full phase summary)
- ecoface_lite/core/platform_bootstrap.py — "detector_provider": "scrfd" in both branches
- ecoface_lite/ai_engine/bootstrap.py — provider selection block (SCRFD/YOLO if/else)
- ecoface_lite/ai_engine/detection/detectors/yolov8_detector.py — full implementation (new)
- scripts/download_yolov8_face.py — gdown, derronqi Drive ID (new)
- AI_CONTEXT/ — all 4 context files updated

### Known technical debt
- face_app loads unconditionally in bootstrap even on YOLO path.
  InsightFace weights load unnecessarily. Track as Phase 7.

### State
- Working: full pipeline, both detector providers,
  Colab T4 GPU, 117.9 FPS YOLOv8 confirmed
- Blocked on: nothing — awaiting Colab end-to-end verification,
  then merge phase6-detector-abstraction to main
- Next task: run Colab verification cell, then merge to main

### Branch
phase6-detector-abstraction — pushed to remote

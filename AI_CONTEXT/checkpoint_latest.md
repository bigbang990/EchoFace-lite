## Checkpoint — 2026-06-08 — Phase 6 feasibility gate

### Done
Ran YOLOv8-face feasibility gate (Phase 1).
Result: NO-GO
Weights file `weights/yolov8n-face.pt` is absent and the download script
`scripts/download_yolov8_face.py` does not yet exist.

### Gate results
A: FAIL  B: not run  C: not run  D: not run — fps not measured  E: skip

### State
- Working: full pipeline, InsightFace/SCRFD on CPU (CPUExecutionProvider)
- platform_bootstrap.py: on phase5-colab-ready branch only — not merged to main
- AI_CONTEXT: complete and accurate
- Blocked on: Gate A — weights file missing; download script also missing
- Next task: create scripts/download_yolov8_face.py, download yolov8n-face.pt,
  then re-run Phase 6 Phase 1 feasibility gate

### Branch
phase6-detector-abstraction (not yet created locally)

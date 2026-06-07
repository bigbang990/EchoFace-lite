## Checkpoint — 2026-06-08 — Phase 6 Phase 1 re-run (partial)

### Done
Download script created at scripts/download_yolov8_face.py.
weights/yolov8n-face.pt downloaded (6.2 MB, HuggingFace source).
Gate A passed locally.
Gate B failed — torch and ultralytics are not installed on this local machine.
The project runs on Google Colab; local Python 3.10 has no ML packages.
Instruction: do not install automatically.

### Gate results
A: PASS (5.96 MB)  B: FAIL (torch not installed locally)
C: not run  D: not run  E: skip

### State
- Working: weights downloaded and valid
- scripts/download_yolov8_face.py: created and tested
- Blocked on: Gates B–E must run on Google Colab (no local torch/ultralytics)
- Next task: re-run gates B–E in Colab by mounting the repo and running
  python _gate_bc.py (or equivalent inline), then report results here

### Branch
phase5-colab-ready

## Checkpoint — 2026-06-09 — Phase 7 Task 2 DONE

### Done
Decoupled face_app from YOLO path in bootstrap.py.
`_create_face_analysis(settings)` now only runs on the SCRFD branch.
YOLO branch sets `face_app = None`; InsightFaceEmbedder handles None
via `_ensure_app()` lazy-load fallback (no embedder.py change needed).

### Files changed
- ecoface_lite/ai_engine/bootstrap.py
  - Line 115: `face_app = _create_face_analysis(settings)` → `face_app = None`
  - Line 145 (SCRFD else-branch): added `face_app = _create_face_analysis(settings)`
    before InsightFaceDetector construction

### State
- Working: YOLO path no longer loads InsightFace FaceAnalysis at startup.
  SCRFD path unchanged — still loads FaceAnalysis once and shares instance.
  Embedder lazy-loads its own app if face_app is None (YOLO path).
- Next task: Phase 7 Task 3 — profile softening
  (validator cutoff reduction for LEFT/RIGHT_PROFILE)

### Branch
phase7-resolution-cap-fix — not yet on main

### Pre-existing test failures (backlog, unchanged)
9 failures in governance/recall/tracking suites.
Must be resolved before Phase 8 starts.

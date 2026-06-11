## Checkpoint — 2026-06-11 — GPU pipeline fully verified

### Done
Platform bootstrap now uses torch.cuda for GPU detection.
bootstrap.py overrides insightface_ctx_id from PLATFORM.
GPU governance path confirmed: interval_ceiling=12,
detector_budget=150ms, hardware_backend_type=1.

### Key metrics (GPU path confirmed)
- hardware_backend_type: 1 (GPU) — first time correct
- total_faces_detected: 687 (was 285 on CPU path)
- detector_rejection_rate: 0.347 (was 0.672)
- recall_per_resolution: 0.65 (was 0.33)
- face_visibility_ratio: 0.95 (was 0.81)
- alerts_per_video: 3
- stable_matches: 50
- identity_switch_rate: 0
- average_processing_fps: 81
- detector_runtime_ms: 13.2ms

### Root cause chain — fully resolved
1. platform_bootstrap checked onnxruntime → FIXED (torch.cuda)
2. prepare_for_detection had second cap → FIXED (flag check)
3. insightface_ctx_id not overridden → FIXED (bootstrap.py)

### State
- Working: full GPU pipeline, all governance paths correct
- Next: create AI_CONTEXT/schema.md, then multi-photo
  enrollment, then merge to main as v0.7.0

### Branch
phase6-detector-abstraction
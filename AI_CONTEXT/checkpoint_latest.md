## Checkpoint — 2026-06-11 — Phase 7B multi-photo enrollment complete

### Phase
7B multi-photo enrollment

### Status
complete

### New endpoint
POST /persons/{person_id}/photos

### Files changed (Phase 7B)
- ecoface_lite/services/person_service.py — add_photos_to_person stub replaced with real impl
- ecoface_lite/api/routers/persons.py — new route added (imports: select, Person, PersonEnrollMultiOut)

### Implementation notes
- Max 5 photos per call enforced in service (HTTPException 400)
- Per-photo dedupe via sha256 against FaceEmbedding.ingest_sha256 for this person
- cv2.imdecode → pipeline.enroll_reference_embedding() → FaceEmbedding stored as float32 bytes
- ValueError from enroll (no face / quality rejected) → rejected counter + reason string
- 404 guard in router before service call; 413 per-file size check matches existing POST /persons pattern
- Existing POST /persons route signature unchanged; pipeline.py not touched

### Previous metrics (GPU path — still baseline)
- hardware_backend_type: 1 (GPU)
- total_faces_detected: 687
- detector_rejection_rate: 0.347
- recall_per_resolution: 0.65
- face_visibility_ratio: 0.95
- alerts_per_video: 3
- stable_matches: 50
- identity_switch_rate: 0
- average_processing_fps: 81
- detector_runtime_ms: 13.2ms

### Root cause chain — fully resolved (carried from prior checkpoint)
1. platform_bootstrap checked onnxruntime → FIXED (torch.cuda)
2. prepare_for_detection had second cap → FIXED (flag check)
3. insightface_ctx_id not overridden → FIXED (bootstrap.py)

### State
- Working: full GPU pipeline + multi-photo enrollment API
- Next: run regression suite, then merge to main as v0.7.0

### Branch
phase6-detector-abstraction

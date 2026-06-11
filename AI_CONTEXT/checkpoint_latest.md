## Checkpoint — 2026-06-12 — Phase 8C case management API complete

### Phase
8C case management API — cameras, incidents, sightings

### Status
complete

### New routers
- ecoface_lite/api/routers/cameras.py
  - POST   /api/v1/cameras          → create camera, 201
  - GET    /api/v1/cameras          → list all
  - GET    /api/v1/cameras/{id}     → get one, 404 if missing
  - PATCH  /api/v1/cameras/{id}     → update is_active only
  - DELETE /api/v1/cameras/{id}     → hard delete, 204

- ecoface_lite/api/routers/incidents.py
  - POST   /api/v1/incidents                    → create, status="open", 201
  - GET    /api/v1/incidents                    → list, optional ?status= filter
  - GET    /api/v1/incidents/{id}               → get one + sightings (IncidentDetailOut)
  - PATCH  /api/v1/incidents/{id}/status        → update status (open/active/closed)
  - POST   /api/v1/incidents/{id}/sightings     → add sighting, 201
  - GET    /api/v1/incidents/{id}/sightings     → list sightings

### New schemas (ecoface_lite/api/schemas.py)
CameraOut, CameraCreate,
IncidentOut, IncidentCreate, IncidentStatusUpdate,
SightingOut, SightingCreate

### Files changed
- ecoface_lite/api/schemas.py — 7 new schemas appended
- ecoface_lite/api/main.py — cameras + incidents routers registered
- ecoface_lite/api/routers/cameras.py — new file
- ecoface_lite/api/routers/incidents.py — new file (includes IncidentDetailOut)

### Regression gate result
- 29/29 pass (test_health.py pre-broken — PersonEnrollMultiOut import error,
  pre-existing before Phase 8A)
- All 11 new routes verified via app.routes introspection
- Existing router prefixes unchanged
- pipeline.py, bootstrap.py, video_service.py untouched

### GPU baseline metrics (still valid)
- hardware_backend_type: 1 (GPU)
- total_faces_detected: 687
- detector_rejection_rate: 0.347
- stable_matches: 50
- identity_switch_rate: 0
- average_processing_fps: 81
- detector_runtime_ms: 13.2ms

### Previous phase
Phase 8A pipeline decompose step 1:
- ecoface_lite/ai_engine/experiment_coordinator.py (ExperimentCoordinator, 4 methods)
- pipeline.py: 1561 → 1525 lines

### Next extraction target
GovernanceCoordinator (pending separate session)
- Candidates: _apply_load_governance (lines 479–681), governance state vars in __init__
- Hard constraint: governance vars tightly coupled to process_frame — careful analysis required

### Branch
phase8-pipeline-decompose

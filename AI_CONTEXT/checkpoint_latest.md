## Checkpoint — 2026-06-12 — Phase 8B incident-person link endpoint complete

### Phase
8B incident-person link endpoint

### Status
complete

### New routes (ecoface_lite/api/routers/incidents.py)
- POST   /api/v1/incidents/{id}/persons/{person_id}
    Load incident + persons (selectinload), load person, 404/409 guards,
    append to incident.persons, return IncidentPersonOut (201)
- DELETE /api/v1/incidents/{id}/persons/{person_id}
    Load incident + persons (selectinload), filter out person, commit, 204
- GET    /api/v1/incidents/{id}/persons
    Load incident + persons (selectinload), 404 if missing,
    return list[PersonOut]

### New schema (ecoface_lite/api/schemas.py)
- IncidentPersonOut: incident_id, person_id, person_name

### Files changed
- ecoface_lite/api/schemas.py — IncidentPersonOut appended
- ecoface_lite/api/routers/incidents.py — selectinload import, Person import,
  IncidentPersonOut/PersonOut schema imports, 3 new route handlers

### Regression gate
- 29/29 pass (test_health.py pre-broken — PersonEnrollMultiOut import, pre-existing)
- All incident routes verified via app.routes introspection:
  /incidents, /incidents/{id}, /incidents/{id}/persons,
  /incidents/{id}/persons/{person_id}, /incidents/{id}/sightings,
  /incidents/{id}/sightings, /incidents/{id}/status

### Previous phases on this branch
- Phase 8C: cameras.py + incidents.py routers, 7 new schemas
- Phase 8A: ExperimentCoordinator extracted from pipeline.py (1561→1525 lines)
- Phase 7B: session lifecycle isolation, multi-photo enrollment

### GPU baseline metrics (still valid)
- hardware_backend_type: 1 (GPU)
- stable_matches: 50
- identity_switch_rate: 0
- average_processing_fps: 81
- detector_runtime_ms: 13.2ms

### Next target
GovernanceCoordinator extraction (pending separate session)

### Branch
phase8-pipeline-decompose

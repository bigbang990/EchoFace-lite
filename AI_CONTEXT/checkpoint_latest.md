## Checkpoint — 2026-06-12 — Phase 8A pipeline decompose step 1 complete

### Phase
8A pipeline decompose — step 1: ExperimentCoordinator extraction

### Status
complete

### What was done
Extracted 4 experiment/observability methods from RecognitionPipeline into a new
coordinator class. RecognitionPipeline retains all public method signatures;
each method now delegates to self._experiment_coordinator.

### New file
- ecoface_lite/ai_engine/experiment_coordinator.py
  - class ExperimentCoordinator
  - __init__(experiment_exporter, detection_metrics, settings, notes_tracker, event_timeline)
  - export_experiment_session(...)
  - record_experiment_adjustment(...)
  - get_experiment_notes()
  - get_event_timeline_statistics()

### Changes to pipeline.py
- Added import: ExperimentCoordinator
- Added self._experiment_coordinator = ExperimentCoordinator(...) at end of __init__
  (experiment_exporter=None, notes_tracker=None, event_timeline=None — these were
  never wired up in any builder; detection_metrics and settings passed through)
- Replaced 4 method bodies with one-line delegation
- pipeline.py line count: 1525 (was 1561 — reduced by 36 lines)

### Regression gate result
- 29/30 tests pass — test_health.py has a pre-existing import error
  (PersonEnrollMultiOut missing from schemas.py) unrelated to this change.
  Confirmed pre-existed via git stash test.
- RecognitionPipeline.__init__ signature: unchanged
- process_frame, enroll_reference_embedding, test_match_frame: unchanged
- bootstrap.py: no new imports

### GPU baseline metrics (still valid — no pipeline logic changed)
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

### State
- Working: full GPU pipeline + ExperimentCoordinator wired
- ExperimentCoordinator currently has all optional deps as None (no external builder
  ever set _experiment_exporter, _notes_tracker, _event_timeline — this was true
  before extraction too)

### Next extraction target
GovernanceCoordinator (pending separate session)
- Candidates: _apply_load_governance (lines 479–681), governance state vars in __init__
- Hard constraint: governance vars are tightly coupled to process_frame flow —
  requires careful feasibility analysis before extraction

### Branch
phase8-pipeline-decompose

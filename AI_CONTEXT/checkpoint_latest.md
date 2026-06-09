## Checkpoint — 2026-06-09 — Phase 7 COMPLETE

### Done
Task 1: resolution cap fix (detection_optimizer.py)
  _select_detector_size now respects DETECTOR_RESOLUTION_CAP_ENABLED flag.

Task 2: face_app decoupled from YOLO path (bootstrap.py)
  _create_face_analysis() only runs on SCRFD branch.
  YOLO path passes face_app=None; embedder lazy-loads its own app.

Task 3: profile softening CASE B — TODO documented (face_candidate_validator.py)
  pose_bucket not available at cutoff site. Debt documented at
  proposal_min_validation_score check. Phase 7B backlog item.

Task 4: confirmation queue CASE B — comment added (track_manager.py)
  Hard limit EXISTS at 25 with quality-based eviction already implemented.
  Queue is bounded. Saturation symptom (counter=77, stable_matches=0) is
  caused by 77-face crowd churning 25 queue slots faster than candidates
  can accumulate confirmation hits. Fix: reduce cap to 15 via env var.
  Comment added at enforcement line (line ~583) documenting tuning path.
  SERVER_ENV recommendation: "GOVERNANCE_MAX_CANDIDATE_QUEUE_SIZE": "15"

### Phase 7B backlog (do before Phase 8)
1. Profile softening: thread pose_bucket into validate_face_candidate.
   classify_pose_bucket() exists in pose_estimator.py. 3-line change once
   pose_bucket is added as a parameter to the function.
2. Fix 8 pre-existing test failures in governance/tracking/recall suites
   before Phase 8 starts.
3. Multi-photo enrollment (Phase 7B feature).
4. Session isolation (/sessions/begin + /sessions/end).

### Regression gate result — Phase 7 final
22 pass, 8 fail. All 8 failures are pre-existing (governance/tracking/recall).
Zero new failures introduced across all 4 Phase 7 tasks.

### Files changed across Phase 7
- ecoface_lite/ai_engine/detection_optimizer.py (Task 1)
- ecoface_lite/ai_engine/bootstrap.py (Task 2)
- ecoface_lite/ai_engine/face_candidate_validator.py (Task 3)
- ecoface_lite/ai_engine/tracking/track_manager.py (Task 4)

### State
- Working: all Phase 7 core fixes committed and pushed
- Next session: Phase 7B — fix pre-existing test failures,
  then merge phase7-resolution-cap-fix into phase6-detector-abstraction

### Branch
phase7-resolution-cap-fix — push complete

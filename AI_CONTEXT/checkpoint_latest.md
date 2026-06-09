## Checkpoint — 2026-06-09 — Phase 7 Task 3 DONE

### Done
CASE B: documented as TODO — pose_bucket not available at cutoff site.
`validate_face_candidate` in face_candidate_validator.py uses only raw
`pose_yaw`/`pose_pitch` floats from local `_estimate_pose()`.
`classify_pose_bucket()` / `PoseBucket` enum is never imported or called
in this file. Restructuring call stack to pass pose_bucket was forbidden.
TODO added immediately before the `proposal_min_validation_score` check
(the final cutoff gate in the function) to document the debt.

### Files changed
- ecoface_lite/ai_engine/face_candidate_validator.py
  Added 2-line TODO comment before line 325
  (`proposal_min_validation_score` cutoff check)

### Regression gate
22 pass, 8 fail. All 8 failures are pre-existing (governance/tracking/recall
suites). Zero new failures introduced.

### State
- Working: validator unchanged in behavior; debt documented at the right site.
  Profile softening deferred until pose_bucket is threaded into the function.
- Next task: Phase 7 Task 4 — confirmation queue saturation fix
  (track_manager.py)

### Branch
phase7-resolution-cap-fix — not yet merged to main

### Pre-existing test failures (backlog, unchanged)
8 failures in governance/recall/tracking suites.
Must be resolved before Phase 8 starts.
(Note: previous checkpoint reported 9; current run shows 8 — likely a
counting difference in how skipped vs failed tests were reported earlier.)

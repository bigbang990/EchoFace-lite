## Checkpoint — 2026-06-10 — Phase 7 MERGED

### Done
All Phase 7 tasks complete and merged to
phase6-detector-abstraction.
Tests: 30 pass, 0 fail.

### What was fixed
- detection_optimizer.py: resolution cap now respects
  DETECTOR_RESOLUTION_CAP_ENABLED flag
- bootstrap.py: face_app only loads on SCRFD path
- track_manager.py: crowd queue saturation documented
- All 8 pre-existing test failures resolved
- Profile softening: TODO at validator line 325,
  Phase 7B 3-line fix pending

### Phase 7B remaining (before Phase 8)
1. Profile softening — call classify_pose_bucket()
   using existing pose_yaw/pose_pitch in
   validate_face_candidate(). 3-line change.
2. Multi-photo enrollment
3. Session isolation (/sessions/begin + /sessions/end)
4. Database schema design (AI_CONTEXT/schema.md)

### Colab SERVER_ENV additions needed
"GOVERNANCE_MAX_CANDIDATE_QUEUE_SIZE": "15"
"DETECTOR_RESOLUTION_CAP_ENABLED": "0"

### State
- Working: phase6-detector-abstraction is stable,
  all tests green
- Next: Phase 7B profile softening (3-line fix),
  then database schema design, then Phase 8
- Branch: phase6-detector-abstraction

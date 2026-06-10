## Checkpoint — 2026-06-10 — Phase 7B profile softening DONE

### Done
Phase 7B profile softening complete.
Commit: 8ade34f on phase6-detector-abstraction.
Tests: 30 pass, 0 fail.

### What was implemented
- face_candidate_validator.py: classify_pose_bucket() called after
  _estimate_pose(); effective_cutoff lowered by
  validator_profile_cutoff_reduction (0.08) when pose_bucket is
  LEFT_PROFILE or RIGHT_PROFILE
- config.py: validator_profile_cutoff_reduction field added
  (default=0.08, alias=VALIDATOR_PROFILE_CUTOFF_REDUCTION, ge=0.0,
  le=0.3)

### Phase 7B remaining (before Phase 8)
1. Multi-photo enrollment
2. Session isolation (/sessions/begin + /sessions/end)
3. Database schema design (AI_CONTEXT/schema.md)

### Colab SERVER_ENV additions needed
"GOVERNANCE_MAX_CANDIDATE_QUEUE_SIZE": "15"
"DETECTOR_RESOLUTION_CAP_ENABLED": "0"

### State
- Working: phase6-detector-abstraction is stable, all tests green
- Next: multi-photo enrollment OR session isolation (user to choose)
- Branch: phase6-detector-abstraction

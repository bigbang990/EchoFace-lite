## Checkpoint — 2026-06-11 — bootstrap ctx_id fix

### Done
Added settings.insightface_ctx_id = PLATFORM["ctx_id"]
to build_recognition_pipeline() in bootstrap.py.
This propagates GPU ctx_id=0 to pipeline governance,
fixing GOVERNANCE [CPU] being used despite T4 GPU.
Commit: aa75f4b on phase6-detector-abstraction.
Tests: 30 pass, 0 fail.

### Root cause chain (now fully resolved)
1. platform_bootstrap checked onnxruntime providers
   → always returned CPU on CUDA 12.8 → FIXED (af13e77)
2. prepare_for_detection had second cap enforcement
   → FIXED (aad4fda)
3. insightface_ctx_id not overridden from PLATFORM
   → THIS FIX (aa75f4b)

### Expected outcome after this fix
Startup log: "GOVERNANCE [GPU]: interval_ceiling=12"
_select_detector_size: GPU path → 640×640
capped_detector_resolution: ~409,600

### Phase 7B remaining (before Phase 8)
1. Multi-photo enrollment
2. Session isolation (/sessions/begin + /sessions/end)
3. Database schema design (AI_CONTEXT/schema.md)

### Colab SERVER_ENV additions needed
"GOVERNANCE_MAX_CANDIDATE_QUEUE_SIZE": "15"
"DETECTOR_RESOLUTION_CAP_ENABLED": "0"

### State
- Next: re-run Colab, verify GOVERNANCE [GPU] in log,
  submit video, confirm capped_detector_resolution
  ~409,600 and stable_matches recovery
- Branch: phase6-detector-abstraction

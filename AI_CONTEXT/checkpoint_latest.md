## Checkpoint — 2026-06-08 — pre-merge housekeeping done

### Done
requirements.txt: gdown added (gdown>=4.7.0,<6.0.0, AI/CV section).
README.md: Stack table, Detector providers, Colab setup,
and Model weights sections added.

### State
- Working: phase6-detector-abstraction complete, docs updated
- Blocked on: real video telemetry test on Colab GPU
  before merging to main
- Next task: run real video through pipeline,
  collect telemetry JSON, compare vs SCRFD-CPU baseline.
  Targets: detector_runtime_ms < 50ms,
           validator_rejection_rate < 0.30,
           identity_switch_rate = 0,
           stable_matches > 35

### Branch
phase6-detector-abstraction — not yet on main

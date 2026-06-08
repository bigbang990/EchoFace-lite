# EchoFace Lite — Claude Code instructions

## Session start (mandatory)
1. Read AI_CONTEXT/architecture.md
2. Read AI_CONTEXT/checkpoint_latest.md (if exists)
3. Do NOT scan the repository
4. Do NOT read source files unless the task requires it
5. Open max 2-3 files per task

## Rules
- Every change must handle CPU and GPU paths —
  never hardcode thresholds for one platform
- Surgical changes only — do not refactor stable systems
- Diffs only in output — never regenerate full files
- Stop and report if any feasibility gate fails
- Do not install packages automatically

## Stable systems — never touch without explicit instruction
tracking/ | governance/ | telemetry/ | embedder.py
identity_matcher.py | face_candidate_validator.py
global_identity_memory.py | pipeline.py internals

## Branch strategy
New branch per phase. Never push to main directly.
Merge to main only after stress suite passes.

## Regression gate — mandatory on every phase close

Before any phase commit:
1. Run identity_stress_suite
2. Report identity_switch_rate, stable_matches,
   bbox_jitter, confirmation_rate
3. Compare vs AI_CONTEXT/checkpoint_latest.md baseline
4. If any metric regresses: STOP, report, do not merge

This rule has no exceptions.
Faster is worthless if identity continuity broke.

## Current phase
See AI_CONTEXT/roadmap.md

## Checkpoint at end of every task
Generate AI_CONTEXT/checkpoint_latest.md before stopping.

## Regression gate — runs on every phase close

Before any phase commit or merge:
1. Run identity_stress_suite
2. Collect: identity_switch_rate, stable_matches,
   bbox_jitter, confirmation_rate, validator_rejection_rate
3. Compare every metric vs the baseline in
   AI_CONTEXT/checkpoint_latest.md
4. If ANY metric regresses vs baseline:
   STOP. Do not commit. Report which metric and by how much.
5. Only if ALL metrics pass or improve: proceed with merge.

This rule has no exceptions.
Faster is worthless if identity continuity broke.
Prettier is worthless if identity continuity broke.

## Modularity rule

Every new component must follow the detector abstraction pattern:
- Define a base class or protocol with a minimal interface
- Implementation details stay inside the class
- Pipeline receives the interface, not the implementation
- DETECTOR_PROVIDER pattern: config selects the implementation

When touching an existing component that is not yet modular:
- Do not rewrite it
- Add the interface layer surgically
- Leave existing behaviour unchanged

## Observability rule

Every significant pipeline decision must emit a structured event:
  events.emit(event_type, {track_id, camera_id,
              frame_index, timestamp, payload})
Not a log line. Not a counter increment. A queryable record.
If the event table does not exist yet: log it and add a TODO.
Do not skip the emit call.

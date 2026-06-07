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

## Current phase
See AI_CONTEXT/roadmap.md

## Checkpoint at end of every task
Generate AI_CONTEXT/checkpoint_latest.md before stopping.

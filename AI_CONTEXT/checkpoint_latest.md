## Checkpoint — 2026-06-07 — AI_CONTEXT setup

### Done
Created AI_CONTEXT/ directory with 5 context files and wrote CLAUDE.md (under 60 lines).
System is now ready for low-token session starts — read architecture.md + checkpoint_latest.md
instead of scanning the repo.

### Files changed
- AI_CONTEXT/architecture.md — new: stack, constraints, detector config from config.py defaults
- AI_CONTEXT/roadmap.md — new: completed phases, active phase 6, pending work, branch index
- AI_CONTEXT/detectors.md — new: DetectedFace contract, SCRFD and YOLOv8 provider specs
- AI_CONTEXT/decisions.md — new: detector abstraction, enrollment, incident system, landmarks
- AI_CONTEXT/session_template.md — new: copy-paste start template + checkpoint template
- CLAUDE.md — new: session rules, stable systems list, branch strategy (under 60 lines)

### State
- Working: full pipeline, InsightFace/SCRFD on CPU via CPUExecutionProvider
- Blocked on: Phase 6 YOLOv8 feasibility gate (weights download + 5-point landmark inspection)
- Next task: run Phase 1 feasibility prompt against downloaded yolov8n-face.pt weights

### Branch
claude/happy-nobel-d5b2ab — worktree branch; commit and merge to main when ready

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

## Frontend

Location: `frontend/` — standalone React + TypeScript app (Vite 5).

**Stack:** React 18 · TypeScript · Tailwind CSS 3 · Framer Motion 11 ·
Zustand 4 · React Router 6 · Recharts 2 · Lucide React

**Dev server:**
```
cd frontend && npm install && npm run dev
# http://localhost:5173  — access codes: DEMO / ADMIN
```

**Stable systems in frontend — never touch without instruction:**
`src/mock/data.ts` shapes must match real API schemas exactly.
`src/components/ProcessingSequence.tsx` — animation timing is demo-calibrated.

**Frontend rules:**
- All mock data lives in `src/mock/data.ts` — shapes mirror real API responses
  so wiring is a data-source swap, not a redesign
- Access mode logic is isolated to `src/components/AccessGate.tsx` and
  `src/store/appStore.ts` — keep it swappable for real auth in a future phase
- Shared components (StatusIndicator, Timeline, ProcessingSequence) must stay
  product-agnostic — no incident-specific imports inside them
- Backend proxy: Vite forwards `/api/*` to `http://localhost:8000`

**Next phase (frontend):** Replace `src/mock/data.ts` exports with fetch/SWR
calls to `/api/v1/incidents`, `/api/v1/incidents/:id/persons`,
`/api/v1/incidents/:id/sightings`, `/api/v1/observability/*`.

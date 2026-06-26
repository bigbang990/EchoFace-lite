# Checkpoint — 2026-06-27 — alert sightings list fix + source label fix

## Phase
Alert sightings list fix + source label fix.
Branch: `vsl-phase3-multi-source`

## Regression baseline metrics (carried forward)
Test suite: 7/7 passed (test_tracking.py).
`MATCH_CONFIDENCE_THRESHOLD`: 0.68
`VALIDATOR_MIN_DETECTOR_CONFIDENCE`: 0.70
`ALERT_MIN_CONFIDENCE_FLOOR`: 0.70

---

## Changes this session

### 6 surgical fixes across two sessions

| Commit | Files | Problem | Fix |
|--------|-------|---------|-----|
| `7b0d878` feat(db) | `models.py`, `session.py`, `schemas.py` | `blur_score` / `pose_bucket` missing from Sighting ORM, schema patch, and API response | Added 2 nullable columns to ORM + 2 ALTER TABLE stmts + 2 fields in SightingOut |
| `44ddc9c` feat(sightings) | `video_service.py`, `alert_session_engine.py` | Every sighting row had `blur_score=NULL`, `pose_bucket=NULL` | Compute both from face crop at write-time; pass through `record_match()` to both audit and active sightings |
| `1a14a8d` fix(snapshots) | `video_service.py` | Snapshot crops 73–92 px wide (ArcFace min is 112 px; unreadable in dashboard) | Upscale snapshot copy to 112 px min via `cv2.INTER_CUBIC` before `imwrite`; ArcFace path unaffected |
| `66a7d16` feat(frontend) | `types/index.ts`, `AlertDetail.tsx` | Hero crop selected by confidence only — 0.758-conf back-of-head won over 0.63-conf frontal | Quality composite: `conf + pose_bonus(0.15 if FRONTAL) + blur_bonus(min(blur/500, 0.15))` |
| `91d02b8` fix(api) | `routers/alerts.py` | `GET /incidents/{id}/alerts` returned `sightings: []` despite `sighting_count >= 1` | Added `selectinload(Alert.sightings)` to list query + `with_sightings=True` to `_alert_out()` call; added `blur_score`/`pose_bucket` to `SightingOut` constructor |
| `80e900d` fix(pipeline) | `video_service.py` | Batch video sightings stored with `source: "live"` | Changed hardcoded `source="live"` to `source="video"` in `record_match()` call |

---

## Root causes fixed

### Task 1 — Sighting quality columns missing
`Sighting` ORM had no `blur_score` or `pose_bucket` columns.
`SightingOut` schema exposed neither field.
`_sqlite_apply_schema_patches()` had no ALTER for either column.
**Fix:** 3-file patch: ORM + schema + API schema.

### Task 2 — Quality fields not written at sighting create time
`record_match()` in `alert_session_engine.py` had no `blur_score`/`pose_bucket` params.
At the `video_service.py` call site, `m.face.landmarks` and `m.face.bbox` are available.
**Fix:**
- `blur_score`: `cv2.Laplacian(gray, cv2.CV_64F).var()` on the face crop.
- `pose_bucket`: `classify_pose_bucket(m.face.landmarks, m.face.bbox).name` → uppercase enum name ("FRONTAL", "PARTIAL", etc.)
- Both passed as kwargs through `record_match()` to both Sighting constructors (below-floor and active session).

### Task 3 — Snapshot crops too small
Face bbox at 56–79 px → 73–92 px padded crop. No size check before `cv2.imwrite`.
**Fix:** After computing quality fields (which need the original-size crop for accurate blur measurement), upscale if either dimension < 112 px using `cv2.INTER_CUBIC`. Upscale applied to the snapshot save copy only — ArcFace fast-path uses `rec_model.get(frame_bgr, fake_face)` with `face.kps` landmarks, never touches `_face_crop`.

### Task 4 — Hero crop selection by confidence only
`snapUrl` useMemo sorted `.sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))`.
Back-of-head crop at 0.758 confidence beat a frontal crop at 0.63.
**Fix:** Composite score with pose bonus (0.15 for FRONTAL) and blur bonus (up to 0.15). Frontal crop must only be 0.30 below the back-of-head confidence to win. Added `blur_score?: number` and `pose_bucket?: string` to the `Sighting` TypeScript interface.

---

## Active threshold config (unchanged this session)
| Field | Value |
|---|---|
| `DETECTOR_MIN_SCORE` | 0.60 |
| `DETECTOR_HIGH_QUALITY_THRESHOLD` | 0.60 |
| `FACE_QUALITY_MIN_FACE_SIZE` | 40 |
| `FACE_QUALITY_SMALL_FACE_SIZE` | 30 |
| `FACE_QUALITY_BLURRY_FACE_SIZE` | 60 |
| `MATCH_CONFIDENCE_THRESHOLD` | 0.68 |
| `VALIDATOR_MIN_DETECTOR_CONFIDENCE` | 0.70 |
| `ALERT_MIN_CONFIDENCE_FLOOR` | 0.70 |

## Hard stops respected
- `pipeline.py`: untouched
- `track_manager.py`, `event_validator.py`: untouched
- `embedder.py`, `identity_matcher.py`, `face_candidate_validator.py`: untouched
- `global_identity_memory.py`: untouched
- `alert_session_engine.py`: field pass-through only (2 new kwargs + 2 Sighting constructor writes)
- `db/`: no Alembic; schema patches only via `_sqlite_apply_schema_patches()`
- ByteTrack internals: untouched
- No new infrastructure, no new packages

## Validation targets (Colab GPU run)
- `GET /api/v1/incidents/{id}/sightings` returns `blur_score` (float) and `pose_bucket` ("FRONTAL"/"PARTIAL"/etc.) for new sightings
- Snapshot files in `data/snapshots/` are ≥ 112 × 112 px
- AlertDetail hero crop is the frontal face, not back-of-head, when both are present
- `identity_switch_rate = 0` (regression gate)
- All 7 test_tracking.py tests pass

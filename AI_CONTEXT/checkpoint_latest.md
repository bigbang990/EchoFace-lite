# Checkpoint — 2026-06-26 — detection & crop fix (foreground face recovery)

## Phase
Detection & crop quality fix — foreground face missed, face_too_small rejection, pose gate.
Branch: `vsl-phase3-multi-source`

## Regression baseline metrics (from prior checkpoint)
Test suite: 7/7 passed (test_tracking.py).
`MATCH_CONFIDENCE_THRESHOLD`: 0.68
`VALIDATOR_MIN_DETECTOR_CONFIDENCE`: 0.70
`ALERT_MIN_CONFIDENCE_FLOOR`: 0.72 → now 0.70

---

## Changes this session

### 4 surgical fixes — one commit each

| Commit | File | Before | After |
|--------|------|--------|-------|
| fix(detector): lower score gate | `config.py` | `detector_min_score=0.82`, `detector_high_quality_threshold=0.82` | both → `0.60` |
| fix(detector): lower face size gate | `config.py` | `face_quality_min_face_size=80`, `small=60`, `blurry=100` | `40`, `30`, `60` |
| fix(embedder): add pose gate | `face_quality.py` | no pose check before embedding | landmark-based pose gate, rejects PARTIAL/PROFILE |
| fix(embedder): lower alert floor | `config.py` | `alert_min_confidence_floor=0.72` | `0.70` |

---

### Fix 1 — Lower score gate for large-area faces

**Root cause:**
`detection_optimizer.evaluate()` for large faces (area_ratio ≥ 0.012) uses
`min_score = max(detector_min_score, detector_high_quality_threshold)`.
Both were 0.82. InsightFace SCRFD at 320-511 px detection assigns 0.65-0.78
to very large close-up faces (anchor mismatch at extreme size) → "weak_detector_score".

**Fix:** `detector_min_score` and `detector_high_quality_threshold`: 0.82 → 0.60.
Large faces now need ≥ 0.60. Medium faces still need 0.88. Small faces 0.93. Crowd clutter unaffected.

---

### Fix 2 — Lower face_quality_min_face_size

**Root cause:**
At coordinate_scale_factor=0.266, a face at 20 px inference → 75 px in 1920-px space.
Dynamic area check passes (3136 ≥ 3110 px²) but linear width check (80 px) rejects.
ArcFace uses landmark-based norm_crop; bbox width does not determine embedding quality.

**Fix:** `face_quality_min_face_size`: 80 → 40. Blurry floor: 100 → 60. Sharp floor: 60 → 30.
Removes "REJECTED: face_too_small" for det ≥ 0.5 faces ≥ 40 px wide.

---

### Fix 3 — Pose gate before embedding

**Root cause:**
`face_candidate_validator` classifies pose but only penalizes geometry_score — never
hard-rejects profile/partial crops. Profile crops reached ArcFace and produced
cosine < 0.45 noise without advancing recognition.

**Fix:** `face_quality.assess()` now calls `classify_pose_bucket(face.landmarks, face.bbox)`.
If pose ∈ {LEFT_PROFILE, RIGHT_PROFILE, PARTIAL}: increment `embedding_suppressed` counter,
return `FaceQualityResult(False, ..., reason="poor_face_angle")`. Faces still get yellow
tracking boxes but are not embedded. Only FRONTAL and UNKNOWN poses generate embeddings.

---

### Fix 4 — Lower alert_min_confidence_floor

**Root cause:** Crowd-video ArcFace cosine for genuine match lands at 0.63-0.72.
Prior floor 0.72 clipped valid alerts. Lowered to 0.70.

---

## Active threshold config after this session
| Field | Value |
|---|---|
| `DETECTOR_MIN_SCORE` | 0.60 (was 0.82) |
| `DETECTOR_HIGH_QUALITY_THRESHOLD` | 0.60 (was 0.82) |
| `FACE_QUALITY_MIN_FACE_SIZE` | 40 (was 80) |
| `FACE_QUALITY_SMALL_FACE_SIZE` | 30 (was 60) |
| `FACE_QUALITY_BLURRY_FACE_SIZE` | 60 (was 100) |
| `MATCH_CONFIDENCE_THRESHOLD` | 0.68 |
| `VALIDATOR_MIN_DETECTOR_CONFIDENCE` | 0.70 |
| `ALERT_MIN_CONFIDENCE_FLOOR` | 0.70 (was 0.72) |
| `GOVERNANCE_MAX_CANDIDATE_QUEUE_SIZE` | 32 (default) / 15 (Colab SERVER_ENV) |

## Hard stops respected
- `pipeline.py`: untouched (no structural changes)
- `track_manager.py`, `event_validator.py`, `alert_session_engine.py`: untouched
- `detector.py`, `embedder.py`, `geometry.py`, `preprocessing.py`: untouched
- ByteTrack internals: untouched
- No new infrastructure, no new packages

## Feasibility gate answers (for next session)
1. `detector_min_face_area=1000` → `detection_optimizer.py:328`, post-`scale_faces()` ✓
2. `face_quality_min_face_size` → `face_quality.py:46-58`, post-scaling, gap fixed by lowering to 40
3. ArcFace crop path → `embedder.py:67` fast path, landmark norm_crop, no bbox used
4. CLAHE → `preprocessing.py:31-54`, already in `prepared.bgr` → embedding path ✓

## Validation target (Colab GPU run)
- Sara (yellow top, foreground) → GREEN box, not absent
- `REJECTED: face_too_small` absent for det ≥ 0.5 faces
- `embeddings_generated` count increases vs prior 117
- `pose_bucket_frontal` embeddings increase vs `pose_bucket_partial`
- `alerts_per_video >= 1`
- `identity_switch_rate = 0`

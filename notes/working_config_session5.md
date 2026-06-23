# Working Config — Session 5 Baseline (2026-06-24)

## Results this config produced

| Metric | Value |
|---|---|
| Crowd footage recall | 0.40 |
| Identity switch rate | 0 / 358 frames |
| Alerts fired | 4 |
| End-to-end FPS (4K crowd) | 6.6 |
| AI pipeline FPS | 11.4 |
| Concurrent candidates handled | 16 |

Recall baseline comparison: 0.40 > 0.35 (single-scale detection literature baseline at this resolution band).

## Branch

`vsl-phase3-multi-source` — commit range leading up to `c971afc`

Key commits in this session:
- `10c9848` — fix(detector): align is_gpu with platform_bootstrap
- `9618326` — fix(preproc): raise frame width cap 640 → 1920
- `75ebd13` — chore(detector): remove [DET_OPT] diagnostic logging
- `c971afc` — chore(runtime): widen integrity range to 1920

## SERVER_ENV (Colab notebook cell)

```python
SERVER_ENV = {
    **os.environ,
    # ── Core ──────────────────────────────────────
    "PYTHONUNBUFFERED":                "1",
    "APP_ENV":                         "colab",
    "DEBUG":                           "true",
    "API_HOST":                        "0.0.0.0",
    "API_PORT":                        "8000",

    # ── InsightFace — GPU ctx on Colab ────────────
    "INSIGHTFACE_MODEL_NAME":          "buffalo_l",
    "INSIGHTFACE_CTX_ID":              "0",

    # ── Detector — RECALL BIASED ──────────────────
    "DETECTOR_PROVIDER":               "yolo",
    "DETECTOR_INPUT_WIDTH":            "640",
    "DETECTOR_INPUT_HEIGHT":           "640",
    "DETECTOR_MAX_INPUT_PIXELS":       "2073600",
    "DETECTION_CONFIDENCE_THRESHOLD":  "0.35",
    "DETECTOR_MIN_SCORE":              "0.35",
    "DETECTOR_HIGH_QUALITY_THRESHOLD": "0.55",
    "DETECTOR_MEDIUM_QUALITY_THRESHOLD":"0.45",
    "DETECTOR_SMALL_FACE_THRESHOLD":   "0.25",
    "DETECTOR_BUDGET_MS":              "200",

    # ── Face quality ──────────────────────────────
    "FACE_QUALITY_MIN_FACE_SIZE":      "52",
    "FACE_QUALITY_SMALL_FACE_SIZE":    "40",

    # ── Validator — FP filter gate ────────────────
    "VALIDATOR_STRICT_CUTOFF":         "0.55",
    "VALIDATOR_MIN_DETECTOR_CONFIDENCE":"0.50",
    "VALIDATOR_MIN_BLUR_VAR":          "30.0",
    "VALIDATOR_MIN_BRIGHTNESS":        "25.0",
    "VALIDATOR_QUALITY_CUTOFF":        "0.28",
    "VALIDATOR_MIN_ASPECT_RATIO":      "0.55",
    "VALIDATOR_MAX_ASPECT_RATIO":      "2.0",

    # ── Matching ──────────────────────────────────
    "MATCH_CONFIDENCE_THRESHOLD":      "0.45",

    # ── Alert ─────────────────────────────────────
    "ALERT_MIN_CONFIDENCE_FLOOR":      "0.52",

    # ── Relaxation ────────────────────────────────
    "RELAXATION_LOW_CONFIDENCE":       "0.40",
    "RELAXATION_LOW_CUTOFF":           "0.50",
    "RELAXATION_MEDIUM_CONFIDENCE":    "0.45",
    "RELAXATION_MEDIUM_CUTOFF":        "0.55",
    "RELAXATION_HIGH_CONFIDENCE":      "0.50",
    "RELAXATION_HIGH_CUTOFF":          "0.60",
    "ENABLE_CONFIRMATION_PROFILING":   "true",
    "CONFIRMATION_PROFILE_DUMP_PATH":  "data/logs/confirmation_profile.json",

    # ── Governance ────────────────────────────────
    "GOVERNANCE_MAX_CANDIDATE_QUEUE_SIZE":"17",
    "GOVERNANCE_HIGH_PRESSURE_INTERVAL": "8",
    "GOVERNANCE_MEDIUM_PRESSURE_INTERVAL": "5",
    "GOVERNANCE_LOW_PRESSURE_INTERVAL":  "3",
}
```

## What was fixed to reach this baseline

**Before this session:** `detection_optimizer` received `frame=(360,640)` on 4K input
because two upstream resize ops capped width at 640px:
- `video_service._resize_for_inference` (`VIDEO_INFERENCE_WIDTH=640` default)
- `preprocessing.FramePreprocessor._resize` (`PREPROCESSING_MAX_WIDTH=640` default)

Both silently downscaled before the optimizer made any sizing decision.

**Also fixed:** `detection_optimizer` used `insightface_ctx_id >= 0` to determine
GPU mode. With `INSIGHTFACE_CTX_ID` defaulting to -1 in `.env`, the optimizer
treated Colab T4 as CPU — skipping GPU adaptive sizing and the clamp-up logic
entirely. Now reads `_PLATFORM.get("backend") == "GPU"` from `platform_bootstrap`
which correctly uses `torch.cuda.is_available()`.

## Regression diff target

If recall drops below 0.35 in a future session, diff the running config against
this SERVER_ENV. Key thresholds to check first:
- `DETECTOR_MIN_SCORE` (0.35) — raising this kills small/distant face recall
- `VALIDATOR_STRICT_CUTOFF` (0.55) — raising above 0.60 starts dropping valid detections
- `MATCH_CONFIDENCE_THRESHOLD` (0.45) — this is already at ArcFace noise-floor boundary
- `VIDEO_INFERENCE_WIDTH` / `PREPROCESSING_MAX_WIDTH` — must both be ≥ 1920 on GPU

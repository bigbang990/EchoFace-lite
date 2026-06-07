# EchoFace Lite — architecture manifest
*Read this file first. Do not scan the repository.*

## Stack
- Runtime: Google Colab T4 / CUDA 12.8 / PyTorch (installed in Colab) / Python 3.10.11
- Detector: InsightFaceDetector (SCRFD via buffalo_l, CPUExecutionProvider only)
- Embedder: ArcFace (InsightFace buffalo_l, shared FaceAnalysis instance)
- Tracker: SORT (track_manager.py)
- Backend: FastAPI + uvicorn + ngrok tunnel
- Frontend: Streamlit (dashboard/)
- DB: SQLite via aiosqlite

## Hard constraints (never revisit these)
- onnxruntime-gpu has no CUDA 12.8 wheel — confirmed across 5 install attempts.
  CUDAExecutionProvider unavailable on this runtime.
- PyTorch 2.6+ sets weights_only=True — monkey-patch required before loading .pt files.
- platform_bootstrap.py exists on phase5-colab-ready branch only (not yet on main).
  It calls detect_platform() at startup — both CUDA device AND CUDAExecutionProvider
  must be present for GPU path; either failure falls back to CPU silently.
- Landmarks required: face_candidate_validator hard-rejects landmarks=None.
  Both detectors must provide 5-point kps.
- No affine alignment in codebase — landmark source does not affect embedding quality.
- Detector and embedder share one FaceAnalysis instance (loaded once per worker).

## GPU config (from platform_bootstrap.py — established, do not re-tune without data)
- backend:               "GPU"
- ctx_id:                0
- providers:             ["CUDAExecutionProvider", "CPUExecutionProvider"]
- det_size:              (640, 640)
- det_interval:          3
- conf_threshold:        0.45
- validator_cutoff:      0.55
- detector_budget_ms:    150
- max_track_survival_ms: 3000
- interval_ceiling:      8

## CPU config (from platform_bootstrap.py — established)
- backend:               "CPU"
- ctx_id:                -1
- providers:             ["CPUExecutionProvider"]
- det_size:              (320, 320)
- det_interval:          6
- conf_threshold:        0.35
- validator_cutoff:      0.40
- detector_budget_ms:    5000
- max_track_survival_ms: 6000
- interval_ceiling:      12

## Repo
github.com/bigbang990/EchoFace-lite

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
- platform_bootstrap.py does not exist in this repo (as of June 2026).
  Hardware detection is handled via settings defaults and env vars in config.py.
- bootstrap.py hardcodes CPUExecutionProvider — no runtime GPU path yet.
- Landmarks required: face_candidate_validator hard-rejects landmarks=None.
  Both detectors must provide 5-point kps.
- No affine alignment in codebase — landmark source does not affect embedding quality.
- Detector and embedder share one FaceAnalysis instance (loaded once per worker).

## Detector input config (from config.py defaults)
- detector_input_width: 320, detector_input_height: 320  (small / CPU)
- detector_medium_width: 416, detector_medium_height: 416
- detector_large_width: 512, detector_large_height: 512
- insightface_ctx_id: -1  (CPU)
- insightface_model_name: buffalo_l

## Detector thresholds (from config.py defaults)
- detector_min_score: 0.82
- detector_high_quality_threshold: 0.82
- detector_medium_quality_threshold: 0.88
- detector_small_face_threshold: 0.93
- detector_interval_frames: 8

## Repo
github.com/bigbang990/EchoFace-lite

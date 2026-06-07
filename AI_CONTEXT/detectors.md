# Detector pipeline

## Interface contract
File: ecoface_lite/ai_engine/detector.py

DetectedFace fields:
- bbox: BoundingBox  (x1, y1, x2, y2: float)
- det_score: float
- aligned_face: np.ndarray | None   — optional chip for embedding
- embedding: np.ndarray | None      — when detector+recognition run together
- landmarks: FaceLandmarks | None
- temporal_score: float | None      — blended score after temporal agreement

FaceLandmarks: points np.ndarray shape (5,2) float32
Point order: [left_eye, right_eye, nose, left_mouth, right_mouth]
Validator hard-rejects landmarks=None ("low_landmarks").
5-point landmarks required for pipeline entry.

## Provider 1: SCRFD / InsightFace (current, default)
- Class: InsightFaceDetector (ecoface_lite/ai_engine/detector.py)
- Via InsightFace buffalo_l, providers=["CPUExecutionProvider"]
- ONNX-based — CPU only on CUDA 12.8 (onnxruntime gap)
- Accuracy: ~91-92% WiderFace Hard AP
- Speed: ~8,000ms on CPU; ~100ms on GPU (when onnxruntime supports CUDA ≤12.4)
- Shares FaceAnalysis instance with embedder (loaded once per worker)
- Use when: local CPU dev, future GPU servers with CUDA ≤12.4

## Provider 2: YOLOv8-face (Phase 6 complete — detect() implemented and merged)
- File: ecoface_lite/ai_engine/detection/detectors/yolov8_detector.py
- PyTorch-native — works on any CUDA PyTorch supports
- Weights: weights/yolov8n-face.pt (gitignored)
- Download: pip install gdown && python scripts/download_yolov8_face.py
  (uses gdown; Drive ID: 1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb — derronqi model)
- Accuracy: ~88-90% WiderFace Hard AP
- Speed: 8.5ms avg, 117.9 FPS on T4 GPU (Gate D confirmed)
- Keypoint order confirmed: [0]=left_eye [1]=right_eye [2]=nose
  [3]=left_mouth [4]=right_mouth — matches FaceLandmarks convention exactly
- Use when: Colab T4, any CUDA 12.8+ environment

## Selection (wired in bootstrap.py — Phase 6 complete)
DETECTOR_PROVIDER=scrfd → SCRFD/InsightFace
DETECTOR_PROVIDER=yolo  → YOLOv8-face
Default: scrfd — platform_bootstrap.py "detector_provider" key set to "scrfd"
in both CPU and GPU branches. Env var overrides the platform dict value.

## Known issue (Phase 7)
face_app = _create_face_analysis(settings) loads unconditionally in bootstrap.py
even when DETECTOR_PROVIDER=yolo. InsightFace weights load unnecessarily.
Fix: decouple embedder construction from detector selection in Phase 7.

## Revert path
Set DETECTOR_PROVIDER=scrfd in Colab .env — no code change required.

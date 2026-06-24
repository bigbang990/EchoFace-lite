"""YOLOv8-face detector — Phase 3 implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ecoface_lite.ai_engine.detection.detectors.base_detector import (
    BaseDetector,
    DetectionConfig,
)
from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace, FaceLandmarks

logger = logging.getLogger(__name__)


class YOLOv8FaceDetector(BaseDetector):
    """YOLOv8-face detector (derronqi, 5-point landmarks).

    Keypoint order: [left_eye, right_eye, nose, left_mouth, right_mouth]
    — matches FaceLandmarks convention exactly.
    """

    def __init__(self, weights_path: Path, det_size: tuple[int, int], face_app=None) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLOv8 weights not found at {weights_path}. "
                "Run: python scripts/download_yolov8_face.py"
            )
        self._weights_path = weights_path
        self._det_size = det_size
        self._face_app = face_app

        import torch
        from ultralytics import YOLO

        # PyTorch 2.6+ sets weights_only=True by default.
        # ultralytics checkpoint contains arbitrary globals
        # (PoseModel, Sequential, etc.) — allowlist is unbounded.
        # Patch torch.load at the serialization module level
        # to force weights_only=False, restore immediately after.
        import torch.serialization as _ts
        _orig = _ts.load
        _ts.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
        torch.load = _ts.load
        try:
            self._model = YOLO(str(weights_path))
        finally:
            _ts.load = _orig
            torch.load = _orig

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        logger.info("YOLOv8FaceDetector loaded on %s", self._device)

    def detect(
        self,
        frame_bgr: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> list[DetectedFace]:
        results = self._model(frame_bgr, imgsz=self._det_size[0], verbose=False)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

        out: list[DetectedFace] = []
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            det_score = float(r.boxes.conf[i].item())

            landmarks = None
            if r.keypoints is not None:
                pts = r.keypoints.xy[i].cpu().numpy().astype("float32")
                landmarks = FaceLandmarks(points=pts)

            # Run genderage via shared InsightFace app.
            # Attribute.get() uses face.bbox (not kps) — pass bbox array directly.
            # ga[0] is np.argmax(pred[:2]): already 0=female or 1=male, not a prob.
            gender_int = None
            if self._face_app is not None:
                try:
                    gender_model = self._face_app.models.get("genderage")
                    if gender_model is not None:
                        fake_face = {
                            "bbox": np.array([x1, y1, x2, y2], dtype=np.float32)
                        }
                        ga = gender_model.get(frame_bgr, fake_face)
                        if ga is not None:
                            gender_int = int(ga[0])
                except Exception:
                    pass  # non-fatal — gate will skip if None

            out.append(
                DetectedFace(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    det_score=det_score,
                    aligned_face=None,
                    embedding=None,
                    landmarks=landmarks,
                    temporal_score=None,
                    gender=gender_int,
                )
            )
        return out

    def get_model_name(self) -> str:
        return "yolov8n-face"

    def get_input_size(self) -> tuple[int, int]:
        return self._det_size

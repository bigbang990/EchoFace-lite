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

    def __init__(self, weights_path: Path, det_size: tuple[int, int]) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLOv8 weights not found at {weights_path}. "
                "Run: python scripts/download_yolov8_face.py"
            )
        self._weights_path = weights_path
        self._det_size = det_size

        import torch
        from ultralytics import YOLO

        self._model = YOLO(str(weights_path))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        logger.info("YOLOv8FaceDetector loaded on %s", self._device)

    def detect(
        self,
        frame_bgr: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> list[DetectedFace]:
        results = self._model(frame_bgr, verbose=False)
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

            out.append(
                DetectedFace(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    det_score=det_score,
                    aligned_face=None,
                    embedding=None,
                    landmarks=landmarks,
                    temporal_score=None,
                )
            )
        return out

    def get_model_name(self) -> str:
        return "yolov8n-face"

    def get_input_size(self) -> tuple[int, int]:
        return self._det_size

"""YOLOv8-face detector stub — Phase 3 pending."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ecoface_lite.ai_engine.detection.detectors.base_detector import (
    BaseDetector,
    DetectionConfig,
)
from ecoface_lite.ai_engine.detector import DetectedFace

logger = logging.getLogger(__name__)


class YOLOv8FaceDetector(BaseDetector):
    """YOLOv8-face detector (derronqi, 5-point landmarks).

    Stub: __init__ validates weights; detect() raises NotImplementedError.
    Real inference implemented in Phase 3.
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
        logger.info("YOLOv8FaceDetector stub loaded — Phase 3 pending")

    def detect(
        self,
        frame_bgr: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> list[DetectedFace]:
        raise NotImplementedError("YOLOv8FaceDetector.detect() — Phase 3 pending")

    def get_model_name(self) -> str:
        return "yolov8n-face"

    def get_input_size(self) -> tuple[int, int]:
        return self._det_size

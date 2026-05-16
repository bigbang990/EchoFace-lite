from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.ai_engine.geometry import compute_face_geometry
from ecoface_lite.core.config import Settings


@dataclass(frozen=True)
class FaceQualityResult:
    accepted: bool
    blur_score: float
    reason: str | None = None


class FaceQualityAssessor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def assess(self, frame_bgr: np.ndarray, face: DetectedFace) -> FaceQualityResult:
        if face.det_score < self._settings.detection_confidence_threshold:
            return FaceQualityResult(False, 0.0, "low_detection_confidence")
        geometry = compute_face_geometry(face, frame_bgr.shape)
        x1, y1, x2, y2 = geometry.x1, geometry.y1, geometry.x2, geometry.y2
        width = geometry.width
        height = geometry.height
        if width < self._settings.face_quality_min_face_size or height < self._settings.face_quality_min_face_size:
            return FaceQualityResult(False, 0.0, "face_too_small")
        skew = max(width / max(height, 1), height / max(width, 1))
        if skew > self._settings.face_quality_max_aspect_ratio_skew:
            return FaceQualityResult(False, 0.0, "poor_face_angle")
        crop = frame_bgr[y1:y2, x1:x2]
        blur_score = self.blur_score(crop)
        if blur_score < self._settings.face_quality_min_blur_score:
            return FaceQualityResult(False, blur_score, "blurry_face")
        return FaceQualityResult(True, blur_score)

    def blur_score(self, face_bgr: np.ndarray) -> float:
        if face_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())


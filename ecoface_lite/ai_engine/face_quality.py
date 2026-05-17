from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.ai_engine.geometry import compute_face_geometry
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


@dataclass(frozen=True)
class FaceQualityResult:
    accepted: bool
    blur_score: float
    brightness_score: float = 0.0
    contrast_score: float = 0.0
    quality_score: float = 0.0
    reason: str | None = None


class FaceQualityAssessor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def assess(self, frame_bgr: np.ndarray, face: DetectedFace) -> FaceQualityResult:
        if face.det_score < self._settings.detection_confidence_threshold:
            return FaceQualityResult(False, 0.0, reason="low_detection_confidence")
        geometry = compute_face_geometry(face, frame_bgr.shape)
        x1, y1, x2, y2 = geometry.x1, geometry.y1, geometry.x2, geometry.y2
        width = geometry.width
        height = geometry.height
        crop = frame_bgr[y1:y2, x1:x2]
        blur_score = self.blur_score(crop)
        brightness_score = self.brightness_score(crop)
        contrast_score = self.contrast_score(crop)
        min_size = self._settings.face_quality_min_face_size
        if blur_score >= self._settings.face_quality_min_blur_score * 1.5:
            min_size = self._settings.face_quality_small_face_size
        elif blur_score < self._settings.face_quality_min_blur_score:
            min_size = self._settings.face_quality_blurry_face_size
        quality_score = self.quality_score(blur_score, brightness_score, contrast_score, width, height)
        metrics.observe("face_blur_score", blur_score)
        metrics.observe("face_brightness_score", brightness_score)
        metrics.observe("face_contrast_score", contrast_score)
        metrics.observe("face_quality_score", quality_score)
        if width < min_size or height < min_size:
            return FaceQualityResult(False, blur_score, brightness_score, contrast_score, quality_score, "face_too_small")
        skew = max(width / max(height, 1), height / max(width, 1))
        if skew > self._settings.face_quality_max_pose_skew:
            return FaceQualityResult(False, blur_score, brightness_score, contrast_score, quality_score, "extreme_face_pose")
        if skew > self._settings.face_quality_max_aspect_ratio_skew:
            return FaceQualityResult(False, blur_score, brightness_score, contrast_score, quality_score, "poor_face_angle")
        if brightness_score < self._settings.face_quality_min_brightness:
            return FaceQualityResult(False, blur_score, brightness_score, contrast_score, quality_score, "poor_brightness")
        if contrast_score < self._settings.face_quality_min_contrast:
            return FaceQualityResult(False, blur_score, brightness_score, contrast_score, quality_score, "poor_contrast")
        if blur_score < self._settings.face_quality_min_blur_score:
            return FaceQualityResult(False, blur_score, brightness_score, contrast_score, quality_score, "blurry_face")
        return FaceQualityResult(True, blur_score, brightness_score, contrast_score, quality_score)

    def blur_score(self, face_bgr: np.ndarray) -> float:
        if face_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def brightness_score(self, face_bgr: np.ndarray) -> float:
        if face_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def contrast_score(self, face_bgr: np.ndarray) -> float:
        if face_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    def quality_score(self, blur_score: float, brightness_score: float, contrast_score: float, width: int, height: int) -> float:
        blur_component = min(1.0, blur_score / max(self._settings.face_quality_min_blur_score * 2.0, 1.0))
        brightness_component = min(1.0, brightness_score / max(self._settings.face_quality_min_brightness * 2.0, 1.0))
        contrast_component = min(1.0, contrast_score / max(self._settings.face_quality_min_contrast * 2.0, 1.0))
        frontalness_component = min(width / max(height, 1), height / max(width, 1))
        return float((0.4 * blur_component) + (0.2 * brightness_component) + (0.2 * contrast_component) + (0.2 * frontalness_component))


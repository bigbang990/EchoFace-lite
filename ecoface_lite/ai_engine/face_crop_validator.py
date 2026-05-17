"""Sanity checks on face crops before embedding generation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.ai_engine.geometry import compute_face_geometry
from ecoface_lite.core.config import Settings


@dataclass(frozen=True)
class CropValidationResult:
    accepted: bool
    reason: str | None = None


class FaceCropValidator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def validate(self, frame_bgr: np.ndarray, face: DetectedFace) -> CropValidationResult:
        geometry = compute_face_geometry(face, frame_bgr.shape)
        crop = frame_bgr[geometry.y1 : geometry.y2, geometry.x1 : geometry.x2]
        if crop.size == 0:
            return CropValidationResult(False, "empty_crop")

        if face.landmarks is not None:
            lm_result = self._validate_landmark_crop(face, geometry.height)
            if not lm_result.accepted:
                return lm_result

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        upper = gray[: max(1, int(h * 0.35)), :]
        lower = gray[max(0, int(h * 0.65)) :, :]
        if upper.size and lower.size:
            upper_mean = float(np.mean(upper))
            lower_mean = float(np.mean(lower))
            if upper_mean > lower_mean * 1.35 and upper_mean - lower_mean > 18:
                return CropValidationResult(False, "oversized_forehead")
            if lower_mean < upper_mean * 0.55:
                return CropValidationResult(False, "missing_chin")

        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur < self._settings.face_quality_min_blur_score * 0.35:
            return CropValidationResult(False, "extreme_crop_blur")

        aspect = geometry.width / max(geometry.height, 1)
        if aspect < self._settings.detector_min_aspect_ratio * 0.9 or aspect > self._settings.detector_max_aspect_ratio * 1.1:
            return CropValidationResult(False, "crop_distortion")

        return CropValidationResult(True)

    def _validate_landmark_crop(self, face: DetectedFace, box_height: int) -> CropValidationResult:
        pts = face.landmarks.points  # type: ignore[union-attr]
        left_eye, right_eye, nose = pts[0], pts[1], pts[2]
        y1, y2 = face.bbox.y1, face.bbox.y2
        box_h = max(1e-6, y2 - y1)
        eye_y = (left_eye[1] + right_eye[1]) / 2.0
        nose_y = nose[1]
        eye_band = (eye_y - y1) / box_h
        chin_band = (y2 - nose_y) / box_h
        forehead_band = (eye_y - y1) / box_h

        if eye_band < self._settings.face_crop_min_eye_band_ratio:
            return CropValidationResult(False, "missing_eyes")
        if forehead_band > self._settings.face_crop_max_forehead_ratio:
            return CropValidationResult(False, "oversized_forehead")
        if chin_band < self._settings.face_crop_min_chin_ratio:
            return CropValidationResult(False, "missing_chin")
        return CropValidationResult(True)

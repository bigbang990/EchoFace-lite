from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.ai_engine.geometry import compute_face_geometry, scale_face_to_original
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


@dataclass(frozen=True)
class DetectionFilterDecision:
    accepted: bool
    reason: str | None = None


class DetectionOptimizer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._last_detection_frame: int | None = None
        self._last_faces: list[DetectedFace] = []

    def should_detect(self, frame_index: int) -> bool:
        interval = max(1, self._settings.detector_interval_frames)
        if self._last_detection_frame is None:
            return True
        due = frame_index - self._last_detection_frame >= interval
        if due:
            metrics.increment("tracker_refresh_count")
        return due

    def prepare_for_detection(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        target_width = self._settings.detector_input_width
        if target_width <= 0 or frame_bgr.shape[1] <= target_width:
            return frame_bgr, 1.0
        scale = target_width / frame_bgr.shape[1]
        height = max(1, int(frame_bgr.shape[0] * scale))
        resized = cv2.resize(frame_bgr, (target_width, height), interpolation=cv2.INTER_AREA)
        return resized, scale

    def scale_faces(self, faces: list[DetectedFace], scale: float) -> list[DetectedFace]:
        if scale == 1.0:
            return faces
        inv = 1.0 / scale
        return [scale_face_to_original(face, scale) for face in faces]

    def filter_faces(self, faces: list[DetectedFace], frame_shape: tuple[int, ...]) -> tuple[list[DetectedFace], list[tuple[DetectedFace, str]]]:
        accepted: list[DetectedFace] = []
        rejected: list[tuple[DetectedFace, str]] = []
        for face in faces:
            decision = self.evaluate(face, frame_shape)
            if decision.accepted:
                accepted.append(face)
            else:
                rejected.append((face, decision.reason or "detector_filter_rejected"))
                metrics.increment(decision.reason or "detector_filter_rejected")
        self._last_faces = accepted
        return accepted, rejected

    def observe_detection_cycle(self, frame_index: int, raw_count: int, accepted_count: int, rejected_count: int) -> None:
        self._last_detection_frame = frame_index
        metrics.increment("detection_cycles")
        metrics.increment("detector_early_rejections", rejected_count)
        if raw_count:
            metrics.observe("detector_rejection_rate", rejected_count / raw_count)
            metrics.observe("face_visibility_ratio", accepted_count / raw_count)
        if raw_count > 0 and accepted_count == 0:
            metrics.increment("detector_missed_face_estimate")
        if raw_count >= self._settings.detector_overload_face_count:
            metrics.increment("detector_overload_warnings")

    def observe_tracking_cycle(self) -> None:
        metrics.increment("tracking_cycles")
        metrics.observe("tracker_reuse_rate", 1.0)

    def evaluate(self, face: DetectedFace, frame_shape: tuple[int, ...]) -> DetectionFilterDecision:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        geometry = compute_face_geometry(face, frame_shape)
        area = geometry.area
        metrics.observe("avg_face_size", area)
        metrics.observe("avg_detection_confidence", face.det_score)
        if face.det_score < self._settings.detector_min_score:
            return DetectionFilterDecision(False, "weak_detector_score")
        min_width, min_height, min_area = self._adaptive_thresholds(width, height)
        if geometry.width < min_width or geometry.height < min_height:
            return DetectionFilterDecision(False, "detector_face_too_small")
        if area < min_area:
            return DetectionFilterDecision(False, "detector_face_area_too_small")
        if geometry.aspect_ratio > self._settings.detector_max_aspect_ratio:
            return DetectionFilterDecision(False, "detector_bad_aspect_ratio")
        margin = self._settings.detector_edge_margin_ratio
        if geometry.x1 <= width * margin or geometry.y1 <= height * margin or geometry.x2 >= width * (1.0 - margin) or geometry.y2 >= height * (1.0 - margin):
            return DetectionFilterDecision(False, "detector_edge_face")
        if self._settings.detector_center_priority_enabled:
            dx = abs(geometry.center_x - width / 2.0) / max(width / 2.0, 1.0)
            dy = abs(geometry.center_y - height / 2.0) / max(height / 2.0, 1.0)
            if (dx * dx + dy * dy) ** 0.5 > self._settings.detector_center_max_distance:
                return DetectionFilterDecision(False, "detector_low_center_priority")
        return DetectionFilterDecision(True)

    def _adaptive_thresholds(self, width: int, height: int) -> tuple[int, int, int]:
        reference_width = 640.0
        scale = min(1.0, max(width, height) / reference_width)
        min_width = max(18, int(round(self._settings.detector_min_face_width * scale)))
        min_height = max(18, int(round(self._settings.detector_min_face_height * scale)))
        min_area = max(500, int(round(self._settings.detector_min_face_area * scale * scale)))
        return min_width, min_height, min_area

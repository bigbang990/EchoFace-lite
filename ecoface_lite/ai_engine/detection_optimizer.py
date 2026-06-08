from __future__ import annotations

from dataclasses import dataclass

import math
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
        self.emergency_mode: bool = False

    def active_track_count(self) -> int:
        return len(self._last_faces)

    def should_detect(
        self,
        frame_index: int,
        *,
        active_tracks: int = 0,
        stable_tracks: int = 0,
        avg_motion_stability: float = 0.0,
        detector_interval_override: int | None = None,
    ) -> bool:
        interval = detector_interval_override or self._adaptive_interval(active_tracks, stable_tracks, avg_motion_stability)
        metrics.observe("detector_adaptive_interval", float(interval))
        if frame_index % interval != 0:
            return False
        if self._last_detection_frame != frame_index:
            metrics.increment("tracker_refresh_count")
        return True

    def _adaptive_interval(
        self,
        active_tracks: int,
        stable_tracks: int,
        avg_motion_stability: float,
    ) -> int:
        base = max(1, self._settings.detector_interval_frames)
        min_iv = max(1, self._settings.detector_interval_min_frames)
        max_iv = max(min_iv, self._settings.detector_interval_max_frames)
        if active_tracks == 0 and not self.emergency_mode:
            return min_iv
        if self.emergency_mode:
            return min_iv # Force minimum interval during emergency
        if avg_motion_stability < self._settings.motion_high_threshold:
            return max(min_iv, self._settings.detector_interval_motion_frames)
        if stable_tracks >= max(1, active_tracks // 2) and active_tracks > 0:
            return min(max_iv, self._settings.detector_interval_stable_frames)
        return min(max_iv, base)

    def prepare_for_detection(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        enhanced = self._enhance_detector_input(frame_bgr)
        target_width, target_height = self._select_detector_size(enhanced.shape)
        
        # Calculate initial target dimensions preserving aspect ratio
        if enhanced.shape[1] > 0:
            initial_scale = target_width / enhanced.shape[1]
            initial_height = max(1, int(enhanced.shape[0] * initial_scale))
        else:
            initial_height = target_height
            
        requested_pixels = target_width * initial_height
        metrics.observe("adaptive_resolution_requested", requested_pixels)
        
        # Apply experiment cap with safe bounds if enabled
        final_width, final_height = target_width, initial_height
        if self._settings.detector_resolution_cap_enabled:
            min_pixels = self._settings.detector_min_input_pixels
            max_pixels = self._settings.detector_max_input_pixels
            is_gpu = self._settings.insightface_ctx_id >= 0

            # Clamp target pixels within safe bounds.
            # Phase 2E: On CPU, NEVER clamp up — upscaling on CPU adds compute
            # with no benefit since the model was prepared at cpu_detector_resolution.
            if requested_pixels > max_pixels:
                final_width, final_height = self._compute_scaled_dims(
                    target_width, initial_height, max_pixels
                )
                metrics.increment("resolution_clamped_down_count")
            elif requested_pixels < min_pixels and is_gpu:
                # GPU only: clamp up to ensure minimum anchor coverage
                final_width, final_height = self._compute_scaled_dims(
                    target_width, initial_height, min_pixels
                )
                metrics.increment("resolution_clamped_up_count")
            elif requested_pixels < min_pixels and not is_gpu:
                # CPU: accept natural resolution, no upscale
                metrics.increment("resolution_cpu_natural_count")
            else:
                metrics.increment("resolution_within_safe_band_count")

        final_pixels = final_width * final_height
        metrics.observe("adaptive_resolution_final", final_pixels)
        metrics.observe("detector_input_resolution", final_pixels)
        metrics.observe("detector_resolution", final_pixels)
        metrics.observe("capped_detector_resolution", final_pixels)
        metrics.observe("original_detector_resolution", requested_pixels)
        
        # Categorize resolution band
        if final_pixels <= self._settings.detector_min_input_pixels:
            metrics.observe("detector_resolution_band", 0.0) # LOW
        elif final_pixels >= self._settings.detector_max_input_pixels:
            metrics.observe("detector_resolution_band", 2.0) # HIGH
        else:
            metrics.observe("detector_resolution_band", 1.0) # MID

        if final_width <= 0 or enhanced.shape[1] <= final_width:
            return enhanced, 1.0
            
        scale = final_width / enhanced.shape[1]
        resized = cv2.resize(enhanced, (final_width, final_height), interpolation=cv2.INTER_AREA)
        metrics.observe("coordinate_scale_factor", scale)
        return resized, scale

    def _compute_scaled_dims(self, width: int, height: int, target_pixels: int) -> tuple[int, int]:
        current_pixels = width * height
        if current_pixels == target_pixels:
            return width, height
        scale = math.sqrt(target_pixels / current_pixels)
        return int(width * scale), int(height * scale)

    def _enhance_detector_input(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self._settings.detector_input_enable_enhancement:
            return frame_bgr
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        merged = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=0.8)
        sharpened = cv2.addWeighted(enhanced, 1.25, blurred, -0.25, 0)
        gamma = 1.1
        inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(sharpened, table)

    def scale_faces(self, faces: list[DetectedFace], scale: float) -> list[DetectedFace]:
        if scale == 1.0:
            return faces
        metrics.increment("coordinate_scaling_validations", len(faces))
        return [scale_face_to_original(face, scale) for face in faces]

    def filter_faces(self, faces: list[DetectedFace], frame_shape: tuple[int, ...], emergency_mode: bool = False) -> tuple[list[DetectedFace], list[tuple[DetectedFace, str]]]:
        accepted: list[DetectedFace] = []
        rejected: list[tuple[DetectedFace, str]] = []
        for face in faces:
            decision = self.evaluate(face, frame_shape, emergency_mode=emergency_mode)
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
            metrics.observe("recall_per_resolution", accepted_count / raw_count)
        if raw_count > 0 and accepted_count == 0:
            metrics.increment("detector_missed_face_estimate")
        if raw_count >= self._settings.detector_overload_face_count:
            metrics.increment("detector_overload_warnings")

    def observe_tracking_cycle(self) -> None:
        metrics.increment("tracking_cycles")
        metrics.increment("tracking_only_cycles")
        metrics.observe("tracker_reuse_rate", 1.0)

    def evaluate(self, face: DetectedFace, frame_shape: tuple[int, ...], emergency_mode: bool = False) -> DetectionFilterDecision:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        geometry = compute_face_geometry(face, frame_shape)
        area = geometry.area
        metrics.observe("avg_face_size", area)
        metrics.observe("avg_detection_confidence", face.det_score)
        score = face.temporal_score if face.temporal_score is not None else face.det_score
        frame_area = max(1, width * height)
        area_ratio = area / frame_area
        
        if emergency_mode:
            min_score = 0.25 # Drastic reduction during emergency
        elif area_ratio < self._settings.detector_small_face_area_ratio:
            min_score = self._settings.detector_small_face_threshold
        elif area_ratio < self._settings.detector_medium_face_area_ratio:
            min_score = self._settings.detector_medium_quality_threshold
        else:
            min_score = max(self._settings.detector_min_score, self._settings.detector_high_quality_threshold)
        
        if score < min_score:
            return DetectionFilterDecision(False, "weak_detector_score")
            
        min_width, min_height, min_area = self._adaptive_thresholds(width, height)
        
        if emergency_mode:
            min_width = 16 # Absolute minimum
            min_height = 16
            min_area = 256
        
        dynamic_min_area = max(min_area, int(frame_area * self._settings.detector_min_face_area_ratio))
        if emergency_mode:
            dynamic_min_area = min_area
        if geometry.width < min_width or geometry.height < min_height:
            return DetectionFilterDecision(False, "detector_face_too_small")
        if area < dynamic_min_area:
            return DetectionFilterDecision(False, "detector_face_area_too_small")
        width_height_ratio = geometry.width / max(geometry.height, 1)
        if (
            width_height_ratio < self._settings.detector_min_aspect_ratio
            or width_height_ratio > self._settings.detector_max_aspect_ratio
        ):
            return DetectionFilterDecision(False, "detector_bad_aspect_ratio")
        if self._outside_frame_ratio(face, width, height) > 0.35:
            return DetectionFilterDecision(False, "detector_edge_face")
        margin = self._settings.detector_edge_margin_ratio
        if geometry.x1 <= width * margin or geometry.y1 <= height * margin or geometry.x2 >= width * (1.0 - margin) or geometry.y2 >= height * (1.0 - margin):
            return DetectionFilterDecision(False, "detector_edge_face")
        if self._settings.detector_center_priority_enabled:
            dx = abs(geometry.center_x - width / 2.0) / max(width / 2.0, 1.0)
            dy = abs(geometry.center_y - height / 2.0) / max(height / 2.0, 1.0)
            if (dx * dx + dy * dy) ** 0.5 > self._settings.detector_center_max_distance:
                return DetectionFilterDecision(False, "detector_low_center_priority")
        return DetectionFilterDecision(True)

    def _select_detector_size(self, frame_shape: tuple[int, ...]) -> tuple[int, int]:
        # Phase 2E: On CPU, never upgrade to medium/large resolution.
        # 416px or 512px at 4+/8+ tracks costs 1200–2000ms on CPU per cycle.
        # Fixed 320px ceiling keeps each cycle at 400–800ms.
        is_gpu = self._settings.insightface_ctx_id >= 0
        if not is_gpu:
            cpu_res = self._settings.cpu_detector_resolution
            metrics.observe("track_count", self.active_track_count())
            return (cpu_res, cpu_res)

        # GPU: existing adaptive logic — upgrade resolution for crowd/density
        active_tracks = self.active_track_count()
        occupancy = self._occupancy_ratio(frame_shape)
        metrics.observe("track_count", active_tracks)
        metrics.observe("frame_occupancy_ratio", occupancy)
        if active_tracks >= self._settings.detector_high_track_count or occupancy >= self._settings.detector_high_occupancy_ratio:
            size = (self._settings.detector_large_width, self._settings.detector_large_height)
        elif active_tracks >= self._settings.detector_medium_track_count or self._last_faces:
            size = (self._settings.detector_medium_width, self._settings.detector_medium_height)
        else:
            size = (self._settings.detector_input_width, self._settings.detector_input_height)
        gpu_res = self._settings.gpu_detector_resolution
        return (min(size[0], gpu_res), min(size[1], gpu_res))

    def _occupancy_ratio(self, frame_shape: tuple[int, ...]) -> float:
        frame_area = max(1, int(frame_shape[0]) * int(frame_shape[1]))
        face_area = 0
        for face in self._last_faces:
            geometry = compute_face_geometry(face, frame_shape)
            face_area += geometry.area
        return min(1.0, face_area / frame_area)

    def _outside_frame_ratio(self, face: DetectedFace, frame_width: int, frame_height: int) -> float:
        raw_width = max(1e-6, face.bbox.x2 - face.bbox.x1)
        raw_height = max(1e-6, face.bbox.y2 - face.bbox.y1)
        raw_area = raw_width * raw_height
        clipped_x1 = max(0.0, min(float(frame_width), face.bbox.x1))
        clipped_y1 = max(0.0, min(float(frame_height), face.bbox.y1))
        clipped_x2 = max(0.0, min(float(frame_width), face.bbox.x2))
        clipped_y2 = max(0.0, min(float(frame_height), face.bbox.y2))
        clipped_area = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)
        return max(0.0, min(1.0, 1.0 - (clipped_area / raw_area)))

    def _adaptive_thresholds(self, width: int, height: int) -> tuple[int, int, int]:
        reference_width = 640.0
        scale = min(1.0, max(width, height) / reference_width)
        min_width = max(18, int(round(self._settings.detector_min_face_width * scale)))
        min_height = max(18, int(round(self._settings.detector_min_face_height * scale)))
        min_area = max(500, int(round(self._settings.detector_min_face_area * scale * scale)))
        return min_width, min_height, min_area

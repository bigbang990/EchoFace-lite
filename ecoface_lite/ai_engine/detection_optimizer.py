from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import math
import time
import cv2
import numpy as np

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.ai_engine.geometry import compute_face_geometry, scale_face_to_original
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


@dataclass
class TimingBlock:
    """Unified timing block using time.perf_counter() exclusively (for consistency)."""
    start: float = 0.0
    end: float = 0.0

    def begin(self):
        self.start = time.perf_counter()

    def stop(self):
        self.end = time.perf_counter()

    @property
    def elapsed_ms(self):
        return (self.end - self.start) * 1000.0


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
        
        # Phase 2D: Detection Acquisition Hardening State
        self.zero_face_streaks: int = 0
        self.consecutive_weak_detections: int = 0
        self.validator_strict_pass_count: int = 0
        self.current_resolution_level: int = 0  # 0=160, 1=320, 2=480
        self.starvation_start_frame: int | None = None
        self.starvation_duration: int = 0
        self.recovery_resolution_used: int = 0
        self.no_face_detected_streak: int = 0
        self.current_cadence_level: int = 0  # 0=8, 1=4, 2=2
        
        # Phase 2D.2: Detection Acquisition Efficiency Audit State
        # Objective 1: Detector Acquisition Substage Isolation
        self.roi_crop_ms: float = 0.0
        self.roi_upscale_ms: float = 0.0
        self.clahe_ms: float = 0.0
        self.detector_inference_ms: float = 0.0
        self.validator_ms: float = 0.0
        self.small_face_rescue_ms: float = 0.0
        self.resolution_escalation_ms: float = 0.0
        self.acquisition_total_ms: float = 0.0
        
        # Objective 2: Resolution Escalation Visibility
        self.number_of_resolution_escalations: int = 0
        self.maximum_resolution_reached: int = 160
        self.resolution_levels_attempted: list[int] = []
        self.resolution_retry_count: int = 0
        self.resolution_escalation_trigger_reason: str = ""
        
        # Objective 3: Small Face Rescue Accounting
        self.small_face_candidates_detected: int = 0
        self.small_face_rescue_attempts: int = 0
        self.small_face_rescue_successes: int = 0
        self.small_face_rescue_failures: int = 0
        self.average_small_face_area: float = 0.0
        self.small_face_rescue_total_ms: float = 0.0
        
        # Objective 4: CLAHE Activation Attribution
        self.clahe_activation_count: int = 0
        self.clahe_skipped_count: int = 0
        self.clahe_average_ms: float = 0.0
        self.clahe_trigger_luminance: float = 0.0
        
        # Objective 5: Detector Invocation Cause Attribution
        self.detector_invocation_cause: str = "cadence_refresh"
        
        # Objective 7: Rolling Acquisition Analytics
        self.detector_inference_history = deque(maxlen=60)
        self.clahe_history = deque(maxlen=60)
        self.rescue_history = deque(maxlen=60)
        self.resolution_escalation_history = deque(maxlen=60)
        self.acquisition_total_history = deque(maxlen=60)
        
        # Phase 2D.3: Detector Throughput Stabilization State
        self.suppress_escalation_frames_remaining: int = 0
        self.stable_tracking_frames_count: int = 0
        self.runtime_budget_violations: int = 0
        self.escalation_suppression_active: bool = False
        self.resolution_decay_events: int = 0
        self.suppressed_small_face_rescues: int = 0
        self.suppressed_resolution_escalations: int = 0

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
        # Phase 2D: Detector Starvation Cadence Steps (8→4→2)
        cadence_intervals = [8,4,2]
        
        # Check starvation trigger
        if active_tracks >0 and self.no_face_detected_streak >=2:
            self.current_cadence_level = min(self.current_cadence_level +1, 2)
            # Start starvation metrics
            if self.starvation_start_frame is None:
                self.starvation_start_frame = self._last_detection_frame or 0
        
        # Check recovery
        if self.no_face_detected_streak ==0 and self.current_cadence_level >0:
            self.current_cadence_level =0
            if self.starvation_start_frame is not None:
                # Record duration and resolution used
                self.starvation_duration = (self._last_detection_frame or 0) - self.starvation_start_frame
                self.recovery_resolution_used = self.current_resolution_level
                metrics.observe("phase2d_starvation_duration", float(self.starvation_duration))
                metrics.observe("phase2d_recovery_resolution", float(self.recovery_resolution_used))
                self.starvation_start_frame = None
        
        metrics.observe("phase2d_cadence_level", float(self.current_cadence_level))
        return cadence_intervals[self.current_cadence_level]

    def prepare_for_detection(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        # Phase 2D.2: Initialize acquisition timing blocks
        acquisition_timer = TimingBlock()
        resolution_escalation_timer = TimingBlock()
        clahe_timer = TimingBlock()
        
        acquisition_timer.begin()
        
        resolution_escalation_timer.begin()
        enhanced = self._enhance_detector_input(frame_bgr, clahe_timer)
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
            
            # Clamp target pixels within safe bounds
            if requested_pixels > max_pixels:
                final_width, final_height = self._compute_scaled_dims(
                    target_width, initial_height, max_pixels
                )
                metrics.increment("resolution_clamped_down_count")
            elif requested_pixels < min_pixels:
                final_width, final_height = self._compute_scaled_dims(
                    target_width, initial_height, min_pixels
                )
                metrics.increment("resolution_clamped_up_count")
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
        resolution_escalation_timer.stop()
        self.resolution_escalation_ms = resolution_escalation_timer.elapsed_ms
        
        # Update resolution escalation visibility telemetry
        resolution_sizes = [160, 320, 480]
        self.maximum_resolution_reached = resolution_sizes[self.current_resolution_level]
        metrics.observe("phase2d2_max_resolution_reached", float(self.maximum_resolution_reached))
        metrics.observe("phase2d2_resolution_escalation_ms", self.resolution_escalation_ms)
        metrics.observe("phase2d2_clahe_ms", self.clahe_ms)
        
        if final_width <= 0 or enhanced.shape[1] <= final_width:
            acquisition_timer.stop()
            self.acquisition_total_ms = acquisition_timer.elapsed_ms
            metrics.observe("phase2d2_acquisition_total_ms", self.acquisition_total_ms)
            return enhanced, 1.0
            
        scale = final_width / enhanced.shape[1]
        resized = cv2.resize(enhanced, (final_width, final_height), interpolation=cv2.INTER_AREA)
        metrics.observe("coordinate_scale_factor", scale)
        
        acquisition_timer.stop()
        self.acquisition_total_ms = acquisition_timer.elapsed_ms
        metrics.observe("phase2d2_acquisition_total_ms", self.acquisition_total_ms)
        
        # Update rolling histories
        self.clahe_history.append(self.clahe_ms)
        self.resolution_escalation_history.append(self.resolution_escalation_ms)
        self.acquisition_total_history.append(self.acquisition_total_ms)
        
        return resized, scale

    def _compute_scaled_dims(self, width: int, height: int, target_pixels: int) -> tuple[int, int]:
        current_pixels = width * height
        if current_pixels == target_pixels:
            return width, height
        scale = math.sqrt(target_pixels / current_pixels)
        return int(width * scale), int(height * scale)

    def _enhance_detector_input(self, frame_bgr: np.ndarray, clahe_timer: TimingBlock) -> np.ndarray:
        if not self._settings.detector_input_enable_enhancement:
            self.clahe_skipped_count +=1
            metrics.observe("phase2d2_clahe_skipped_count", float(self.clahe_skipped_count))
            return frame_bgr
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe_timer.begin()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        clahe_timer.stop()
        self.clahe_ms = clahe_timer.elapsed_ms
        self.clahe_activation_count +=1
        self.clahe_trigger_luminance = float(np.mean(l_channel))
        metrics.observe("phase2d2_clahe_activation_count", float(self.clahe_activation_count))
        metrics.observe("phase2d2_clahe_trigger_luminance", self.clahe_trigger_luminance)
        
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
        
        # Phase 2D: Update state variables
        if accepted_count == 0:
            self.zero_face_streaks +=1
            self.no_face_detected_streak +=1
            self.validator_strict_pass_count =0
            self.stable_tracking_frames_count = max(0, self.stable_tracking_frames_count -1)
        else:
            self.zero_face_streaks =0
            self.no_face_detected_streak =0
            self.validator_strict_pass_count +=1
            self.stable_tracking_frames_count +=1
            
        # Check consecutive weak detections
        if raw_count >0:
            # Check if any accepted detections have confidence <0.45
            weak_count = sum(1 for face in self._last_faces if face.det_score <0.45)
            if weak_count >0 or (accepted_count ==0 and raw_count>0):
                self.consecutive_weak_detections +=1
            else:
                self.consecutive_weak_detections =0
        
        if raw_count:
            metrics.observe("detector_rejection_rate", rejected_count / raw_count)
            metrics.observe("face_visibility_ratio", accepted_count / raw_count)
            metrics.observe("recall_per_resolution", accepted_count / raw_count)
            metrics.observe("phase2d_zero_face_streaks", float(self.zero_face_streaks))
            metrics.observe("phase2d_consecutive_weak_detections", float(self.consecutive_weak_detections))
        if raw_count > 0 and accepted_count == 0:
            metrics.increment("detector_missed_face_estimate")
        if raw_count >= self._settings.detector_overload_face_count:
            metrics.increment("detector_overload_warnings")
            
        # Phase 2D.3: Check runtime budget and apply escalation cooldown
        if self.detector_inference_ms > self._settings.phase2d3_max_detector_runtime_ms:
            self.runtime_budget_violations +=1
            self.suppress_escalation_frames_remaining = self._settings.phase2d3_suppress_escalation_frames
            self.escalation_suppression_active = True
            metrics.increment("phase2d3_runtime_budget_violations")
            metrics.observe("phase2d3_escalation_suppression_active", 1.0)
        elif self.suppress_escalation_frames_remaining >0:
            self.suppress_escalation_frames_remaining -=1
            if self.suppress_escalation_frames_remaining ==0:
                self.escalation_suppression_active = False
                metrics.observe("phase2d3_escalation_suppression_active", 0.0)
                
        # Phase 2D.3: Resolution decay recovery (gradually return to baseline)
        if self.stable_tracking_frames_count >= self._settings.phase2d3_resolution_decay_frames and self.current_resolution_level >0:
            self.current_resolution_level -=1
            self.resolution_decay_events +=1
            self.stable_tracking_frames_count =0
            metrics.increment("phase2d3_resolution_decay_events")
            
        metrics.observe("phase2d3_stable_tracking_frames", float(self.stable_tracking_frames_count))

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
        
        # Phase 2D.2: Small Face Rescue Accounting
        if area < 1200:  # <1200px area as per objective 3
            if self.escalation_suppression_active:
                self.suppressed_small_face_rescues +=1
                metrics.observe("phase2d3_suppressed_small_face_rescues", float(self.suppressed_small_face_rescues))
            else:
                self.small_face_candidates_detected +=1
                self.small_face_rescue_attempts +=1
                metrics.observe("phase2d2_small_face_candidates_detected", float(self.small_face_candidates_detected))
                metrics.observe("phase2d2_small_face_rescue_attempts", float(self.small_face_rescue_attempts))
        
        if emergency_mode:
            min_score = 0.25 # Drastic reduction during emergency
        elif area_ratio < self._settings.detector_small_face_area_ratio:
            min_score = self._settings.detector_small_face_threshold
        elif area_ratio < self._settings.detector_medium_face_area_ratio:
            min_score = self._settings.detector_medium_quality_threshold
        else:
            min_score = max(self._settings.detector_min_score, self._settings.detector_high_quality_threshold)
        
        if score < min_score:
            if area <1200:
                self.small_face_rescue_failures +=1
                metrics.observe("phase2d2_small_face_rescue_failures", float(self.small_face_rescue_failures))
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
            if area <1200:
                self.small_face_rescue_failures +=1
                metrics.observe("phase2d2_small_face_rescue_failures", float(self.small_face_rescue_failures))
            return DetectionFilterDecision(False, "detector_face_too_small")
        if area < dynamic_min_area:
            if area <1200:
                self.small_face_rescue_failures +=1
                metrics.observe("phase2d2_small_face_rescue_failures", float(self.small_face_rescue_failures))
            return DetectionFilterDecision(False, "detector_face_area_too_small")
        width_height_ratio = geometry.width / max(geometry.height, 1)
        if (
            width_height_ratio < self._settings.detector_min_aspect_ratio
            or width_height_ratio > self._settings.detector_max_aspect_ratio
        ):
            if area <1200:
                self.small_face_rescue_failures +=1
                metrics.observe("phase2d2_small_face_rescue_failures", float(self.small_face_rescue_failures))
            return DetectionFilterDecision(False, "detector_bad_aspect_ratio")
        if self._outside_frame_ratio(face, width, height) > 0.35:
            if area <1200:
                self.small_face_rescue_failures +=1
                metrics.observe("phase2d2_small_face_rescue_failures", float(self.small_face_rescue_failures))
            return DetectionFilterDecision(False, "detector_edge_face")
        margin = self._settings.detector_edge_margin_ratio
        if geometry.x1 <= width * margin or geometry.y1 <= height * margin or geometry.x2 >= width * (1.0 - margin) or geometry.y2 >= height * (1.0 - margin):
            if area <1200:
                self.small_face_rescue_failures +=1
                metrics.observe("phase2d2_small_face_rescue_failures", float(self.small_face_rescue_failures))
            return DetectionFilterDecision(False, "detector_edge_face")
        if self._settings.detector_center_priority_enabled:
            dx = abs(geometry.center_x - width / 2.0) / max(width / 2.0, 1.0)
            dy = abs(geometry.center_y - height / 2.0) / max(height / 2.0, 1.0)
            if (dx * dx + dy * dy) ** 0.5 > self._settings.detector_center_max_distance:
                if area <1200:
                    self.small_face_rescue_failures +=1
                    metrics.observe("phase2d2_small_face_rescue_failures", float(self.small_face_rescue_failures))
                return DetectionFilterDecision(False, "detector_low_center_priority")
        
        # Phase 2D.2: Small face rescue success!
        if area <1200:
            self.small_face_rescue_successes +=1
            self.average_small_face_area = (self.average_small_face_area * (self.small_face_rescue_successes -1) + area) / self.small_face_rescue_successes
            metrics.observe("phase2d2_small_face_rescue_successes", float(self.small_face_rescue_successes))
            metrics.observe("phase2d2_average_small_face_area", self.average_small_face_area)
        return DetectionFilterDecision(True)

    def _select_detector_size(self, frame_shape: tuple[int, ...]) -> tuple[int, int]:
        active_tracks = self.active_track_count()
        occupancy = self._occupancy_ratio(frame_shape)
        metrics.observe("track_count", active_tracks)
        metrics.observe("frame_occupancy_ratio", occupancy)
        
        # Phase 2D: Adaptive Resolution Escalation (160x160 →320x320→480x480)
        resolution_sizes = [(160, 160), (320, 320), (480, 480)]
        
        # Phase 2D.3: Check if escalation is suppressed
        if not self.escalation_suppression_active:
            # Check escalation conditions
            if self.zero_face_streaks >=3 or self.consecutive_weak_detections >=4:
                self.current_resolution_level = min(self.current_resolution_level +1, 2)
        else:
            # Escalation is suppressed!
            self.suppressed_resolution_escalations +=1
            metrics.observe("phase2d3_suppressed_resolution_escalations", float(self.suppressed_resolution_escalations))
            
        # Check reversion condition
        if self.validator_strict_pass_count >=3:
            self.current_resolution_level = 0
        
        metrics.observe("phase2d_resolution_level", float(self.current_resolution_level))
        return resolution_sizes[self.current_resolution_level]

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

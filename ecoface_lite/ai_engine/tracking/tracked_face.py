"""Per-track state carried through detection, tracking, and recognition."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ecoface_lite.ai_engine.tracking.track_state import ACTIVE_RECOGNITION_STATES, TrackLifecycleState


@dataclass
class TrackedFace:
    track_id: str

    bbox: tuple[float, float, float, float]
    confidence: float

    last_seen_frame: int
    first_seen_frame: int

    visibility_age: int
    lost_frames: int

    first_seen_ts: float = field(default_factory=time.monotonic)
    last_seen_ts: float = field(default_factory=time.monotonic)

    last_embedding: np.ndarray | None = None
    embedding_timestamp: float = 0.0
    last_embedding_frame: int = 0

    identity: int | None = None
    identity_confidence: float = 0.0

    smoothed_confidence: float = 0.0

    state: str = TrackLifecycleState.NEW.value

    center_point: tuple[float, float] = (0.0, 0.0)
    face_area: float = 0.0

    recognition_count: int = 0
    stable_match_count: int = 0

    recent_matches: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    recovery_count: int = 0
    identity_switch_count: int = 0

    confirmation_hits: int = 0
    track_quality_score: float = 0.0
    governance_protected: bool = True
    
    # ── Stability Hardening (Phase 3) ────────────────────────────────────────
    smoothed_bbox: tuple[float, float, float, float] | None = None

    # ── Phase 6: Telemetry & Grace ──────────────────────────────────────────
    governance_lockout_active: bool = False
    emergency_rebuild_active: bool = False
    recovery_grace_frames: int = 0
    
    # ── Phase 2C.4: Adaptive Continuity Confidence Refinement ─────────────
    # Objective 1: Temporal Confidence Decay Model
    confidence_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    confidence_velocity: float = 0.0
    last_stable_confidence: float = 0.0
    decay_window_frames: int = 0
    
    # Objective 2: Occlusion-Aware Continuity Memory
    occlusion_frame_count: int = 0
    predicted_visibility_window: int = 0
    continuity_memory_strength: float = 1.0
    occlusion_state: str = "visible"  # visible, occluded, recovering, reidentified
    
    # Objective 3: Profile-Angle Adaptive Acceptance
    profile_likelihood: float = 0.0
    profile_persistence_frames: int = 0
    aspect_ratio_baseline: float = 0.0
    
    # Objective 4: Adaptive Motion Responsiveness
    motion_intensity: float = 0.0
    acceleration_magnitude: float = 0.0
    direction_change_intensity: float = 0.0
    
    # Objective 5: Small-Face Continuity Tolerance
    small_face_mode: bool = False
    small_face_tolerance_multiplier: float = 1.0
    
    metadata: dict = field(default_factory=dict)

    @property
    def numeric_track_id(self) -> int:
        """Stable integer id for APIs that still expect int track ids."""
        raw = self.track_id.replace("track_", "")
        try:
            return int(raw)
        except ValueError:
            return hash(self.track_id) % 1_000_000

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_RECOGNITION_STATES or self.state == TrackLifecycleState.LOST.value

    @property
    def is_stable(self) -> bool:
        return self.state == TrackLifecycleState.STABLE.value

    @property
    def lifetime_ms(self) -> float:
        return (time.monotonic() - self.first_seen_ts) * 1000.0

    @property
    def time_since_last_seen_ms(self) -> float:
        return (time.monotonic() - self.last_seen_ts) * 1000.0

    def touch_embedding(self, embedding: np.ndarray, frame_index: int) -> None:
        self.last_embedding = embedding
        self.last_embedding_frame = frame_index
        self.embedding_timestamp = time.monotonic()

    def record_identity_match(self, person_id: int, confidence: float, alpha: float) -> None:
        from ecoface_lite.ai_engine.temporal_identity_state import get_temporal_identity

        previous_identity = self.identity
        self.recent_matches.append(person_id)
        temporal = get_temporal_identity(self)
        leading, hyp_conf = temporal.leading_identity()
        voted = leading if leading is not None else self._majority_identity()
        if voted is not None:
            if previous_identity is not None and voted != previous_identity:
                self.identity_switch_count += 1
            self.identity = voted
        self.identity_confidence = max(confidence, hyp_conf)
        if self.smoothed_confidence <= 0:
            self.smoothed_confidence = confidence
        else:
            self.smoothed_confidence = (alpha * confidence) + ((1.0 - alpha) * self.smoothed_confidence)
        self.recognition_count += 1
        dominant = self.identity if self.identity is not None else self._majority_identity()
        if dominant is not None:
            self.stable_match_count = sum(1 for pid in self.recent_matches if pid == dominant)
        self.metadata["temporal_consistency"] = temporal.temporal_consistency

    def update_bbox(self, new_bbox: tuple[float, float, float, float], alpha: float) -> None:
        """Apply EMA smoothing to bounding box coordinates."""
        from ecoface_lite.core.metrics import metrics
        
        if self.smoothed_bbox is None:
            self.smoothed_bbox = new_bbox
            self.bbox = new_bbox
            return

        # Telemetry for stability analysis
        old_bbox = self.bbox
        delta = sum(abs(a - b) for a, b in zip(old_bbox, new_bbox)) / 4.0
        metrics.observe("avg_bbox_delta_before", delta)

        # Apply EMA: smoothed = alpha * new + (1 - alpha) * old
        smoothed = tuple(
            alpha * n + (1.0 - alpha) * o 
            for n, o in zip(new_bbox, self.smoothed_bbox)
        )
        self.smoothed_bbox = smoothed
        self.bbox = smoothed # Tracker uses smoothed coords
        
        new_delta = sum(abs(a - b) for a, b in zip(old_bbox, smoothed)) / 4.0
        metrics.observe("avg_bbox_delta_after", new_delta)
        metrics.increment("bbox_smoothing_applied_count")

    def _majority_identity(self) -> int | None:
        if not self.recent_matches:
            return None
        return max(set(self.recent_matches), key=list(self.recent_matches).count)

    def apply_temporal_confidence_decay(
        self,
        current_confidence: float,
        is_detected: bool,
        decay_alpha: float = 0.95,
        recovery_alpha: float = 0.7,
        strong_detection_threshold: float = 0.85,
        weak_detection_threshold: float = 0.5,
    ) -> float:
        """
        Apply temporal confidence decay model (Objective 1).
        
        Replaces abrupt confidence collapse with gradual decay during instability.
        
        Args:
            current_confidence: Raw detector confidence for current frame
            is_detected: Whether face was detected this frame
            decay_alpha: Decay factor during instability (higher = slower decay)
            recovery_alpha: Recovery factor when strong detection returns
            strong_detection_threshold: Threshold for immediate confidence refresh
            weak_detection_threshold: Threshold for partial confidence reduction
            
        Returns:
            Adjusted confidence with temporal decay applied
        """
        from ecoface_lite.core.metrics import metrics
        
        # Update confidence history
        self.confidence_history.append(current_confidence)
        
        # Calculate confidence velocity (rate of change)
        if len(self.confidence_history) >= 2:
            self.confidence_velocity = self.confidence_history[-1] - self.confidence_history[-2]
        
        if is_detected:
            # Detection present
            if current_confidence >= strong_detection_threshold:
                # Strong detection: immediate refresh
                self.last_stable_confidence = current_confidence
                self.decay_window_frames = 0
                metrics.increment("confidence_refresh_strong")
                return current_confidence
            elif current_confidence >= weak_detection_threshold:
                # Weak detection: partial reduction
                decayed = (recovery_alpha * current_confidence) + ((1.0 - recovery_alpha) * self.last_stable_confidence)
                self.last_stable_confidence = max(self.last_stable_confidence, decayed)
                self.decay_window_frames = 0
                metrics.increment("confidence_refresh_weak")
                return decayed
            else:
                # Very weak detection: decay but preserve memory
                if self.last_stable_confidence > 0:
                    decayed = (decay_alpha * current_confidence) + ((1.0 - decay_alpha) * self.last_stable_confidence)
                    self.decay_window_frames += 1
                    metrics.increment("confidence_decay_weak")
                    return decayed
                return current_confidence
        else:
            # Missing detection: temporal decay
            if self.last_stable_confidence > 0:
                # Decay gradually based on decay window
                # Longer decay window = slower decay (more persistent memory)
                adaptive_decay = decay_alpha ** (1.0 + min(5.0, self.decay_window_frames / 3.0))
                decayed = adaptive_decay * self.last_stable_confidence
                self.decay_window_frames += 1
                metrics.increment("confidence_decay_missing")
                metrics.observe("confidence_decay_window", self.decay_window_frames)
                return decayed
            return current_confidence

    def update_occlusion_state(
        self,
        is_detected: bool,
        max_occlusion_frames: int = 8,
        recovery_frames: int = 5,
    ) -> str:
        """
        Update occlusion-aware continuity memory (Objective 2).
        
        Manages VISIBLE → OCCLUDED → RECOVERING → REIDENTIFIED state machine.
        
        Args:
            is_detected: Whether face was detected this frame
            max_occlusion_frames: Max frames to maintain occluded state before full decay
            recovery_frames: Frames required for full recovery
            
        Returns:
            Current occlusion state
        """
        from ecoface_lite.core.metrics import metrics
        
        if is_detected:
            if self.occlusion_state == "occluded":
                # Transition to recovering
                self.occlusion_state = "recovering"
                self.predicted_visibility_window = recovery_frames
                metrics.increment("occlusion_recovery_start")
            elif self.occlusion_state == "recovering":
                # Continue recovery
                self.predicted_visibility_window -= 1
                if self.predicted_visibility_window <= 0:
                    self.occlusion_state = "visible"
                    self.continuity_memory_strength = 1.0
                    metrics.increment("occlusion_recovery_complete")
            elif self.occlusion_state == "reidentified":
                # Already reidentified, transition back to visible
                self.occlusion_state = "visible"
                self.continuity_memory_strength = 1.0
                metrics.increment("occlusion_reidentified_to_visible")
            
            # Reset occlusion count when visible
            if self.occlusion_state == "visible":
                self.occlusion_frame_count = 0
                self.continuity_memory_strength = min(1.0, self.continuity_memory_strength + 0.1)
        else:
            # Not detected - enter or continue occlusion
            if self.occlusion_state == "visible":
                if self.occlusion_frame_count < max_occlusion_frames:
                    self.occlusion_state = "occluded"
                    self.predicted_visibility_window = max_occlusion_frames
                    metrics.increment("occlusion_enter")
            
            if self.occlusion_state in {"occluded", "recovering"}:
                self.occlusion_frame_count += 1
                # Gradually weaken continuity memory during prolonged occlusion
                if self.occlusion_frame_count > max_occlusion_frames:
                    self.continuity_memory_strength *= 0.9
                    if self.continuity_memory_strength < 0.3:
                        self.occlusion_state = "reidentified"
                        metrics.increment("occlusion_reidentified")
        
        metrics.observe("occlusion_state_duration", self.occlusion_frame_count)
        metrics.observe("continuity_memory_strength", self.continuity_memory_strength)
        return self.occlusion_state

    def estimate_profile_likelihood(
        self,
        current_aspect_ratio: float,
        aspect_ratio_change_threshold: float = 0.15,
        persistence_threshold: int = 3,
    ) -> float:
        """
        Estimate profile turn likelihood (Objective 3).
        
        Uses aspect ratio changes as a lightweight heuristic for profile detection.
        
        Args:
            current_aspect_ratio: Current bbox aspect ratio (width/height)
            aspect_ratio_change_threshold: Threshold for significant ratio change
            persistence_threshold: Frames of change to confirm profile
            
        Returns:
            Profile likelihood (0.0 = frontal, 1.0 = strong profile)
        """
        from ecoface_lite.core.metrics import metrics
        
        # Initialize baseline on first call
        if self.aspect_ratio_baseline == 0.0:
            self.aspect_ratio_baseline = current_aspect_ratio
            return 0.0
        
        # Calculate aspect ratio deviation
        ratio_change = abs(current_aspect_ratio - self.aspect_ratio_baseline) / max(0.01, self.aspect_ratio_baseline)
        
        if ratio_change > aspect_ratio_change_threshold:
            self.profile_persistence_frames += 1
            # Profile likelihood increases with persistence
            self.profile_likelihood = min(1.0, self.profile_persistence_frames / persistence_threshold)
        else:
            # Aspect ratio returned to normal - reset profile likelihood
            self.profile_persistence_frames = max(0, self.profile_persistence_frames - 1)
            self.profile_likelihood = max(0.0, self.profile_persistence_frames / persistence_threshold)
            # Update baseline if stable for a while
            if self.profile_persistence_frames == 0:
                self.aspect_ratio_baseline = (0.9 * self.aspect_ratio_baseline) + (0.1 * current_aspect_ratio)
        
        metrics.observe("profile_likelihood", self.profile_likelihood)
        metrics.observe("aspect_ratio_change", ratio_change)
        return self.profile_likelihood

    def update_motion_metrics(
        self,
        velocity: tuple[float, float],
        previous_velocity: tuple[float, float] | None = None,
    ) -> None:
        """
        Update motion responsiveness metrics (Objective 4).
        
        Calculates motion intensity, acceleration, and direction change for adaptive smoothing.
        
        Args:
            velocity: Current velocity (vx, vy)
            previous_velocity: Previous velocity for acceleration calculation
        """
        from ecoface_lite.core.metrics import metrics
        import math
        
        # Calculate motion intensity (velocity magnitude)
        self.motion_intensity = math.sqrt(velocity[0]**2 + velocity[1]**2)
        
        # Calculate acceleration if previous velocity available
        if previous_velocity is not None:
            ax = velocity[0] - previous_velocity[0]
            ay = velocity[1] - previous_velocity[1]
            self.acceleration_magnitude = math.sqrt(ax**2 + ay**2)
            
            # Calculate direction change intensity
            v_prev_mag = math.sqrt(previous_velocity[0]**2 + previous_velocity[1]**2)
            if v_prev_mag > 0.1:
                dot_product = (velocity[0] * previous_velocity[0] + velocity[1] * previous_velocity[1])
                cosine_angle = dot_product / (self.motion_intensity * v_prev_mag)
                cosine_angle = max(-1.0, min(1.0, cosine_angle))  # Clamp to valid range
                self.direction_change_intensity = 1.0 - cosine_angle  # 0 = same direction, 1 = opposite
            else:
                self.direction_change_intensity = 0.0
        else:
            self.acceleration_magnitude = 0.0
            self.direction_change_intensity = 0.0
        
        metrics.observe("motion_intensity", self.motion_intensity)
        metrics.observe("acceleration_magnitude", self.acceleration_magnitude)
        metrics.observe("direction_change_intensity", self.direction_change_intensity)

    def update_small_face_mode(
        self,
        face_area: float,
        small_face_threshold: float = 2500.0,
    ) -> None:
        """
        Update small-face continuity tolerance mode (Objective 5).
        
        Enables localized tolerance for small-face tracks.
        
        Args:
            face_area: Current face area in pixels
            small_face_threshold: Area threshold for small-face classification
        """
        from ecoface_lite.core.metrics import metrics
        
        self.small_face_mode = face_area < small_face_threshold
        
        if self.small_face_mode:
            # Increase tolerance multiplier for small faces
            self.small_face_tolerance_multiplier = 1.5
            metrics.increment("small_face_mode_active")
        else:
            self.small_face_tolerance_multiplier = 1.0
        
        metrics.observe("small_face_mode", 1.0 if self.small_face_mode else 0.0)

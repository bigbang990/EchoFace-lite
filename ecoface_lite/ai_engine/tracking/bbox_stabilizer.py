from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from ecoface_lite.core.metrics import metrics

@dataclass
class StabilizationConfig:
    """Configurable parameters for BBox stabilization."""
    alpha_min: float = 0.25      # Strong smoothing for stable targets
    alpha_max: float = 0.85      # Weak smoothing for fast-moving targets
    velocity_threshold: float = 8.0 # Pixel displacement for alpha scaling
    aspect_ratio_tolerance: float = 0.08
    ratio_persistence_frames: int = 5
    velocity_memory: int = 3
    
    # ── Phase 2C.4: Adaptive Motion Responsiveness (Objective 4) ─────────────
    motion_intensity_low_threshold: float = 3.0  # Pixels per frame
    motion_intensity_high_threshold: float = 15.0  # Pixels per frame
    acceleration_threshold: float = 5.0  # Pixels per frame^2
    direction_change_threshold: float = 0.5  # Cosine-based (0-1)
    reentry_boost_frames: int = 5

class BBoxStabilizer:
    """
    Localized stabilization layer for bounding box temporal smoothing.
    
    Implements:
    - Adaptive exponential smoothing
    - Aspect ratio stabilization
    - Lightweight velocity-aware prediction
    - Adaptive motion responsiveness (Phase 2C.4)
    """
    
    def __init__(self, config: StabilizationConfig | None = None):
        self.cfg = config or StabilizationConfig()
        
        self.last_raw_bbox: tuple[float, float, float, float] | None = None
        self.last_smoothed_bbox: tuple[float, float, float, float] | None = None
        
        # --- Task 2: State variables (x, y, vx, vy) ---
        self.state_center: tuple[float, float] | None = None
        self.state_velocity: tuple[float, float] = (0.0, 0.0)
        
        self.velocity_history: deque[tuple[float, float]] = deque(maxlen=self.cfg.velocity_memory)
        self.ratio_history: deque[float] = deque(maxlen=self.cfg.ratio_persistence_frames)
        self.stable_ratio: float | None = None
        
        # ── Phase 2C.4: Adaptive Motion Responsiveness ───────────────────────
        self.previous_velocity: tuple[float, float] = (0.0, 0.0)
        self.reentry_frame_count: int = 0

    def predict(self) -> tuple[float, float, float, float] | None:
        """
        Task 2: Constant-velocity prediction.
        Used ONLY during tracking-only frames (detector gaps).
        """
        if self.state_center is None or self.last_smoothed_bbox is None:
            return self.last_smoothed_bbox

        # Advance state
        vx, vy = self.state_velocity
        self.state_center = (self.state_center[0] + vx, self.state_center[1] + vy)
        
        # Reconstruct bbox from predicted center
        lx1, ly1, lx2, ly2 = self.last_smoothed_bbox
        w, h = lx2 - lx1, ly2 - ly1
        cx, cy = self.state_center
        
        predicted = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
        self.last_smoothed_bbox = predicted
        return predicted

    def stabilize(self, new_bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """
        Authoritative detector update with Snap Suppression (Task 3).
        """
        nx1, ny1, nx2, ny2 = new_bbox
        n_width = nx2 - nx1
        n_height = ny2 - ny1
        n_center = ((nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0)
        
        if self.last_smoothed_bbox is None:
            self.last_raw_bbox = new_bbox
            self.last_smoothed_bbox = new_bbox
            self.state_center = n_center
            self.stable_ratio = n_width / max(1.0, n_height)
            return new_bbox

        # 1. Velocity Update (Lightweight Kalman-like)
        if self.state_center is not None:
            # Observed velocity
            v_obs = (n_center[0] - self.state_center[0], n_center[1] - self.state_center[1])
            # Blend velocity (momentum)
            v_alpha = 0.4
            self.state_velocity = (
                v_alpha * v_obs[0] + (1.0 - v_alpha) * self.state_velocity[0],
                v_alpha * v_obs[1] + (1.0 - v_alpha) * self.state_velocity[1]
            )
            self.velocity_history.append(self.state_velocity)

        # 2. Snap Suppression (Task 3)
        # If detector bbox differs heavily from predicted, blend gradually
        lx1, ly1, lx2, ly2 = self.last_smoothed_bbox
        displacement = np.sqrt((n_center[0] - self.state_center[0])**2 + (n_center[1] - self.state_center[1])**2)
        
        # If displacement is large, reduce correction strength to prevent snaps
        # correction_strength: 1.0 (instant snap) -> lower (gradual)
        if displacement > 40.0:
            # Scale correction by magnitude
            correction_strength = max(0.2, 1.0 - min(0.7, (displacement - 40.0) / 100.0))
        else:
            correction_strength = 0.8 # Standard high-confidence follow

        # Blend authoritative detection into state
        self.state_center = (
            self.state_center[0] + correction_strength * (n_center[0] - self.state_center[0]),
            self.state_center[1] + correction_strength * (n_center[1] - self.state_center[1])
        )

        # Apply same blending to bbox dimensions to prevent size snaps
        smoothed = tuple(
            correction_strength * n + (1.0 - correction_strength) * s
            for n, s in zip(new_bbox, self.last_smoothed_bbox)
        )
        
        # 3. Aspect Ratio Stabilization (Persistent)
        # (Keeping existing ratio logic but updating it to use 'smoothed' result)
        s_x1, s_y1, s_x2, s_y2 = smoothed
        s_width = s_x2 - s_x1
        s_height = s_y2 - s_y1
        s_ratio = s_width / max(1.0, s_height)
        
        self.ratio_history.append(s_ratio)
        
        # Check if current ratio is far from stable ratio
        if self.stable_ratio is not None:
            ratio_diff = abs(s_ratio - self.stable_ratio) / self.stable_ratio
            if ratio_diff > self.cfg.aspect_ratio_tolerance:
                # Check if this change is persistent
                if len(self.ratio_history) == self.cfg.ratio_persistence_frames:
                    recent_avg_ratio = np.mean(self.ratio_history)
                    avg_diff = abs(recent_avg_ratio - self.stable_ratio) / self.stable_ratio
                    if avg_diff > self.cfg.aspect_ratio_tolerance:
                        # Persistence achieved, update stable ratio
                        self.stable_ratio = recent_avg_ratio
                
                # Enforce stable ratio on smoothed box
                # Keep center and area (roughly), adjust width/height to match stable ratio
                target_ratio = self.stable_ratio
                center_x = (s_x1 + s_x2) / 2.0
                center_y = (s_y1 + s_y2) / 2.0
                
                # Adjustment while preserving height (less prone to collapse than area-preserving)
                new_s_width = s_height * target_ratio
                half_w = new_s_width / 2.0
                smoothed = (center_x - half_w, s_y1, center_x + half_w, s_y2)
                
                metrics.increment("bbox_aspect_ratio_stabilized")

        self.last_raw_bbox = new_bbox
        self.last_smoothed_bbox = smoothed
        
        return smoothed

    def compute_adaptive_correction_strength(
        self,
        motion_intensity: float,
        acceleration_magnitude: float,
        direction_change_intensity: float,
        is_reentry: bool = False,
    ) -> float:
        """
        Compute adaptive correction strength based on motion metrics (Objective 4).
        
        Adjusts smoothing strength dynamically:
        - LOW MOTION: strong smoothing (stable)
        - HIGH MOTION: weaker smoothing (responsive)
        - RE-ENTRY: temporarily prioritize responsiveness
        - CAMERA SHAKE: preserve stabilization
        
        Args:
            motion_intensity: Velocity magnitude in pixels per frame
            acceleration_magnitude: Acceleration magnitude
            direction_change_intensity: Direction change intensity (0-1)
            is_reentry: Whether this is a re-entry event
            
        Returns:
            Adaptive correction strength (0.0-1.0, higher = more responsive)
        """
        import math
        
        # Base correction strength
        base_strength = 0.8
        
        # Motion intensity adaptation
        if motion_intensity < self.cfg.motion_intensity_low_threshold:
            # Low motion: strong smoothing
            motion_factor = 0.6
        elif motion_intensity > self.cfg.motion_intensity_high_threshold:
            # High motion: weaker smoothing (more responsive)
            motion_factor = 0.95
        else:
            # Linear interpolation between thresholds
            ratio = (motion_intensity - self.cfg.motion_intensity_low_threshold) / (
                self.cfg.motion_intensity_high_threshold - self.cfg.motion_intensity_low_threshold
            )
            motion_factor = 0.6 + (0.35 * ratio)
        
        # Acceleration adaptation
        if acceleration_magnitude > self.cfg.acceleration_threshold:
            # High acceleration: increase responsiveness
            accel_factor = 0.9
        else:
            accel_factor = 0.7
        
        # Direction change adaptation
        if direction_change_intensity > self.cfg.direction_change_threshold:
            # Sharp direction change: increase responsiveness
            direction_factor = 0.9
        else:
            direction_factor = 0.7
        
        # Re-entry boost
        if is_reentry and self.reentry_frame_count < self.cfg.reentry_boost_frames:
            self.reentry_frame_count += 1
            reentry_factor = 0.95
            metrics.increment("adaptive_reentry_boost")
        else:
            reentry_factor = 0.7
            self.reentry_frame_count = 0
        
        # Combine factors (weighted average)
        adaptive_strength = (
            0.4 * motion_factor +
            0.2 * accel_factor +
            0.2 * direction_factor +
            0.2 * reentry_factor
        )
        
        # Clamp to valid range
        adaptive_strength = max(0.2, min(1.0, adaptive_strength))
        
        metrics.observe("adaptive_correction_strength", adaptive_strength)
        metrics.observe("motion_intensity", motion_intensity)
        
        return adaptive_strength

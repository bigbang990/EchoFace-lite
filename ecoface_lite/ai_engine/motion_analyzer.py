"""Motion consistency analysis for track-centric false-detection rejection.

Real faces move smoothly; jittery or teleporting boxes are often detector noise.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


@dataclass(frozen=True)
class MotionSnapshot:
    velocity: tuple[float, float] = (0.0, 0.0)
    acceleration: tuple[float, float] = (0.0, 0.0)
    bbox_jitter: float = 0.0
    motion_smoothness: float = 1.0
    directional_continuity: float = 1.0
    motion_stability_score: float = 1.0


@dataclass
class _MotionHistory:
    centers: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=16))
    velocities: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=8))
    areas: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    last_frame: int = -1


class MotionAnalyzer:
    """Per-track motion history and stability scoring."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._history: dict[str, _MotionHistory] = {}

    def update(
        self,
        track_id: str,
        bbox: tuple[float, float, float, float],
        frame_index: int,
    ) -> MotionSnapshot:
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        area = max(1.0, (x2 - x1) * (y2 - y1))
        hist = self._history.setdefault(track_id, _MotionHistory())

        velocity = (0.0, 0.0)
        acceleration = (0.0, 0.0)
        if len(hist.centers) > 0:
            prev = hist.centers[-1]
            velocity = (center[0] - prev[0], center[1] - prev[1])
        if len(hist.velocities) > 0:
            prev_v = hist.velocities[-1]
            acceleration = (velocity[0] - prev_v[0], velocity[1] - prev_v[1])

        hist.centers.append(center)
        hist.velocities.append(velocity)
        hist.areas.append(area)
        hist.last_frame = frame_index

        jitter = self._bbox_jitter(hist.areas)
        smoothness = self._motion_smoothness(hist.velocities)
        continuity = self._directional_continuity(hist.velocities)
        teleport_penalty = self._teleport_penalty(velocity)
        stability = float(
            max(
                0.0,
                min(
                    1.0,
                    (0.35 * smoothness)
                    + (0.30 * continuity)
                    + (0.20 * (1.0 - jitter))
                    + (0.15 * (1.0 - teleport_penalty)),
                ),
            )
        )
        metrics.observe("motion_stability_score", stability)
        metrics.observe("bbox_jitter", jitter)
        return MotionSnapshot(
            velocity=velocity,
            acceleration=acceleration,
            bbox_jitter=jitter,
            motion_smoothness=smoothness,
            directional_continuity=continuity,
            motion_stability_score=stability,
        )

    def remove(self, track_id: str) -> None:
        self._history.pop(track_id, None)

    def _bbox_jitter(self, areas: deque[float]) -> float:
        if len(areas) < 2:
            return 0.0
        values = list(areas)
        mean_area = sum(values) / len(values)
        if mean_area <= 0:
            return 0.0
        variance = sum((a - mean_area) ** 2 for a in values) / len(values)
        rel_std = (variance ** 0.5) / mean_area
        return float(min(1.0, rel_std / max(self._settings.motion_max_area_jitter_ratio, 1e-6)))

    def _motion_smoothness(self, velocities: deque[tuple[float, float]]) -> float:
        if len(velocities) < 2:
            return 1.0
        speeds = [(v[0] ** 2 + v[1] ** 2) ** 0.5 for v in velocities]
        if not speeds:
            return 1.0
        mean_speed = sum(speeds) / len(speeds)
        if mean_speed < 1e-3:
            return 1.0
        variance = sum((s - mean_speed) ** 2 for s in speeds) / len(speeds)
        rel = (variance ** 0.5) / mean_speed
        return float(max(0.0, 1.0 - min(1.0, rel / max(self._settings.motion_max_speed_variance_ratio, 1e-6))))

    def _directional_continuity(self, velocities: deque[tuple[float, float]]) -> float:
        if len(velocities) < 2:
            return 1.0
        pairs = list(velocities)[-min(6, len(velocities)) :]
        dots = 0.0
        count = 0
        for i in range(1, len(pairs)):
            v0, v1 = pairs[i - 1], pairs[i]
            n0 = (v0[0] ** 2 + v0[1] ** 2) ** 0.5
            n1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
            if n0 < 1e-3 or n1 < 1e-3:
                continue
            cos_angle = (v0[0] * v1[0] + v0[1] * v1[1]) / (n0 * n1)
            dots += max(-1.0, min(1.0, cos_angle))
            count += 1
        if count == 0:
            return 1.0
        avg_cos = dots / count
        return float(max(0.0, min(1.0, (avg_cos + 1.0) / 2.0)))

    def _teleport_penalty(self, velocity: tuple[float, float]) -> float:
        speed = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
        max_jump = self._settings.motion_max_frame_displacement_px
        if speed <= max_jump:
            return 0.0
        return float(min(1.0, (speed - max_jump) / max(max_jump, 1.0)))

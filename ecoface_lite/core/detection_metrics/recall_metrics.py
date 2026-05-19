"""Recall metrics calculator for measuring detection performance."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RecallMetrics:
    """Recall performance metrics."""
    tiny_face_recall_rate: float
    crowd_face_miss_rate: float
    detection_flicker_rate: float
    temporal_stability_score: float
    tracker_id_fragmentation_rate: float
    overall_recall_score: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "tiny_face_recall_rate": self.tiny_face_recall_rate,
            "crowd_face_miss_rate": self.crowd_face_miss_rate,
            "detection_flicker_rate": self.detection_flicker_rate,
            "temporal_stability_score": self.temporal_stability_score,
            "tracker_id_fragmentation_rate": self.tracker_id_fragmentation_rate,
            "overall_recall_score": self.overall_recall_score,
        }


class RecallMetricsCalculator:
    """Calculates and tracks recall metrics over time."""

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._tiny_face_detections: deque[int] = deque(maxlen=window_size)
        self._tiny_face_ground_truth: deque[int] = deque(maxlen=window_size)
        self._crowd_face_detections: deque[int] = deque(maxlen=window_size)
        self._crowd_face_ground_truth: deque[int] = deque(maxlen=window_size)
        self._detection_flickers: deque[bool] = deque(maxlen=window_size)
        self._temporal_stability_scores: deque[float] = deque(maxlen=window_size)
        self._tracker_id_changes: deque[int] = deque(maxlen=window_size)
        self._total_tracks: deque[int] = deque(maxlen=window_size)

    def record_tiny_face_detection(self, detected: int, ground_truth: int) -> None:
        """Record tiny face detection vs ground truth."""
        self._tiny_face_detections.append(detected)
        self._tiny_face_ground_truth.append(ground_truth)

    def record_crowd_face_detection(self, detected: int, ground_truth: int) -> None:
        """Record crowd face detection vs ground truth."""
        self._crowd_face_detections.append(detected)
        self._crowd_face_ground_truth.append(ground_truth)

    def record_detection_flicker(self, flickered: bool) -> None:
        """Record whether detection flickered in current frame."""
        self._detection_flickers.append(flickered)

    def record_temporal_stability(self, stability_score: float) -> None:
        """Record temporal stability score (0.0-1.0)."""
        self._temporal_stability_scores.append(stability_score)

    def record_tracker_id_change(self, id_changes: int, total_tracks: int) -> None:
        """Record tracker ID changes and total tracks."""
        self._tracker_id_changes.append(id_changes)
        self._total_tracks.append(total_tracks)

    def calculate_metrics(self) -> RecallMetrics:
        """Calculate current recall metrics."""
        tiny_recall = self._calculate_tiny_face_recall()
        crowd_miss = self._calculate_crowd_face_miss_rate()
        flicker_rate = self._calculate_detection_flicker_rate()
        temporal_stability = self._calculate_temporal_stability()
        id_fragmentation = self._calculate_id_fragmentation_rate()
        overall = self._calculate_overall_recall(
            tiny_recall, crowd_miss, flicker_rate, temporal_stability, id_fragmentation
        )

        return RecallMetrics(
            tiny_face_recall_rate=tiny_recall,
            crowd_face_miss_rate=crowd_miss,
            detection_flicker_rate=flicker_rate,
            temporal_stability_score=temporal_stability,
            tracker_id_fragmentation_rate=id_fragmentation,
            overall_recall_score=overall,
        )

    def _calculate_tiny_face_recall(self) -> float:
        """Calculate tiny face recall rate."""
        if not self._tiny_face_ground_truth:
            return 0.0

        total_gt = sum(self._tiny_face_ground_truth)
        if total_gt == 0:
            return 0.0

        total_detected = sum(self._tiny_face_detections)
        return min(1.0, total_detected / total_gt)

    def _calculate_crowd_face_miss_rate(self) -> float:
        """Calculate crowd face miss rate."""
        if not self._crowd_face_ground_truth:
            return 0.0

        total_gt = sum(self._crowd_face_ground_truth)
        if total_gt == 0:
            return 0.0

        total_detected = sum(self._crowd_face_detections)
        missed = max(0, total_gt - total_detected)
        return min(1.0, missed / total_gt)

    def _calculate_detection_flicker_rate(self) -> float:
        """Calculate detection flicker rate (fraction of frames with flicker)."""
        if not self._detection_flickers:
            return 0.0

        flicker_count = sum(1 for f in self._detection_flickers if f)
        return flicker_count / len(self._detection_flickers)

    def _calculate_temporal_stability(self) -> float:
        """Calculate average temporal stability score."""
        if not self._temporal_stability_scores:
            return 0.0

        return np.mean(self._temporal_stability_scores)

    def _calculate_id_fragmentation_rate(self) -> float:
        """Calculate tracker ID fragmentation rate."""
        if not self._total_tracks:
            return 0.0

        total_tracks = sum(self._total_tracks)
        if total_tracks == 0:
            return 0.0

        total_changes = sum(self._tracker_id_changes)
        return min(1.0, total_changes / total_tracks)

    def _calculate_overall_recall(
        self,
        tiny_recall: float,
        crowd_miss: float,
        flicker_rate: float,
        temporal_stability: float,
        id_fragmentation: float,
    ) -> float:
        """Calculate overall recall score as weighted combination."""
        # Higher is better for all metrics except miss rate and flicker
        crowd_recall = 1.0 - crowd_miss
        stability = 1.0 - flicker_rate
        id_consistency = 1.0 - id_fragmentation

        # Weighted average (tune weights based on priorities)
        weights = {
            "tiny_recall": 0.30,
            "crowd_recall": 0.25,
            "stability": 0.20,
            "temporal": 0.15,
            "id_consistency": 0.10,
        }

        overall = (
            weights["tiny_recall"] * tiny_recall
            + weights["crowd_recall"] * crowd_recall
            + weights["stability"] * stability
            + weights["temporal"] * temporal_stability
            + weights["id_consistency"] * id_consistency
        )

        return overall

    def get_target_progress(self) -> dict[str, float]:
        """Get progress toward target metrics."""
        current = self.calculate_metrics()
        targets = {
            "tiny_face_recall_rate": 0.85,  # Target >85%
            "crowd_face_miss_rate": 0.30,  # Target <30% (70% reduction)
            "detection_flicker_rate": 0.20,  # Target <20% (80% reduction)
            "temporal_stability_score": 0.80,  # Target >80%
            "tracker_id_fragmentation_rate": 0.50,  # Target <50% (50% reduction)
        }

        progress = {}
        for metric, target in targets.items():
            current_value = getattr(current, metric)
            if metric in ["crowd_face_miss_rate", "detection_flicker_rate", "tracker_id_fragmentation_rate"]:
                # Lower is better
                progress[metric] = max(0.0, 1.0 - (current_value / target)) if target > 0 else 0.0
            else:
                # Higher is better
                progress[metric] = min(1.0, current_value / target) if target > 0 else 0.0

        return progress

    def reset(self) -> None:
        """Reset all metrics."""
        self._tiny_face_detections.clear()
        self._tiny_face_ground_truth.clear()
        self._crowd_face_detections.clear()
        self._crowd_face_ground_truth.clear()
        self._detection_flickers.clear()
        self._temporal_stability_scores.clear()
        self._tracker_id_changes.clear()
        self._total_tracks.clear()

"""Per-frame detection metrics collector for enterprise observability."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


class FaceSizeBucket(str, Enum):
    """Face size categories for metrics tracking."""
    TINY = "tiny"  # <30px
    SMALL = "small"  # 30-60px
    MEDIUM = "medium"  # 60-120px
    LARGE = "large"  # >120px


@dataclass
class FrameDetectionMetrics:
    """Per-frame detection metrics."""
    frame_id: int
    timestamp: float
    total_faces_detected: int
    tiny_faces_detected: int
    small_faces_detected: int
    medium_faces_detected: int
    large_faces_detected: int
    validator_rejections: int
    average_face_size: float
    minimum_face_size: float
    maximum_face_size: float
    detection_latency_ms: float
    tracker_survival_time_frames: float
    weak_detection_promotions: int
    false_positive_count: int
    face_size_buckets: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "total_faces_detected": self.total_faces_detected,
            "tiny_faces_detected": self.tiny_faces_detected,
            "small_faces_detected": self.small_faces_detected,
            "medium_faces_detected": self.medium_faces_detected,
            "large_faces_detected": self.large_faces_detected,
            "validator_rejections": self.validator_rejections,
            "average_face_size": self.average_face_size,
            "minimum_face_size": self.minimum_face_size,
            "maximum_face_size": self.maximum_face_size,
            "detection_latency_ms": self.detection_latency_ms,
            "tracker_survival_time_frames": self.tracker_survival_time_frames,
            "weak_detection_promotions": self.weak_detection_promotions,
            "false_positive_count": self.false_positive_count,
            "face_size_buckets": self.face_size_buckets,
        }


class DetectionMetricsCollector:
    """Collects and aggregates per-frame detection metrics."""

    def __init__(self, export_dir: Path, export_interval: int = 100) -> None:
        self._export_dir = export_dir
        self._export_interval = export_interval
        self._frame_metrics: list[FrameDetectionMetrics] = []
        self._current_frame_id = 0
        self._export_count = 0

        if self._export_dir:
            self._export_dir.mkdir(parents=True, exist_ok=True)

    def record_frame_start(self, frame_id: int) -> float:
        """Record frame start time and return timestamp."""
        self._current_frame_id = frame_id
        return datetime.now().timestamp()

    def record_detection(
        self,
        faces: list[Any],
        frame_shape: tuple[int, int],
        detection_latency_ms: float,
        validator_rejections: int = 0,
        weak_promotions: int = 0,
        false_positives: int = 0,
        tracker_survival_time: float = 0.0,
    ) -> None:
        """Record detection metrics for current frame."""
        if not faces:
            self._record_empty_frame(
                detection_latency_ms,
                validator_rejections,
                weak_promotions,
                false_positives,
                tracker_survival_time,
            )
            return

        face_sizes = self._extract_face_sizes(faces, frame_shape)
        size_buckets = self._categorize_face_sizes(face_sizes)

        metrics = FrameDetectionMetrics(
            frame_id=self._current_frame_id,
            timestamp=datetime.now().timestamp(),
            total_faces_detected=len(faces),
            tiny_faces_detected=size_buckets.get(FaceSizeBucket.TINY.value, 0),
            small_faces_detected=size_buckets.get(FaceSizeBucket.SMALL.value, 0),
            medium_faces_detected=size_buckets.get(FaceSizeBucket.MEDIUM.value, 0),
            large_faces_detected=size_buckets.get(FaceSizeBucket.LARGE.value, 0),
            validator_rejections=validator_rejections,
            average_face_size=np.mean(face_sizes) if face_sizes else 0.0,
            minimum_face_size=np.min(face_sizes) if face_sizes else 0.0,
            maximum_face_size=np.max(face_sizes) if face_sizes else 0.0,
            detection_latency_ms=detection_latency_ms,
            tracker_survival_time_frames=tracker_survival_time,
            weak_detection_promotions=weak_promotions,
            false_positive_count=false_positives,
            face_size_buckets=size_buckets,
        )

        self._frame_metrics.append(metrics)

        if len(self._frame_metrics) >= self._export_interval:
            self._export_metrics()

    def _record_empty_frame(
        self,
        detection_latency_ms: float,
        validator_rejections: int,
        weak_promotions: int,
        false_positives: int,
        tracker_survival_time: float,
    ) -> None:
        """Record metrics for frame with no detections."""
        metrics = FrameDetectionMetrics(
            frame_id=self._current_frame_id,
            timestamp=datetime.now().timestamp(),
            total_faces_detected=0,
            tiny_faces_detected=0,
            small_faces_detected=0,
            medium_faces_detected=0,
            large_faces_detected=0,
            validator_rejections=validator_rejections,
            average_face_size=0.0,
            minimum_face_size=0.0,
            maximum_face_size=0.0,
            detection_latency_ms=detection_latency_ms,
            tracker_survival_time_frames=tracker_survival_time,
            weak_detection_promotions=weak_promotions,
            false_positive_count=false_positives,
            face_size_buckets={},
        )
        self._frame_metrics.append(metrics)

        if len(self._frame_metrics) >= self._export_interval:
            self._export_metrics()

    def _extract_face_sizes(self, faces: list[Any], frame_shape: tuple[int, int]) -> list[float]:
        """Extract face sizes from detected faces."""
        sizes = []
        for face in faces:
            if hasattr(face, "bbox"):
                bbox = face.bbox
                width = bbox.x2 - bbox.x1
                height = bbox.y2 - bbox.y1
                size = max(width, height)
                sizes.append(size)
        return sizes

    def _categorize_face_sizes(self, face_sizes: list[float]) -> dict[str, int]:
        """Categorize faces into size buckets."""
        buckets = {
            FaceSizeBucket.TINY.value: 0,
            FaceSizeBucket.SMALL.value: 0,
            FaceSizeBucket.MEDIUM.value: 0,
            FaceSizeBucket.LARGE.value: 0,
        }

        for size in face_sizes:
            if size < 30:
                buckets[FaceSizeBucket.TINY.value] += 1
            elif size < 60:
                buckets[FaceSizeBucket.SMALL.value] += 1
            elif size < 120:
                buckets[FaceSizeBucket.MEDIUM.value] += 1
            else:
                buckets[FaceSizeBucket.LARGE.value] += 1

        return buckets

    def _export_metrics(self) -> None:
        """Export accumulated metrics to JSON file."""
        if not self._export_dir or not self._frame_metrics:
            return

        self._export_count += 1
        export_path = self._export_dir / f"detection_metrics_{self._export_count:06d}.json"

        metrics_data = [m.to_dict() for m in self._frame_metrics]
        with open(export_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(
            f"Exported {len(self._frame_metrics)} frame metrics to {export_path}"
        )
        self._frame_metrics.clear()

    def get_recent_metrics(self, n: int = 100) -> list[FrameDetectionMetrics]:
        """Get the most recent n frame metrics."""
        return self._frame_metrics[-n:]

    def get_aggregate_metrics(self) -> dict[str, Any]:
        """Get aggregate statistics across all recorded frames."""
        if not self._frame_metrics:
            return {}

        total_frames = len(self._frame_metrics)
        total_faces = sum(m.total_faces_detected for m in self._frame_metrics)
        total_tiny = sum(m.tiny_faces_detected for m in self._frame_metrics)
        total_small = sum(m.small_faces_detected for m in self._frame_metrics)
        total_medium = sum(m.medium_faces_detected for m in self._frame_metrics)
        total_large = sum(m.large_faces_detected for m in self._frame_metrics)
        total_rejections = sum(m.validator_rejections for m in self._frame_metrics)
        total_promotions = sum(m.weak_detection_promotions for m in self._frame_metrics)
        total_false_positives = sum(m.false_positive_count for m in self._frame_metrics)

        avg_latency = np.mean([m.detection_latency_ms for m in self._frame_metrics])
        avg_face_size = np.mean([m.average_face_size for m in self._frame_metrics if m.average_face_size > 0])

        tiny_ratio = total_tiny / total_faces if total_faces > 0 else 0.0
        small_ratio = total_small / total_faces if total_faces > 0 else 0.0
        medium_ratio = total_medium / total_faces if total_faces > 0 else 0.0
        large_ratio = total_large / total_faces if total_faces > 0 else 0.0

        rejection_rate = total_rejections / total_faces if total_faces > 0 else 0.0

        return {
            "total_frames": total_frames,
            "total_faces_detected": total_faces,
            "total_tiny_faces": total_tiny,
            "total_small_faces": total_small,
            "total_medium_faces": total_medium,
            "total_large_faces": total_large,
            "total_validator_rejections": total_rejections,
            "total_weak_promotions": total_promotions,
            "total_false_positives": total_false_positives,
            "average_faces_per_frame": total_faces / total_frames if total_frames > 0 else 0.0,
            "average_detection_latency_ms": avg_latency,
            "average_face_size": avg_face_size,
            "tiny_face_ratio": tiny_ratio,
            "small_face_ratio": small_ratio,
            "medium_face_ratio": medium_ratio,
            "large_face_ratio": large_ratio,
            "validator_rejection_rate": rejection_rate,
            "weak_promotion_rate": total_promotions / total_faces if total_faces > 0 else 0.0,
            "false_positive_rate": total_false_positives / total_faces if total_faces > 0 else 0.0,
        }

    def flush(self) -> None:
        """Force export of remaining metrics."""
        if self._frame_metrics:
            self._export_metrics()

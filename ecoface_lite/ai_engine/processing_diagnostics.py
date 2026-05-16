from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class VideoJobDiagnostics:
    job_id: str | None = None
    frames_processed: int = 0
    faces_detected: int = 0
    faces_rejected: int = 0
    blur_rejections: int = 0
    size_rejections: int = 0
    low_confidence_rejections: int = 0
    duplicate_suppressions: int = 0
    alerts_created: int = 0
    confidence_sum: float = 0.0
    confidence_count: int = 0
    started_at: float = field(default_factory=perf_counter)

    def observe_confidence(self, confidence: float | None) -> None:
        if confidence is None:
            return
        self.confidence_sum += float(confidence)
        self.confidence_count += 1

    @property
    def avg_confidence(self) -> float:
        if self.confidence_count == 0:
            return 0.0
        return self.confidence_sum / self.confidence_count

    @property
    def duration_seconds(self) -> float:
        return perf_counter() - self.started_at

    @property
    def avg_fps(self) -> float:
        duration = self.duration_seconds
        if duration <= 0:
            return 0.0
        return self.frames_processed / duration

    def as_analytics(self) -> dict[str, float | int]:
        return {
            "avg_fps": self.avg_fps,
            "avg_confidence": self.avg_confidence,
            "total_faces_detected": self.faces_detected,
            "total_faces_rejected": self.faces_rejected,
            "blur_rejections": self.blur_rejections,
            "duplicate_suppressions": self.duplicate_suppressions,
            "processing_duration_seconds": self.duration_seconds,
        }

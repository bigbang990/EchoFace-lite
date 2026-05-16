from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from threading import Lock
from time import time
from typing import Any


@dataclass(frozen=True)
class DiagnosticEvent:
    category: str
    reason: str
    frame_index: int | None = None
    job_id: str | None = None
    person_id: int | None = None
    confidence: float | None = None
    threshold: float | None = None
    metadata: dict[str, Any] | None = None
    created_at: float = 0.0


class DiagnosticsRecorder:
    def __init__(self, recent_window: int = 300) -> None:
        self._recent_events: deque[DiagnosticEvent] = deque(maxlen=recent_window)
        self._reason_counts: Counter[str] = Counter()
        self._category_counts: Counter[str] = Counter()
        self._confidence_values: deque[float] = deque(maxlen=recent_window)
        self._lock = Lock()

    def record(
        self,
        category: str,
        reason: str,
        *,
        frame_index: int | None = None,
        job_id: str | None = None,
        person_id: int | None = None,
        confidence: float | None = None,
        threshold: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = DiagnosticEvent(
            category=category,
            reason=reason,
            frame_index=frame_index,
            job_id=job_id,
            person_id=person_id,
            confidence=confidence,
            threshold=threshold,
            metadata=metadata,
            created_at=time(),
        )
        with self._lock:
            self._recent_events.append(event)
            self._reason_counts[reason] += 1
            self._category_counts[category] += 1
            if confidence is not None:
                self._confidence_values.append(float(confidence))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            confidences = list(self._confidence_values)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return {
                "reason_counts": dict(self._reason_counts),
                "category_counts": dict(self._category_counts),
                "recent_events": [asdict(event) for event in list(self._recent_events)[-100:]],
                "confidence_values": confidences,
                "average_confidence": avg_confidence,
            }

    def reset(self) -> None:
        with self._lock:
            self._recent_events.clear()
            self._reason_counts.clear()
            self._category_counts.clear()
            self._confidence_values.clear()


diagnostics = DiagnosticsRecorder()

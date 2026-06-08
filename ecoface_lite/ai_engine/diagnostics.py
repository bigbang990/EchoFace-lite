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
        
        # --- Phase 3: Telemetry Contract Hardening ---
        self.telemetry_contract_version = "1.1.0"
        self._unknown_kwarg_count = 0
        self._normalized_calls = 0
        self._interface_mismatch_count = 0

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
        **kwargs: Any,
    ) -> None:
        """Record a diagnostic event with backward-compatible normalization."""
        try:
            # Step 2: Interface Normalization
            # If "count" exists, normalize into metadata or ignore if it was meant for record(..., count=X)
            if "count" in kwargs:
                if metadata is None:
                    metadata = {}
                metadata["count"] = kwargs.pop("count")
                self._normalized_calls += 1

            # Detect unknown kwargs
            if kwargs:
                self._unknown_kwarg_count += 1
                self._interface_mismatch_count += 1
                # We could log a debug warning here but we must remain non-blocking
                # from ecoface_lite.core.logging import get_logger
                # logger = get_logger(__name__)
                # logger.debug("Telemetry mismatch: unknown kwargs %s", kwargs.keys())

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
        except Exception:
            # --- Phase 4: Pipeline Safety Guarantee ---
            # Telemetry must NEVER terminate processing loops.
            # We fail silently or increment a failure counter if we had one for internal errors.
            pass

    def increment(self, reason: str, category: str = "general", **kwargs: Any) -> None:
        """Shortcut for recording a simple counter increment."""
        self.record(category, reason, **kwargs)

    def warning(self, category: str, msg: str, **kwargs: Any) -> None:
        """Record a warning diagnostic."""
        self.record(category, msg, **kwargs)

    def timing(self, category: str, reason: str, duration_ms: float, **kwargs: Any) -> None:
        """Record a timing measurement."""
        if "metadata" not in kwargs:
            kwargs["metadata"] = {}
        kwargs["metadata"]["duration_ms"] = duration_ms
        self.record(category, reason, **kwargs)

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
                "telemetry_health": {
                    "contract_version": self.telemetry_contract_version,
                    "unknown_kwarg_count": self._unknown_kwarg_count,
                    "normalized_calls": self._normalized_calls,
                    "interface_mismatch_count": self._interface_mismatch_count,
                }
            }

    def reset(self) -> None:
        with self._lock:
            self._recent_events.clear()
            self._reason_counts.clear()
            self._category_counts.clear()
            self._confidence_values.clear()


diagnostics = DiagnosticsRecorder()

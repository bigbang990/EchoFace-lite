"""Event timeline collector for detection event tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionEvent:
    """A detection event in the timeline."""
    timestamp: float
    frame_id: int
    event_type: str
    track_id: int | None = None
    face_size: float = 0.0
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "event": self.event_type,
            "track_id": self.track_id,
            "face_size": self.face_size,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
            "metadata": self.metadata,
        }


class EventTimeline:
    """Collect and manage detection event timeline."""

    def __init__(self, max_events: int = 10000) -> None:
        self._events: deque[DetectionEvent] = deque(maxlen=max_events)
        self._max_events = max_events

    def record_event(
        self,
        frame_id: int,
        event_type: str,
        track_id: int | None = None,
        face_size: float = 0.0,
        confidence_before: float = 0.0,
        confidence_after: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a detection event.

        Args:
            frame_id: Frame index
            event_type: Type of event (e.g., "detection_created", "weak_detection_promoted")
            track_id: Associated track ID
            face_size: Face bounding box size
            confidence_before: Confidence before the event
            confidence_after: Confidence after the event
            metadata: Additional event metadata
        """
        event = DetectionEvent(
            timestamp=datetime.utcnow().timestamp(),
            frame_id=frame_id,
            event_type=event_type,
            track_id=track_id,
            face_size=face_size,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            metadata=metadata or {},
        )
        self._events.append(event)

    def record_detection_created(
        self,
        frame_id: int,
        track_id: int,
        face_size: float,
        confidence: float,
    ) -> None:
        """Record a detection created event."""
        self.record_event(
            frame_id=frame_id,
            event_type="detection_created",
            track_id=track_id,
            face_size=face_size,
            confidence_after=confidence,
        )

    def record_weak_detection_promoted(
        self,
        frame_id: int,
        track_id: int,
        face_size: float,
        confidence_before: float,
        confidence_after: float,
    ) -> None:
        """Record a weak detection promoted event."""
        self.record_event(
            frame_id=frame_id,
            event_type="weak_detection_promoted",
            track_id=track_id,
            face_size=face_size,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
        )

    def record_validator_rejected(
        self,
        frame_id: int,
        track_id: int | None,
        face_size: float,
        confidence: float,
        rejection_reason: str,
    ) -> None:
        """Record a validator rejected event."""
        self.record_event(
            frame_id=frame_id,
            event_type="validator_rejected",
            track_id=track_id,
            face_size=face_size,
            confidence_after=confidence,
            metadata={"rejection_reason": rejection_reason},
        )

    def record_track_created(
        self,
        frame_id: int,
        track_id: int,
        face_size: float,
        confidence: float,
    ) -> None:
        """Record a track created event."""
        self.record_event(
            frame_id=frame_id,
            event_type="track_created",
            track_id=track_id,
            face_size=face_size,
            confidence_after=confidence,
        )

    def record_track_lost(
        self,
        frame_id: int,
        track_id: int,
        face_size: float,
        confidence: float,
    ) -> None:
        """Record a track lost event."""
        self.record_event(
            frame_id=frame_id,
            event_type="track_lost",
            track_id=track_id,
            face_size=face_size,
            confidence_after=confidence,
        )

    def record_identity_matched(
        self,
        frame_id: int,
        track_id: int,
        person_id: int,
        confidence: float,
    ) -> None:
        """Record an identity matched event."""
        self.record_event(
            frame_id=frame_id,
            event_type="identity_matched",
            track_id=track_id,
            confidence_after=confidence,
            metadata={"person_id": person_id},
        )

    def get_events_by_type(self, event_type: str) -> list[DetectionEvent]:
        """Get all events of a specific type.

        Args:
            event_type: Type of event to filter

        Returns:
            List of events of the specified type
        """
        return [e for e in self._events if e.event_type == event_type]

    def get_events_by_track(self, track_id: int) -> list[DetectionEvent]:
        """Get all events for a specific track.

        Args:
            track_id: Track ID to filter

        Returns:
            List of events for the specified track
        """
        return [e for e in self._events if e.track_id == track_id]

    def get_events_by_frame(self, frame_id: int) -> list[DetectionEvent]:
        """Get all events for a specific frame.

        Args:
            frame_id: Frame ID to filter

        Returns:
            List of events for the specified frame
        """
        return [e for e in self._events if e.frame_id == frame_id]

    def to_dict(self) -> dict[str, Any]:
        """Convert timeline to dictionary format.

        Returns:
            Dictionary with all events
        """
        return {
            "events": [e.to_dict() for e in self._events],
            "total_events": len(self._events),
        }

    def clear(self) -> None:
        """Clear all events from the timeline."""
        self._events.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get timeline statistics.

        Returns:
            Dictionary with event statistics
        """
        event_types = {}
        for event in self._events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

        return {
            "total_events": len(self._events),
            "event_types": event_types,
        }

"""Track lifecycle states for temporal face perception."""

from __future__ import annotations

from enum import Enum


class TrackLifecycleState(str, Enum):
    """Lifecycle of a face track through the perception pipeline."""

    NEW = "new"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    STABLE = "stable"
    COARSE = "coarse"
    LOST = "lost"
    REMOVED = "removed"


# States eligible for recognition / overlay output
ACTIVE_RECOGNITION_STATES = frozenset(
    {
        TrackLifecycleState.NEW.value,
        TrackLifecycleState.CANDIDATE.value,
        TrackLifecycleState.CONFIRMED.value,
        TrackLifecycleState.STABLE.value,
        TrackLifecycleState.COARSE.value,
    }
)

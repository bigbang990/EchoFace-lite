"""Experiment notes tracker for recording experimental adjustments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Adjustment:
    """An experimental adjustment recorded during the session."""
    timestamp: str
    adjustment: str
    old_value: Any
    new_value: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "adjustment": self.adjustment,
            "old_value": str(self.old_value),
            "new_value": str(self.new_value),
            "reason": self.reason,
        }


class ExperimentNotesTracker:
    """Track experimental adjustments and notes during a session."""

    def __init__(self, max_adjustments: int = 100) -> None:
        self._adjustments: deque[Adjustment] = deque(maxlen=max_adjustments)
        self._max_adjustments = max_adjustments

    def record_adjustment(
        self,
        adjustment: str,
        old_value: Any,
        new_value: Any,
        reason: str,
    ) -> None:
        """Record an experimental adjustment.

        Args:
            adjustment: Description of the adjustment (e.g., "reduced confidence threshold")
            old_value: Previous value
            new_value: New value
            reason: Reason for the adjustment
        """
        adj = Adjustment(
            timestamp=datetime.utcnow().isoformat(),
            adjustment=adjustment,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
        )
        self._adjustments.append(adj)
        logger.info(
            "Adjustment recorded: %s (%s -> %s) - %s",
            adjustment,
            old_value,
            new_value,
            reason,
        )

    def record_confidence_threshold_change(
        self,
        old_value: float,
        new_value: float,
        reason: str,
    ) -> None:
        """Record a confidence threshold change."""
        self.record_adjustment(
            adjustment="confidence_threshold",
            old_value=old_value,
            new_value=new_value,
            reason=reason,
        )

    def record_feature_toggle(
        self,
        feature_name: str,
        enabled: bool,
        reason: str,
    ) -> None:
        """Record a feature flag toggle."""
        self.record_adjustment(
            adjustment=f"feature_flag_{feature_name}",
            old_value=not enabled,
            new_value=enabled,
            reason=reason,
        )

    def record_detector_change(
        self,
        old_detector: str,
        new_detector: str,
        reason: str,
    ) -> None:
        """Record a detector change."""
        self.record_adjustment(
            adjustment="detector_change",
            old_value=old_detector,
            new_value=new_detector,
            reason=reason,
        )

    def record_validator_modification(
        self,
        parameter: str,
        old_value: Any,
        new_value: Any,
        reason: str,
    ) -> None:
        """Record a validator parameter modification."""
        self.record_adjustment(
            adjustment=f"validator_{parameter}",
            old_value=old_value,
            new_value=new_value,
            reason=reason,
        )

    def get_adjustments(self) -> list[Adjustment]:
        """Get all recorded adjustments.

        Returns:
            List of adjustments
        """
        return list(self._adjustments)

    def get_adjustments_by_type(self, adjustment_type: str) -> list[Adjustment]:
        """Get adjustments of a specific type.

        Args:
            adjustment_type: Type of adjustment to filter

        Returns:
            List of adjustments of the specified type
        """
        return [a for a in self._adjustments if adjustment_type in a.adjustment]

    def to_dict(self) -> dict[str, Any]:
        """Convert adjustments to dictionary format.

        Returns:
            Dictionary with all adjustments
        """
        return {
            "adjustments": [a.to_dict() for a in self._adjustments],
            "total_adjustments": len(self._adjustments),
        }

    def clear(self) -> None:
        """Clear all adjustments."""
        self._adjustments.clear()

    def get_summary(self) -> str:
        """Get a summary of all adjustments.

        Returns:
            Formatted summary string
        """
        if not self._adjustments:
            return "No adjustments recorded."

        summary_lines = ["Experimental Adjustments:\n"]
        for adj in self._adjustments:
            summary_lines.append(
                f"- [{adj.timestamp}] {adj.adjustment}: {adj.old_value} -> {adj.new_value}"
            )
            summary_lines.append(f"  Reason: {adj.reason}")

        return "\n".join(summary_lines)

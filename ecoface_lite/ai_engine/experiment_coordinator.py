"""Coordinator for experiment session, notes, and event-timeline methods.

Extracted from RecognitionPipeline (Phase 8A). RecognitionPipeline still
exposes these methods — they delegate here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class ExperimentCoordinator:
    def __init__(
        self,
        experiment_exporter: Optional[Any],
        detection_metrics: Optional[Any],
        settings: Any,
        notes_tracker: Optional[Any],
        event_timeline: Optional[Any],
    ) -> None:
        self._experiment_exporter = experiment_exporter
        self._detection_metrics = detection_metrics
        self._settings = settings
        self._notes_tracker = notes_tracker
        self._event_timeline = event_timeline

    def export_experiment_session(
        self,
        video_name: str,
        video_duration: float = 0.0,
        frame_count: int = 0,
        test_operator: str = "",
        notes: str = "",
    ) -> Path:
        """Export complete experiment session."""
        if not self._experiment_exporter:
            raise RuntimeError("Experiment export system is not enabled")

        self._experiment_exporter.set_metadata(
            video_name=video_name,
            video_duration=video_duration,
            frame_count=frame_count,
            test_operator=test_operator,
            notes=notes,
        )

        metrics_data = {}
        if self._detection_metrics:
            metrics_data["per_frame_metrics"] = self._detection_metrics.get_all_metrics()

        export_dir = self._settings.resolved_export_dir()
        export_dir.mkdir(parents=True, exist_ok=True)

        return self._experiment_exporter.export_session(export_dir, metrics_data)

    def record_experiment_adjustment(
        self,
        adjustment: str,
        old_value: Any,
        new_value: Any,
        reason: str,
    ) -> None:
        """Record an experimental adjustment."""
        if self._notes_tracker:
            self._notes_tracker.record_adjustment(
                adjustment=adjustment,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
            )

    def get_experiment_notes(self) -> str:
        """Get experiment notes summary."""
        if self._notes_tracker:
            return self._notes_tracker.get_summary()
        return "No experiment notes available."

    def get_event_timeline_statistics(self) -> dict[str, Any]:
        """Get event timeline statistics."""
        if self._event_timeline:
            return self._event_timeline.get_statistics()
        return {}

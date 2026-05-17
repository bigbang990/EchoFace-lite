from __future__ import annotations

from dataclasses import dataclass

from ecoface_lite.ai_engine.recognition_session import StableRecognition
from ecoface_lite.config.tracking import get_tracking_config
from ecoface_lite.core.config import Settings


@dataclass(frozen=True)
class EventDecision:
    should_emit: bool
    reason: str


class EventValidator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cfg = get_tracking_config(settings)
        self._last_event_frame_by_person: dict[int, int] = {}

    def evaluate(self, recognition: StableRecognition, frame_index: int) -> EventDecision:
        if not recognition.stable:
            return EventDecision(False, "unstable_recognition")
        if recognition.stable_match_count < self._cfg.min_stable_matches:
            return EventDecision(False, "insufficient_stable_duration")
        if recognition.visibility_age < self._cfg.min_track_age:
            return EventDecision(False, "track_too_young")
        if recognition.confirmations < self._settings.event_min_stable_frames:
            return EventDecision(False, "insufficient_stable_duration")
        previous = self._last_event_frame_by_person.get(recognition.person_id)
        if previous is not None and frame_index - previous < self._settings.event_cooldown_frames:
            return EventDecision(False, "cooldown")
        self._last_event_frame_by_person[recognition.person_id] = frame_index
        return EventDecision(True, "accepted")

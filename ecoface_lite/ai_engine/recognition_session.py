from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ecoface_lite.ai_engine.track_manager import TrackManager, TrackState
from ecoface_lite.core.config import Settings


@dataclass(frozen=True)
class StableRecognition:
    track_id: int
    person_id: int
    confidence: float
    confirmations: int
    stable: bool
    state: str
    age_frames: int


class RecognitionSession:
    def __init__(self, settings: Settings, track_manager: TrackManager | None = None) -> None:
        self._settings = settings
        self._track_manager = track_manager or TrackManager(settings)

    def observe(self, face: Any, frame_index: int, person_id: int, confidence: float) -> StableRecognition:
        track = self._track_manager.update(face, frame_index, person_id, confidence)
        return self._stable_state(track)

    def candidate_track_id(self, face: Any, frame_index: int) -> int | None:
        track = self._track_manager.candidate_track(face, frame_index)
        return track.track_id if track is not None else None

    def _stable_state(self, track: TrackState) -> StableRecognition:
        person_id = track.dominant_person_id()
        confirmations = track.dominant_count()
        average_confidence = track.average_confidence()
        stable = (
            person_id is not None
            and confirmations >= self._settings.temporal_min_confirmations
            and average_confidence >= self._settings.temporal_min_average_confidence
        )
        if stable:
            state = "stable_match"
        elif confirmations > 0:
            state = "tentative_match"
        else:
            state = "candidate_match"
        return StableRecognition(
            track_id=track.track_id,
            person_id=int(person_id) if person_id is not None else -1,
            confidence=average_confidence,
            confirmations=confirmations,
            stable=stable,
            state=state,
            age_frames=track.age_frames,
        )

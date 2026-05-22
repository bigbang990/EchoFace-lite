from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ecoface_lite.ai_engine.tracking.track_manager import FaceTrackManager
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.config.tracking import get_tracking_config
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


@dataclass(frozen=True)
class StableRecognition:
    track_id: int
    person_id: int
    confidence: float
    confirmations: int
    stable: bool
    state: str
    age_frames: int
    visibility_age: int = 0
    stable_match_count: int = 0
    smoothed_confidence: float = 0.0


class RecognitionSession:
    def __init__(self, settings: Settings, track_manager: FaceTrackManager | None = None) -> None:
        self._settings = settings
        self._cfg = get_tracking_config(settings)
        self._track_manager = track_manager or FaceTrackManager(settings)

    @property
    def track_manager(self) -> FaceTrackManager:
        return self._track_manager

    def observe(self, face: Any, frame_index: int, person_id: int, confidence: float) -> StableRecognition:
        track = self._track_manager.candidate_track(face, frame_index)
        if track is None:
            # Fallback 1: Try associating with existing tracks or spawning new
            results = self._track_manager.update_from_detections([face], frame_index)
            track = results[0][1] if results else None
            
        if track is None:
            # Fallback 2: Direct spawn if association failed (preventing pipeline crash)
            # This ensures we always have a track to attach recognition data to.
            track = self._track_manager._spawn_track(face, frame_index)
            
        return self.observe_track(track, person_id, confidence)

    def stable_from_track(self, track: TrackedFace) -> StableRecognition:
        """Read stabilized recognition state without adding another vote."""
        return self._stable_state(track)

    def observe_track(self, track: TrackedFace, person_id: int, confidence: float) -> StableRecognition:
        prev_switches = track.identity_switch_count
        track.record_identity_match(person_id, confidence, self._cfg.ema_alpha)
        if track.identity_switch_count > prev_switches:
            metrics.increment("identity_switches")
        return self._stable_state(track)

    def candidate_track_id(self, face: Any, frame_index: int) -> int | None:
        track = self._track_manager.candidate_track(face, frame_index)
        return track.numeric_track_id if track is not None else None

    def _stable_state(self, track: TrackedFace) -> StableRecognition:
        from ecoface_lite.ai_engine.temporal_identity_state import get_temporal_identity

        temporal = get_temporal_identity(track)
        leading, hyp_conf = temporal.leading_identity(min_confidence=self._settings.tracking_min_soft_threshold * 0.5)
        person_id = leading if leading is not None else track.identity
        confirmations = track.stable_match_count
        average_confidence = max(track.smoothed_confidence or 0.0, track.identity_confidence, hyp_conf)
        temporal_ok = temporal.temporal_consistency >= self._settings.tracking_temporal_lock_min_consistency
        stable = (
            person_id is not None
            and confirmations >= self._settings.temporal_min_confirmations
            and average_confidence >= self._settings.temporal_min_average_confidence
            and (temporal_ok or confirmations >= self._settings.tracking_temporal_lock_min_agreement)
        )
        if stable:
            state = "stable_match"
            metrics.increment("stable_matches")
        elif confirmations > 0:
            state = "tentative_match"
        else:
            state = "candidate_match"
        return StableRecognition(
            track_id=track.numeric_track_id,
            person_id=int(person_id) if person_id is not None else -1,
            confidence=average_confidence,
            confirmations=confirmations,
            stable=stable,
            state=state,
            age_frames=track.visibility_age,
            visibility_age=track.visibility_age,
            stable_match_count=track.stable_match_count,
            smoothed_confidence=track.smoothed_confidence,
        )

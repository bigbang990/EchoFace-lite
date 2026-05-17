"""Backward-compatible shim — prefer `ecoface_lite.ai_engine.tracking`."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.ai_engine.tracking.track_manager import FaceTrackManager

__all__ = ["TrackManager", "TrackState", "FaceTrackManager"]


@dataclass
class TrackState:
    """Legacy track state used by older imports; maps to TrackedFace fields."""

    track_id: int
    centroid: tuple[float, float]
    bbox: tuple[float, float, float, float]
    first_frame_index: int
    last_frame_index: int
    person_votes: deque[int] = field(default_factory=deque)
    confidences: deque[float] = field(default_factory=deque)
    trajectory: deque[tuple[float, float]] = field(default_factory=deque)

    def update(
        self,
        face: DetectedFace,
        centroid: tuple[float, float],
        frame_index: int,
        person_id: int,
        confidence: float,
        window_size: int,
    ) -> None:
        self.centroid = centroid
        self.bbox = (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2)
        self.last_frame_index = frame_index
        self.person_votes.append(person_id)
        self.confidences.append(confidence)
        self.trajectory.append(centroid)
        while len(self.person_votes) > window_size:
            self.person_votes.popleft()
        while len(self.confidences) > window_size:
            self.confidences.popleft()
        while len(self.trajectory) > window_size:
            self.trajectory.popleft()

    @property
    def age_frames(self) -> int:
        return max(1, self.last_frame_index - self.first_frame_index + 1)

    def dominant_person_id(self) -> int | None:
        if not self.person_votes:
            return None
        return max(set(self.person_votes), key=list(self.person_votes).count)

    def dominant_count(self) -> int:
        dominant = self.dominant_person_id()
        if dominant is None:
            return 0
        return sum(1 for person_id in self.person_votes if person_id == dominant)

    def average_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        weights = range(1, len(self.confidences) + 1)
        weighted_total = sum(confidence * weight for confidence, weight in zip(self.confidences, weights))
        return float(weighted_total / sum(weights))


class TrackManager:
    """Legacy wrapper delegating to FaceTrackManager for incremental migration."""

    def __init__(self, settings) -> None:
        self._settings = settings
        self._inner = FaceTrackManager(settings)
        self._legacy: dict[int, TrackState] = {}

    def update(self, face: DetectedFace, frame_index: int, person_id: int, confidence: float) -> TrackState:
        tracks = self._inner.update_from_detections([face], frame_index)
        track = tracks[0] if tracks else self._inner._spawn_track(face, frame_index)
        track.record_identity_match(person_id, confidence, self._settings.tracking_ema_alpha)
        legacy = self._to_legacy(track, frame_index, person_id, confidence)
        self._legacy[legacy.track_id] = legacy
        return legacy

    def candidate_track(self, face: DetectedFace, frame_index: int) -> TrackState | None:
        tracked = self._inner.candidate_track(face, frame_index)
        if tracked is None:
            return None
        return self._legacy.get(tracked.numeric_track_id) or self._to_legacy(tracked, frame_index, -1, 0.0)

    def _to_legacy(
        self,
        track,
        frame_index: int,
        person_id: int,
        confidence: float,
    ) -> TrackState:
        tid = track.numeric_track_id
        existing = self._legacy.get(tid)
        if existing is None:
            existing = TrackState(
                track_id=tid,
                centroid=track.center_point,
                bbox=track.bbox,
                first_frame_index=track.first_seen_frame,
                last_frame_index=frame_index,
            )
        if person_id >= 0:
            existing.update(
                self._inner.to_detected_face(track),
                track.center_point,
                frame_index,
                person_id,
                confidence,
                self._settings.temporal_window_size,
            )
        return existing

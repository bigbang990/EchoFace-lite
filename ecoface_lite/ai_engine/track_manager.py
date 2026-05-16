from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.core.config import Settings


@dataclass
class TrackState:
    track_id: int
    centroid: tuple[float, float]
    last_frame_index: int
    person_votes: deque[int] = field(default_factory=deque)
    confidences: deque[float] = field(default_factory=deque)

    def update(self, centroid: tuple[float, float], frame_index: int, person_id: int, confidence: float, window_size: int) -> None:
        self.centroid = centroid
        self.last_frame_index = frame_index
        self.person_votes.append(person_id)
        self.confidences.append(confidence)
        while len(self.person_votes) > window_size:
            self.person_votes.popleft()
        while len(self.confidences) > window_size:
            self.confidences.popleft()

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
        return float(sum(self.confidences) / len(self.confidences))


class TrackManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tracks: dict[int, TrackState] = {}
        self._next_id = 1

    def update(self, face: DetectedFace, frame_index: int, person_id: int, confidence: float) -> TrackState:
        self._expire(frame_index)
        centroid = ((face.bbox.x1 + face.bbox.x2) / 2.0, (face.bbox.y1 + face.bbox.y2) / 2.0)
        track = self._nearest_track(centroid)
        if track is None:
            track = TrackState(track_id=self._next_id, centroid=centroid, last_frame_index=frame_index)
            self._tracks[track.track_id] = track
            self._next_id += 1
        track.update(centroid, frame_index, person_id, confidence, self._settings.temporal_window_size)
        return track

    def _nearest_track(self, centroid: tuple[float, float]) -> TrackState | None:
        best: TrackState | None = None
        best_distance = self._settings.temporal_max_track_distance
        for track in self._tracks.values():
            dx = centroid[0] - track.centroid[0]
            dy = centroid[1] - track.centroid[1]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= best_distance:
                best = track
                best_distance = distance
        return best

    def _expire(self, frame_index: int) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame_index > self._settings.temporal_track_ttl_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

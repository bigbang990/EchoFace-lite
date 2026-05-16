from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.ai_engine.geometry import bbox_iou
from ecoface_lite.core.config import Settings


@dataclass
class TrackState:
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
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tracks: dict[int, TrackState] = {}
        self._next_id = 1

    def update(self, face: DetectedFace, frame_index: int, person_id: int, confidence: float) -> TrackState:
        self._expire(frame_index)
        centroid = ((face.bbox.x1 + face.bbox.x2) / 2.0, (face.bbox.y1 + face.bbox.y2) / 2.0)
        track = self._best_track(face, centroid)
        if track is None:
            track = TrackState(
                track_id=self._next_id,
                centroid=centroid,
                bbox=(face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2),
                first_frame_index=frame_index,
                last_frame_index=frame_index,
            )
            self._tracks[track.track_id] = track
            self._next_id += 1
        track.update(face, centroid, frame_index, person_id, confidence, self._settings.temporal_window_size)
        return track

    def candidate_track(self, face: DetectedFace, frame_index: int) -> TrackState | None:
        self._expire(frame_index)
        centroid = ((face.bbox.x1 + face.bbox.x2) / 2.0, (face.bbox.y1 + face.bbox.y2) / 2.0)
        return self._best_track(face, centroid)

    def _best_track(self, face: DetectedFace, centroid: tuple[float, float]) -> TrackState | None:
        best: TrackState | None = None
        best_score = 0.0
        for track in self._tracks.values():
            dx = centroid[0] - track.centroid[0]
            dy = centroid[1] - track.centroid[1]
            distance = (dx * dx + dy * dy) ** 0.5
            distance_score = max(0.0, 1.0 - (distance / max(self._settings.temporal_max_track_distance, 1.0)))
            track_bbox = type(face.bbox)(*track.bbox)
            iou = bbox_iou(face.bbox, track_bbox)
            score = (0.7 * iou) + (0.3 * distance_score)
            if (iou >= self._settings.temporal_min_track_iou or distance <= self._settings.temporal_max_track_distance) and score > best_score:
                best = track
                best_score = score
        return best

    def _expire(self, frame_index: int) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame_index > self._settings.temporal_track_ttl_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

"""SORT-style multi-face tracker — primary real-time perception layer.

Uses IoU + centroid association with linear bbox prediction on skipped detection
frames. Detections must be confirmed across multiple detector cycles before a track
is admitted (reduces single-frame hallucinations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace
from ecoface_lite.ai_engine.geometry import bbox_iou, compute_face_geometry
from ecoface_lite.ai_engine.motion_analyzer import MotionAnalyzer
from ecoface_lite.ai_engine.track_quality_engine import TrackQualityEngine, TrackQualitySnapshot
from ecoface_lite.ai_engine.tracking.track_state import ACTIVE_RECOGNITION_STATES, TrackLifecycleState
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.config.tracking import TrackingConfig, get_tracking_config
from ecoface_lite.core.logging import get_logger
from ecoface_lite.core.metrics import metrics

if TYPE_CHECKING:
    from ecoface_lite.core.config import Settings

logger = get_logger(__name__)


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(1.0, (x2 - x1) * (y2 - y1))


@dataclass
class _PendingCandidate:
    face: DetectedFace
    hits: int = 1
    last_frame: int = 0
    centroid: tuple[float, float] = (0.0, 0.0)


class FaceTrackManager:
    """Assigns persistent track ids and propagates faces between detector cycles."""

    def __init__(self, settings: Settings | None = None, config: TrackingConfig | None = None) -> None:
        from ecoface_lite.core.config import get_settings

        self._settings = settings or get_settings()
        self._cfg = config or get_tracking_config(self._settings)
        self._tracks: dict[str, TrackedFace] = {}
        self._pending: list[_PendingCandidate] = []
        self._removed_buffer: list[TrackedFace] = []
        self._next_id = 1
        self._motion = MotionAnalyzer(self._settings)
        self._quality_engine = TrackQualityEngine(self._settings)

    @property
    def active_track_count(self) -> int:
        return sum(
            1
            for t in self._tracks.values()
            if t.is_active and t.state not in {TrackLifecycleState.LOST.value, TrackLifecycleState.REMOVED.value}
        )

    def active_tracks(self) -> list[TrackedFace]:
        return [t for t in self._tracks.values() if t.is_active and t.state != TrackLifecycleState.REMOVED.value]

    def consume_removed_tracks(self) -> list[TrackedFace]:
        """Tracks transitioned to REMOVED since last consume (for global re-id memory)."""
        out = self._removed_buffer
        self._removed_buffer = []
        return out

    def visible_tracks(self) -> list[TrackedFace]:
        """Tracks that should produce overlay / pipeline output this frame."""
        return [t for t in self._tracks.values() if t.state in ACTIVE_RECOGNITION_STATES and self._is_admitted(t)]

    def average_motion_stability(self) -> float:
        scores = [float(t.metadata.get("motion_score", 1.0)) for t in self.active_tracks()]
        if len(scores) == 0:
            return 0.0
        return sum(scores) / len(scores)

    def candidate_track(self, face: DetectedFace, frame_index: int) -> TrackedFace | None:
        self._expire_removed()
        centroid = _bbox_center((face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2))
        track = self._best_match(face, centroid, frame_index)
        if track is not None and self._is_admitted(track):
            return track
        return None

    def update_from_detections(
        self,
        faces: list[DetectedFace],
        frame_index: int,
        *,
        frame_shape: tuple[int, ...] | None = None,
        frame_bgr: np.ndarray | None = None,
    ) -> list[tuple[DetectedFace, TrackedFace | None]]:
        """Associate detector outputs to tracks (detection frame).
        
        Returns:
            List of (DetectedFace, TrackedFace | None) pairs in the same order as input faces.
        """
        self._expire_removed()
        matched_ids: set[str] = set()
        results: list[tuple[DetectedFace, TrackedFace | None]] = []

        for face in faces:
            bbox = (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2)
            centroid = _bbox_center(bbox)
            track = self._best_match(face, centroid, frame_index)
            if track is None:
                track = self._admit_or_queue_pending(face, frame_index)
                if track is None:
                    # New track pending confirmation or rejected
                    results.append((face, None))
                    continue
                metrics.increment("new_tracks_created")
            elif track.state == TrackLifecycleState.LOST.value:
                track.recovery_count += 1
                metrics.increment("recovered_tracks")
                self._transition(track, TrackLifecycleState.CONFIRMED, frame_index, "recovered")
            
            matched_ids.add(track.track_id)
            self._apply_detection(track, face, frame_index, frame_shape, frame_bgr)
            
            if self._is_admitted(track):
                results.append((face, track))
            else:
                results.append((face, None))

        self._decay_pending(frame_index)
        self._prune_low_quality_tracks(frame_index)

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_ids:
                continue
            if track.state == TrackLifecycleState.REMOVED.value:
                continue
            track.lost_frames += 1
            self._quality_engine.decay_lost_track(track)
            if track.state != TrackLifecycleState.LOST.value:
                self._transition(track, TrackLifecycleState.LOST, frame_index, "unmatched")
            if track.lost_frames > self._cfg.max_lost_frames:
                self._transition(track, TrackLifecycleState.REMOVED, frame_index, "expired")
                metrics.increment("stale_track_replacements")

        metrics.observe("active_tracks", self.active_track_count)
        # Filter out None tracks for metrics calculation
        valid_tracks = [t for _, t in results if t is not None]
        if valid_tracks:
            metrics.observe("avg_track_duration", sum(t.visibility_age for t in valid_tracks) / len(valid_tracks))
            metrics.observe("avg_track_quality", sum(t.track_quality_score for t in valid_tracks) / len(valid_tracks))
        return results

    def propagate(self, frame_index: int) -> list[TrackedFace]:
        """Predict track positions on frames where the detector is skipped."""
        self._expire_removed()
        propagated: list[TrackedFace] = []
        for track in list(self._tracks.values()):
            if not track.is_active or track.state == TrackLifecycleState.REMOVED.value:
                continue
            if not self._is_admitted(track):
                continue
            velocity = track.metadata.get("velocity", (0.0, 0.0))
            x1, y1, x2, y2 = track.bbox
            track.bbox = (x1 + velocity[0], y1 + velocity[1], x2 + velocity[0], y2 + velocity[1])
            track.center_point = _bbox_center(track.bbox)
            track.face_area = _bbox_area(track.bbox)
            track.last_seen_frame = frame_index
            track.visibility_age = frame_index - track.first_seen_frame + 1
            motion = self._motion.update(track.track_id, track.bbox, frame_index)
            track.metadata["velocity"] = motion.velocity
            track.metadata["motion_score"] = motion.motion_stability_score
            self._quality_engine.update(track, None, None, motion, frame_index)
            self._advance_lifecycle(track, frame_index)
            propagated.append(track)
        metrics.observe("active_tracks", self.active_track_count)
        metrics.observe("tracker_reuse_rate", 1.0)
        return propagated

    def should_compute_embedding(
        self,
        track: TrackedFace,
        frame_index: int,
        face: DetectedFace | None = None,
        *,
        quality: TrackQualitySnapshot | None = None,
    ) -> bool:
        if track.state not in {
            TrackLifecycleState.CONFIRMED.value,
            TrackLifecycleState.STABLE.value,
        }:
            metrics.increment("embedding_suppressed_unconfirmed")
            return False

        if quality is not None and not quality.recognition_allowed:
            metrics.increment("embedding_suppressed_low_quality")
            return False

        cooldown = self._cfg.embedding_cooldown_frames
        if track.last_embedding is not None and frame_index - track.last_embedding_frame < cooldown:
            if face is None:
                metrics.increment("embedding_skips")
                return False

        if track.last_embedding is None:
            return True
        if frame_index - track.last_embedding_frame >= self._cfg.recognition_interval:
            metrics.increment("embedding_cache_refresh_due")
            return True
        if face is not None:
            overlap = bbox_iou(face.bbox, BoundingBox(*track.bbox))
            if overlap < self._cfg.recognition_cache_min_iou:
                metrics.increment("embedding_cache_invalidations")
                return True
            prev_area = track.metadata.get("last_face_area", track.face_area)
            new_area = _bbox_area((face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2))
            if prev_area > 0 and abs(new_area - prev_area) / prev_area > self._cfg.bbox_area_change_ratio:
                metrics.increment("embedding_cache_invalidations")
                return True
            score = face.temporal_score if face.temporal_score is not None else face.det_score
            if score + self._cfg.confidence_drop_threshold < track.confidence:
                metrics.increment("embedding_cache_invalidations")
                return True
        metrics.increment("embedding_skips")
        return False

    def update_track_quality(
        self,
        track: TrackedFace,
        face: DetectedFace | None,
        frame_bgr: np.ndarray | None,
        frame_index: int,
    ) -> TrackQualitySnapshot:
        motion = self._motion.update(track.track_id, track.bbox, frame_index)
        track.metadata["velocity"] = motion.velocity
        return self._quality_engine.update(track, face, frame_bgr, motion, frame_index)

    def to_detected_face(self, track: TrackedFace) -> DetectedFace:
        x1, y1, x2, y2 = track.bbox
        landmarks = track.metadata.get("landmarks")
        return DetectedFace(
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            det_score=track.confidence,
            landmarks=landmarks,
        )

    def _admit_or_queue_pending(self, face: DetectedFace, frame_index: int) -> TrackedFace | None:
        bbox = face.bbox
        centroid = _bbox_center((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        for pending in self._pending:
            if bbox_iou(bbox, pending.face.bbox) >= self._settings.temporal_min_track_iou:
                pending.hits += 1
                pending.face = face
                pending.last_frame = frame_index
                pending.centroid = _bbox_center((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
                if pending.hits >= self._cfg.confirm_frames:
                    self._pending.remove(pending)
                    track = self._spawn_track(face, frame_index)
                    track.confirmation_hits = pending.hits
                    self._transition(track, TrackLifecycleState.CANDIDATE, frame_index, "confirmed_pending")
                    return track
                metrics.increment("track_confirmation_pending")
                return None
        candidate = _PendingCandidate(
            face=face,
            hits=1,
            last_frame=frame_index,
            centroid=centroid,
        )
        if candidate.hits >= self._cfg.confirm_frames:
            track = self._spawn_track(face, frame_index)
            track.confirmation_hits = candidate.hits
            self._transition(track, TrackLifecycleState.CANDIDATE, frame_index, "instant_confirm")
            return track
        self._pending.append(candidate)
        metrics.increment("track_confirmation_pending")
        return None

    def _decay_pending(self, frame_index: int) -> None:
        ttl = max(2, self._cfg.confirm_frames + 1)
        self._pending = [p for p in self._pending if frame_index - p.last_frame <= ttl]

    def _spawn_track(self, face: DetectedFace, frame_index: int) -> TrackedFace:
        track_id = f"track_{self._next_id}"
        self._next_id += 1
        bbox = (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2)
        track = TrackedFace(
            track_id=track_id,
            bbox=bbox,
            confidence=face.det_score,
            last_seen_frame=frame_index,
            first_seen_frame=frame_index,
            visibility_age=1,
            lost_frames=0,
            center_point=_bbox_center(bbox),
            face_area=_bbox_area(bbox),
            state=TrackLifecycleState.NEW.value,
            confirmation_hits=1,
        )
        if face.landmarks is not None:
            track.metadata["landmarks"] = face.landmarks
        self._tracks[track_id] = track
        metrics.increment("track_state_new")
        return track

    def _apply_detection(
        self,
        track: TrackedFace,
        face: DetectedFace,
        frame_index: int,
        frame_shape: tuple[int, ...] | None,
        frame_bgr: np.ndarray | None,
    ) -> None:
        prev_center = track.center_point
        bbox = (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2)
        track.bbox = bbox
        score = face.temporal_score if face.temporal_score is not None else face.det_score
        track.confidence = score
        track.center_point = _bbox_center(bbox)
        track.face_area = _bbox_area(bbox)
        track.metadata["last_face_area"] = track.face_area
        track.last_seen_frame = frame_index
        track.visibility_age = frame_index - track.first_seen_frame + 1
        track.lost_frames = 0
        track.confirmation_hits = max(track.confirmation_hits, self._cfg.confirm_frames)
        if face.landmarks is not None:
            track.metadata["landmarks"] = face.landmarks
        if prev_center != (0.0, 0.0):
            track.metadata["velocity"] = (
                track.center_point[0] - prev_center[0],
                track.center_point[1] - prev_center[1],
            )
        motion = self._motion.update(track.track_id, track.bbox, frame_index)
        track.metadata["velocity"] = motion.velocity
        self._quality_engine.update(track, face, frame_bgr, motion, frame_index)
        self._advance_lifecycle(track, frame_index)
        metrics.observe("face_visibility_ratio", 1.0)

    def _advance_lifecycle(self, track: TrackedFace, frame_index: int) -> None:
        if track.confirmation_hits < self._cfg.confirm_frames:
            return
        if track.state == TrackLifecycleState.NEW.value:
            self._transition(track, TrackLifecycleState.CANDIDATE, frame_index, "confirmation_hits")
        if track.state in {TrackLifecycleState.NEW.value, TrackLifecycleState.CANDIDATE.value}:
            if track.visibility_age >= self._cfg.confirm_frames:
                self._transition(track, TrackLifecycleState.CONFIRMED, frame_index, "visibility_confirmed")
        if (
            track.state == TrackLifecycleState.CONFIRMED.value
            and track.visibility_age >= self._cfg.stable_frames
            and track.track_quality_score >= self._cfg.min_recognition_quality
        ):
            self._transition(track, TrackLifecycleState.STABLE, frame_index, "stable_track")

    def _transition(
        self,
        track: TrackedFace,
        new_state: TrackLifecycleState,
        frame_index: int,
        reason: str,
    ) -> None:
        if track.state == new_state.value:
            return
        prev = track.state
        if new_state == TrackLifecycleState.REMOVED and prev != TrackLifecycleState.REMOVED.value:
            self._removed_buffer.append(track)
        track.state = new_state.value
        metrics.increment(f"track_state_{new_state.value}")
        logger.debug(
            "track_state_transition id=%s %s->%s frame=%s reason=%s",
            track.track_id,
            prev,
            new_state.value,
            frame_index,
            reason,
        )

    def _prune_low_quality_tracks(self, frame_index: int) -> None:
        min_q = self._settings.tracking_min_quality_score
        for track_id, track in list(self._tracks.items()):
            if track.state == TrackLifecycleState.REMOVED.value:
                continue
            if track.visibility_age < self._cfg.confirm_frames:
                continue
            motion_score = float(track.metadata.get("motion_score", 1.0))
            if track.track_quality_score < min_q and track.lost_frames == 0 and motion_score < self._cfg.min_motion_stability:
                self._transition(track, TrackLifecycleState.REMOVED, frame_index, "low_quality")
                metrics.increment("low_quality_track_kills")

    def _is_admitted(self, track: TrackedFace) -> bool:
        return track.confirmation_hits >= self._cfg.confirm_frames

    def _best_match(
        self,
        face: DetectedFace,
        centroid: tuple[float, float],
        frame_index: int,
    ) -> TrackedFace | None:
        best: TrackedFace | None = None
        best_score = 0.0
        max_distance = self._settings.temporal_max_track_distance
        for track in self._tracks.values():
            if not track.is_active or track.state == TrackLifecycleState.REMOVED.value:
                continue
            if frame_index - track.last_seen_frame > self._cfg.max_lost_frames:
                continue
            dx = centroid[0] - track.center_point[0]
            dy = centroid[1] - track.center_point[1]
            distance = (dx * dx + dy * dy) ** 0.5
            distance_score = max(0.0, 1.0 - (distance / max(max_distance, 1.0)))
            iou = bbox_iou(face.bbox, BoundingBox(*track.bbox))
            score = (0.7 * iou) + (0.3 * distance_score)
            if (
                iou >= self._settings.temporal_min_track_iou or distance <= max_distance
            ) and score > best_score:
                best = track
                best_score = score
        return best

    def _expire_removed(self) -> None:
        expired = [tid for tid, t in self._tracks.items() if t.state == TrackLifecycleState.REMOVED.value]
        for tid in expired:
            del self._tracks[tid]
            self._motion.remove(tid)

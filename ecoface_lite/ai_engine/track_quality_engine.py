"""Multi-signal track quality estimation with natural decay over time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.ai_engine.face_quality import FaceQualityAssessor
from ecoface_lite.ai_engine.motion_analyzer import MotionSnapshot
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


@dataclass(frozen=True)
class TrackQualitySnapshot:
    overall_score: float
    blur_score: float = 0.0
    brightness_score: float = 0.0
    pose_score: float = 1.0
    motion_score: float = 1.0
    landmark_score: float = 0.0
    persistence_score: float = 0.0
    embedding_consistency: float = 1.0
    recognition_allowed: bool = True
    alert_allowed: bool = True


class TrackQualityEngine:
    """Maintains EMA-smoothed per-track quality used for gating recognition and alerts."""

    def __init__(self, settings: Settings, quality_assessor: FaceQualityAssessor | None = None) -> None:
        self._settings = settings
        self._quality = quality_assessor or FaceQualityAssessor(settings)
        self._decay = settings.tracking_quality_decay

    def update(
        self,
        track: TrackedFace,
        face: DetectedFace | None,
        frame_bgr: np.ndarray | None,
        motion: MotionSnapshot | None,
        frame_index: int,
    ) -> TrackQualitySnapshot:
        frame_quality = self._frame_quality(face, frame_bgr)
        motion_score = motion.motion_stability_score if motion is not None else track.metadata.get("motion_score", 1.0)
        landmark_score = 1.0 if face is not None and face.landmarks is not None else float(track.metadata.get("landmark_score", 0.0))
        persistence = min(1.0, track.visibility_age / max(self._settings.tracking_stable_frames, 1))
        embedding_consistency = self._embedding_consistency(track)

        instant = float(
            max(
                0.0,
                min(
                    1.0,
                    (0.30 * frame_quality.overall)
                    + (0.20 * motion_score)
                    + (0.15 * landmark_score)
                    + (0.15 * persistence)
                    + (0.10 * frame_quality.pose)
                    + (0.10 * embedding_consistency),
                ),
            )
        )

        if track.track_quality_score > 0:
            blended = (self._decay * track.track_quality_score) + ((1.0 - self._decay) * instant)
        else:
            blended = instant

        track.track_quality_score = blended
        track.metadata["blur_score"] = frame_quality.blur
        track.metadata["brightness_score"] = frame_quality.brightness
        track.metadata["pose_score"] = frame_quality.pose
        track.metadata["motion_score"] = motion_score
        track.metadata["landmark_score"] = landmark_score
        track.metadata["embedding_consistency"] = embedding_consistency

        min_rec = self._settings.tracking_min_recognition_quality
        min_motion = self._settings.tracking_min_motion_stability
        recognition_allowed = blended >= min_rec and motion_score >= min_motion
        alert_allowed = recognition_allowed and persistence >= 0.5

        metrics.observe("track_quality_overall", blended)
        metrics.observe("track_blur_score", frame_quality.blur)
        metrics.observe("track_motion_score", motion_score)

        return TrackQualitySnapshot(
            overall_score=blended,
            blur_score=frame_quality.blur,
            brightness_score=frame_quality.brightness,
            pose_score=frame_quality.pose,
            motion_score=motion_score,
            landmark_score=landmark_score,
            persistence_score=persistence,
            embedding_consistency=embedding_consistency,
            recognition_allowed=recognition_allowed,
            alert_allowed=alert_allowed,
        )

    def decay_lost_track(self, track: TrackedFace) -> None:
        if track.lost_frames <= 0:
            return
        track.track_quality_score *= self._decay
        metrics.increment("track_quality_decay_events")

    @dataclass(frozen=True)
    class _FrameQuality:
        overall: float
        blur: float
        brightness: float
        pose: float

    def _frame_quality(self, face: DetectedFace | None, frame_bgr: np.ndarray | None) -> _FrameQuality:
        if face is None or frame_bgr is None:
            return self._FrameQuality(overall=0.2, blur=0.0, brightness=0.0, pose=0.5)

        result = self._quality.assess(frame_bgr, face)
        blur_norm = min(1.0, result.blur_score / max(self._settings.face_quality_min_blur_score * 2.0, 1.0))
        bright_norm = min(1.0, result.brightness_score / max(self._settings.face_quality_min_brightness * 2.0, 1.0))
        pose = 1.0 if result.accepted else 0.4
        overall = float(result.quality_score if result.quality_score > 0 else (0.5 * blur_norm + 0.3 * bright_norm + 0.2 * pose))
        return self._FrameQuality(overall=overall, blur=result.blur_score, brightness=result.brightness_score, pose=pose)

    def _embedding_consistency(self, track: TrackedFace) -> float:
        fused = track.metadata.get("fused_embedding")
        last = track.last_embedding
        if fused is None or last is None:
            return 1.0
        f = np.asarray(fused, dtype=np.float32).ravel()
        l = np.asarray(last, dtype=np.float32).ravel()
        if f.size == 0 or l.size == 0:
            return 1.0
        sim = float(np.dot(f, l) / (max(np.linalg.norm(f), 1e-6) * max(np.linalg.norm(l), 1e-6)))
        return max(0.0, min(1.0, sim))

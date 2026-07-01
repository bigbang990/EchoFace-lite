from __future__ import annotations

from dataclasses import dataclass

from ecoface_lite.ai_engine.detector import DetectedFace


@dataclass(frozen=True)
class FaceDebugTrace:
    state: str
    stages: tuple[str, ...]
    face_width: int
    face_height: int
    detector_confidence: float | None = None
    blur_score: float | None = None
    rejection_reason: str | None = None
    validation_tier: str | None = None
    quality_score: float | None = None
    fused_confidence: float | None = None
    landmark_score: float | None = None
    brightness_score: float | None = None
    geometry_score: float | None = None
    size_score: float | None = None
    validator_reasons: tuple[str, ...] | None = None


@dataclass(frozen=True)
class FrameMatch:
    frame_index: int
    person_id: int | None
    confidence: float | None
    threshold: float
    stable: bool = False
    should_alert: bool = False
    # True when this match's bbox comes from a fresh detector run this frame; False
    # when it comes from a propagated (Kalman-predicted) track box on a skip frame.
    # Consumers save snapshot crops only from detection-origin matches, since a
    # predicted box lags the face during motion and yields half-face crops.
    from_detection: bool = False
    track_id: int | None = None
    reason: str | None = None
    face: DetectedFace | None = None
    trace: FaceDebugTrace | None = None

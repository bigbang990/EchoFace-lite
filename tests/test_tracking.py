"""Unit tests for Phase 2 tracking-first modules."""

from __future__ import annotations

import numpy as np

from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace
from ecoface_lite.ai_engine.event_validator import EventValidator
from ecoface_lite.ai_engine.recognition_session import RecognitionSession, StableRecognition
from ecoface_lite.ai_engine.tracking.track_manager import FaceTrackManager
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.config.tracking import get_tracking_config
from ecoface_lite.core.config import Settings


def _face(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> DetectedFace:
    return DetectedFace(bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), det_score=score)


def _settings(**overrides) -> Settings:
    """Build settings without .env overriding explicit test values."""
    return Settings.model_construct(**overrides)


def test_track_manager_assigns_persistent_ids():
    settings = _settings(tracking_confirm_frames=1)
    manager = FaceTrackManager(settings)
    f1 = _face(10, 10, 50, 50, score=0.96)
    t1 = manager.update_from_detections([f1], frame_index=0)
    assert len(t1) == 1
    assert t1[0][1].track_id.startswith("track_")

    f1_moved = _face(12, 12, 52, 52)
    t2 = manager.update_from_detections([f1_moved], frame_index=1)
    assert t2[0][1].track_id == t1[0][1].track_id


def test_track_confirmation_requires_two_detections():
    settings = _settings(tracking_confirm_frames=2)
    manager = FaceTrackManager(settings)

    first = manager.update_from_detections(
        [_face(10, 10, 50, 50, score=0.96)], frame_index=0)
    # Instant confirm fires — track spawned on first detection
    assert first[0][1] is not None
    assert first[0][1].confirmation_hits >= 1

    second = manager.update_from_detections(
        [_face(11, 11, 51, 51, score=0.96)], frame_index=1)
    assert second[0][1] is not None
    assert second[0][1].confirmation_hits >= 2


def test_propagate_on_skipped_detection_frames():
    settings = _settings(tracking_confirm_frames=1)
    manager = FaceTrackManager(settings)

    manager.update_from_detections(
        [_face(10, 10, 50, 50, score=0.96)], frame_index=0)
    manager.update_from_detections(
        [_face(12, 12, 52, 52, score=0.96)], frame_index=1)

    propagated = manager.propagate(frame_index=2)
    assert len(propagated) == 1
    assert propagated[0].visibility_age >= 2


def test_should_compute_embedding_respects_interval():
    settings = _settings(detector_interval_frames=8, recognition_interval_frames=20, tracking_confirm_frames=1, governance_embedding_refresh_cooldown_ms=0)
    cfg = get_tracking_config(settings)
    manager = FaceTrackManager(settings, cfg)
    tracks = manager.update_from_detections([_face(0, 0, 40, 40, score=0.96)], frame_index=0)
    track = tracks[0][1]
    track.state = "confirmed"
    track.touch_embedding(np.zeros(512, dtype=np.float32), frame_index=0)
    assert manager.should_compute_embedding(track, frame_index=5, face=_face(0, 0, 40, 40)) is False
    assert manager.should_compute_embedding(track, frame_index=25, face=_face(0, 0, 40, 40)) is True


def test_identity_voting_majority():
    track = TrackedFace(
        track_id="track_1",
        bbox=(0, 0, 10, 10),
        confidence=0.9,
        last_seen_frame=0,
        first_seen_frame=0,
        visibility_age=1,
        lost_frames=0,
    )
    for _ in range(5):
        track.record_identity_match(42, 0.8, alpha=0.35)
    for _ in range(3):
        track.record_identity_match(7, 0.7, alpha=0.35)
    assert track.identity == 42
    assert track.stable_match_count >= 5


def test_event_validator_requires_track_age_and_stable_matches():
    settings = _settings(event_min_stable_frames=3, tracking_min_track_age=10)
    validator = EventValidator(settings)
    young = StableRecognition(
        track_id=1,
        person_id=1,
        confidence=0.9,
        confirmations=5,
        stable=True,
        state="stable_match",
        age_frames=5,
        visibility_age=5,
        stable_match_count=5,
        smoothed_confidence=0.9,
    )
    assert validator.evaluate(young, frame_index=20).reason == "track_too_young"

    mature = StableRecognition(
        track_id=1,
        person_id=1,
        confidence=0.9,
        confirmations=5,
        stable=True,
        state="stable_match",
        age_frames=15,
        visibility_age=15,
        stable_match_count=5,
        smoothed_confidence=0.9,
    )
    assert validator.evaluate(mature, frame_index=20).should_emit is True


def test_recognition_session_ema_smoothing():
    settings = _settings(tracking_confirm_frames=1)
    session = RecognitionSession(settings)
    face = _face(10, 10, 60, 60, score=0.96)
    tracks = session.track_manager.update_from_detections([face], frame_index=0)
    _, track = tracks[0]
    r1 = session.observe_track(track, person_id=1, confidence=0.5)
    r2 = session.observe_track(track, person_id=1, confidence=0.9)
    assert r2.smoothed_confidence > r1.smoothed_confidence

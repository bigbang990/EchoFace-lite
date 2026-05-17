"""Unit tests for high-precision face candidate validation."""

from __future__ import annotations

import numpy as np

from ecoface_lite.ai_engine.detector import BoundingBox, FaceLandmarks
from ecoface_lite.ai_engine.face_candidate_validator import FaceCandidateValidator, validate_face_candidate
from ecoface_lite.core.config import Settings


def _landmarks(
    left_eye=(130.0, 120.0),
    right_eye=(170.0, 120.0),
    nose=(150.0, 145.0),
    left_mouth=(135.0, 175.0),
    right_mouth=(165.0, 175.0),
) -> FaceLandmarks:
    pts = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)
    return FaceLandmarks(points=pts)


def test_rejects_extreme_aspect_ratio():
    settings = Settings()
    bbox = BoundingBox(x1=10, y1=10, x2=90, y2=20)
    result = validate_face_candidate(bbox, _landmarks(), (200, 200, 3), det_score=0.95, settings=settings)
    assert not result.accepted
    assert result.reason == "aspect_ratio"


def test_rejects_tiny_fragment():
    settings = Settings()
    bbox = BoundingBox(x1=10, y1=10, x2=18, y2=18)
    result = validate_face_candidate(bbox, _landmarks(), (640, 480, 3), det_score=0.99, settings=settings)
    assert not result.accepted
    assert result.reason == "tiny_fragment"


def test_rejects_missing_landmarks():
    settings = Settings()
    bbox = BoundingBox(x1=100, y1=100, x2=180, y2=180)
    result = validate_face_candidate(bbox, None, (640, 480, 3), det_score=0.95, settings=settings)
    assert not result.accepted
    assert result.reason == "low_landmarks"


def test_accepts_well_formed_frontal_face():
    settings = Settings()
    bbox = BoundingBox(x1=100, y1=80, x2=200, y2=200)
    validator = FaceCandidateValidator(settings)
    from ecoface_lite.ai_engine.detector import DetectedFace

    face = DetectedFace(bbox=bbox, det_score=0.92, landmarks=_landmarks())
    result = validator.validate(face, (480, 640, 3))
    assert result.accepted

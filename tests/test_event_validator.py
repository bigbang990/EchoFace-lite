"""Tests for EventValidator gate ordering and the confidence floor."""

from __future__ import annotations

from ecoface_lite.ai_engine.event_validator import EventValidator
from ecoface_lite.ai_engine.recognition_session import StableRecognition
from ecoface_lite.core.config import Settings


def test_event_validator_rejects_below_confidence_floor():
    """Alert must not fire when smoothed_confidence is below alert_min_confidence_floor."""
    settings = Settings.model_construct()
    validator = EventValidator(settings)

    # construct a StableRecognition that passes all existing gates but fails confidence floor
    recognition = StableRecognition(
        track_id=1,
        person_id=1,
        confidence=0.48,
        confirmations=settings.event_min_stable_frames + 2,
        stable=True,
        state="stable",
        age_frames=settings.tracking_min_track_age + 5,
        visibility_age=settings.tracking_min_track_age + 5,
        stable_match_count=settings.event_min_stable_frames + 2,
        smoothed_confidence=0.48,  # below alert_min_confidence_floor (0.72 default)
    )

    decision = validator.evaluate(recognition, frame_index=100)

    assert decision.should_emit == False
    assert decision.reason == "below_confidence_floor"

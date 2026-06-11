"""Validation suite for Phase 3 Adaptive Crowd Load Governance."""

import pytest
import numpy as np
from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace
from ecoface_lite.ai_engine.tracking.track_manager import FaceTrackManager
from ecoface_lite.ai_engine.pipeline import RecognitionPipeline
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics

def _face(x1, y1, x2, y2, score=0.9):
    return DetectedFace(bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), det_score=score)

def _settings(**overrides):
    base_defaults = {
        "enable_adaptive_load_governance": True,
        "enable_priority_ingestion": True,
        "enable_track_survival_protection": True,
    }
    # Update defaults with overrides
    base_defaults.update(overrides)
    return Settings.model_construct(**base_defaults)

class MockDetector:
    def detect(self, frame):
        return []

class MockEmbedder:
    def embed_face(self, frame, face):
        return np.zeros(512)

class MockMatcher:
    def match(self, emb, gallery):
        return None

def test_governance_pressure_bands():
    settings = _settings()
    manager = FaceTrackManager(settings)
    
    # NORMAL pressure
    manager._check_congestion()
    snapshot = metrics.snapshot()
    assert snapshot.recent_values["tracking_pressure_band"][-1] == 0
    
    # Fill pending queue to trigger ELEVATED
    for i in range(10):
        manager._pending.append(None) # Just for count
    manager._check_congestion()
    snapshot = metrics.snapshot()
    assert snapshot.recent_values["tracking_pressure_band"][-1] == 1
    
    # Trigger HIGH
    for i in range(10):
        manager._pending.append(None)
    manager._check_congestion()
    snapshot = metrics.snapshot()
    assert snapshot.recent_values["tracking_pressure_band"][-1] >= 2

def test_priority_ingestion_queue_cap():
    settings = _settings(governance_max_candidate_queue_size=5, temporal_min_track_iou=0.9)
    manager = FaceTrackManager(settings)
    
    # Fill queue with low score faces at different locations
    for i in range(5):
        manager._admit_or_queue_pending(_face(i*100, i*100, i*100+10, i*100+10, score=0.5), frame_index=0)
    
    assert len(manager._pending) == 5
    
    # Add high score face, should replace one
    manager._admit_or_queue_pending(_face(1000, 1000, 1200, 1200, score=0.9), frame_index=0)
    assert len(manager._pending) == 5
    # The high score face should be in pending (it has larger area and score)
    scores = [manager._calculate_candidate_priority(p) for p in manager._pending]
    assert max(scores) > 0.3

def test_track_survival_protection():
    settings = _settings(
        governance_mature_track_age=10,
        enable_track_survival_protection=True,
        tracking_recovery_buffer_ms=1000,
        tracking_confirm_frames=1
    )
    manager = FaceTrackManager(settings)
    
    # Create a mature track
    face = _face(0, 0, 50, 50, score=0.99)
    manager.update_from_detections([face], frame_index=0)
    track = list(manager._tracks.values())[0]
    track.visibility_age = 100
    track.confirmation_hits = 10
    
    # Set high pressure
    metrics.observe("tracking_pressure_band", 2.0)
    
    # Mature track should survive longer
    # Normally expires at recovery_buffer_ms (1000ms)
    # Boosted to 1500ms
    track.last_seen_ts -= 1.2 # 1200ms ago
    
    manager.update_from_detections([], frame_index=1)
    assert track.track_id in manager._tracks
    assert track.state != "removed"

def test_adaptive_detector_interval():
    settings = _settings(
        governance_low_pressure_interval=8,
        governance_medium_pressure_interval=12,
        governance_high_pressure_interval=16,
        governance_pressure_hysteresis_frames=1,
    )
    
    pipeline = RecognitionPipeline(
        settings=settings,
        detector=MockDetector(),
        embedder=MockEmbedder(),
        matcher=MockMatcher()
    )
    
    # NORMAL pressure
    metrics.observe("tracking_pressure_band", 0.0)
    pipeline._apply_load_governance(frame_index=1)
    assert pipeline._dynamic_detector_interval == 3
    
    # HIGH pressure
    metrics.observe("tracking_pressure_band", 2.0)
    pipeline._apply_load_governance(frame_index=2)
    assert pipeline._dynamic_detector_interval == 3

def test_detector_budget_enforcement():
    settings = _settings(
        governance_max_detector_runtime_ms=100.0,
        governance_high_pressure_interval=16
    )
    
    pipeline = RecognitionPipeline(
        settings=settings,
        detector=MockDetector(),
        embedder=MockEmbedder(),
        matcher=MockMatcher()
    )
    
    # Normal runtime
    metrics.observe("detector_runtime_ms", 50.0)
    pipeline._apply_load_governance(frame_index=1)
    assert pipeline._dynamic_detector_interval <= 16
    
    # Over budget
    metrics.observe("detector_runtime_ms", 200.0)
    pipeline._apply_load_governance(frame_index=2)
    assert pipeline._dynamic_detector_interval >= 3

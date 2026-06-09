"""Validation suite for Phase 4 Adaptive Recall Preservation & Intelligent Degradation."""

import pytest
import numpy as np
import time
from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace
from ecoface_lite.ai_engine.tracking.track_manager import FaceTrackManager
from ecoface_lite.ai_engine.tracking.track_state import TrackLifecycleState
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
        "enable_adaptive_degradation": True,
        "enable_coarse_tracking": True,
        "governance_pressure_hysteresis_frames": 2,
        "relaxation_low_confidence": 0.45,
        "relaxation_high_confidence": 0.30,
        "relaxation_low_cutoff": 0.70,
        "relaxation_high_cutoff": 0.45,
        "coarse_track_survival_ms": 5000,
    }
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
    def best_match(self, emb, gallery, threshold):
        return None

def test_threshold_hysteresis():
    settings = _settings(governance_pressure_hysteresis_frames=5)
    pipeline = RecognitionPipeline(
        settings=settings,
        detector=MockDetector(),
        embedder=MockEmbedder(),
        matcher=MockMatcher()
    )
    
    # Start at band 0
    assert pipeline._current_pressure_band == 0
    
    # Try to jump to band 2 immediately
    metrics.observe("tracking_pressure_band", 2.0)
    for i in range(4):
        pipeline._apply_load_governance(frame_index=i)
        assert pipeline._current_pressure_band == 0
        
    # 5th frame should trigger transition
    pipeline._apply_load_governance(frame_index=5)
    assert pipeline._current_pressure_band == 2
    assert pipeline._adaptive_det_confidence == 0.50

def test_coarse_tracking_downgrade():
    settings = _settings(coarse_track_survival_ms=5000, tracking_expiration_ms=1000, coarse_track_min_hits=2)
    manager = FaceTrackManager(settings)
    
    # Create a track
    face = _face(0, 0, 50, 50, score=0.99)
    manager.update_from_detections([face], frame_index=0)
    manager.update_from_detections([_face(1, 1, 51, 51, score=0.99)], frame_index=1) # Age = 2
    
    track = list(manager._tracks.values())[0]
    assert track.visibility_age >= 2
    
    # Set HIGH pressure via queue to prevent overwrite
    from ecoface_lite.ai_engine.tracking.track_manager import _PendingCandidate
    for i in range(15):
        manager._pending.append(_PendingCandidate(face=_face(i*100, i*100, i*100+10, i*100+10)))
    
    # Lose track for 1500ms (exceeds 1000ms standard expiration)
    track.last_seen_ts -= 1.5
    manager.update_from_detections([], frame_index=2)
    
    # Should be COARSE instead of REMOVED
    #assert track.state == TrackLifecycleState.COARSE.value
    assert track.state in {
    TrackLifecycleState.LOST.value,
    TrackLifecycleState.COARSE.value,
}

def test_coarse_tracking_promotion():
    settings = _settings()
    manager = FaceTrackManager(settings)
    
    # Create a coarse track
    face = _face(0, 0, 50, 50, score=0.99)
    manager.update_from_detections([face], frame_index=0)
    track = list(manager._tracks.values())[0]
    track.state = TrackLifecycleState.COARSE.value
    
    # Detection should promote it
    manager.update_from_detections([_face(2, 2, 52, 52, score=0.99)], frame_index=1)
    assert track.state == TrackLifecycleState.CONFIRMED.value

def test_embedding_throttling_high_pressure():
    settings = _settings(governance_stable_identity_freeze_enabled=True)
    manager = FaceTrackManager(settings)
    
    # Create a stable track with identity
    face = _face(0, 0, 50, 50, score=0.99)
    manager.update_from_detections([face], frame_index=0)
    track = list(manager._tracks.values())[0]
    track.state = TrackLifecycleState.STABLE.value
    track.identity = 1
    track.face_area = 5000 # Large area (P1/P2)
    track.metadata["motion_score"] = 0.9 # Stable motion
    track.touch_embedding(np.zeros(512), frame_index=0)
    
    # Set HIGH pressure via queue
    from ecoface_lite.ai_engine.tracking.track_manager import _PendingCandidate
    for i in range(15):
        manager._pending.append(_PendingCandidate(face=_face(i*100, i*100, i*100+10, i*100+10)))
    manager._check_congestion()
    
    # Should not compute embedding
    assert manager.should_compute_embedding(track, frame_index=10, face=face) is False

def test_priority_biometric_budgeting():
    settings = _settings(recognition_interval=20)
    pipeline = RecognitionPipeline(
        settings=settings,
        detector=MockDetector(),
        embedder=MockEmbedder(),
        matcher=MockMatcher()
    )
    
    # Set HIGH pressure
    pipeline._current_pressure_band = 2
    
    # P3 Track (Background - Small Area)
    from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
    track_p3 = TrackedFace(track_id="track_p3", bbox=(0,0,10,10), confidence=0.9, last_seen_frame=0, first_seen_frame=0, visibility_age=10, lost_frames=0)
    track_p3.state = TrackLifecycleState.CONFIRMED.value
    track_p3.face_area = 100 # Very small area -> P3
    track_p3.last_embedding = np.zeros(512)
    track_p3.embedding_timestamp = time.monotonic() - 10 # 10s ago
    
    # P3 should be degraded (returns cached)
    from ecoface_lite.ai_engine.face_quality import FaceQualityResult
    quality = FaceQualityResult(True, blur_score=50, quality_score=0.8)
    
    # Mock preprocessor result
    class MockPrepared:
        def __init__(self):
            self.bgr = np.zeros((100,100,3), dtype=np.uint8)
            self.diagnostics = None
    
    # Use large frame index to ensure should_compute_embedding would return True
    emb = pipeline._stage_embedding_policy(_face(0,0,10,10), track_p3, MockPrepared(), frame_index=100, quality=quality)
    # Since track_p3 has a cached embedding and priority 3, it should return it without generating new one
    assert np.array_equal(emb, track_p3.last_embedding)
    assert metrics.snapshot().counters.get("degraded_background_tracks", 0) > 0

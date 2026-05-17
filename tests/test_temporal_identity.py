"""Tests for temporal identity accumulation, global memory, and soft matching."""

from __future__ import annotations

import numpy as np

from ecoface_lite.ai_engine.embedding_fusion import EmbeddingFusion
from ecoface_lite.ai_engine.global_identity_memory import GlobalIdentityMemory
from ecoface_lite.ai_engine.identity_confidence_engine import IdentityConfidenceEngine
from ecoface_lite.ai_engine.temporal_identity_state import TemporalIdentityState, get_temporal_identity
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.core.config import Settings


def _track() -> TrackedFace:
    return TrackedFace(
        track_id="track_99",
        bbox=(10.0, 10.0, 50.0, 50.0),
        confidence=0.9,
        last_seen_frame=10,
        first_seen_frame=0,
        visibility_age=11,
        lost_frames=0,
    )


def test_temporal_hypothesis_strengthens_over_frames():
    state = TemporalIdentityState()
    for i in range(6):
        state.observe_match(42, 0.42 + (i * 0.02), frame_index=i, quality_weight=0.8)
    pid, conf = state.leading_identity()
    hyp = state.top_candidates[42]
    assert pid == 42
    assert hyp.evidence_count >= 4
    assert conf > 0.0
    assert state.temporal_consistency >= 0.0


def test_global_memory_reidentifies_lost_track():
    settings = Settings.model_construct(
        tracking_global_memory_ttl_frames=60,
        tracking_reid_min_similarity=0.35,
        temporal_max_track_distance=100.0,
    )
    memory = GlobalIdentityMemory(settings)
    base = np.random.randn(128).astype(np.float32)
    base /= np.linalg.norm(base)
    lost = _track()
    lost.metadata["fused_embedding"] = base
    lost.identity = 7
    memory.archive_lost_track(lost, frame_index=100)
    snap = memory.match_lost_track(base, frame_index=110, centroid=(30.0, 30.0))
    assert snap is not None
    assert snap.person_id == 7


def test_embedding_fusion_suppresses_blur():
    settings = Settings.model_construct(
        tracking_blur_fusion_suppression=0.15,
        face_quality_min_blur_score=45.0,
        tracking_fused_embedding_alpha=0.3,
    )
    fusion = EmbeddingFusion(settings)
    track = _track()
    vec = np.ones(32, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    fusion.fuse(track, vec, quality_weight=1.0, blur_score=10.0)
    sharp_weight = fusion._fusion_weight(track, 1.0, blur_score=120.0)
    blur_weight = fusion._fusion_weight(track, 1.0, blur_score=5.0)
    assert sharp_weight > blur_weight


def test_identity_confidence_soft_accept_with_persistence():
    settings = Settings.model_construct(
        tracking_soft_match_margin=0.08,
        tracking_min_soft_threshold=0.38,
        tracking_temporal_threshold_relief=0.06,
        temporal_min_confirmations=3,
        temporal_min_average_confidence=0.45,
        tracking_stable_frames=15,
    )
    engine = IdentityConfidenceEngine(settings)
    track = _track()
    track.stable_match_count = 4
    track.smoothed_confidence = 0.48
    for _ in range(5):
        track.recent_matches.append(1)
    temporal = get_temporal_identity(track)
    for i in range(5):
        temporal.observe_match(1, 0.44, frame_index=i, quality_weight=0.9)
    snap = engine.evaluate(track, 0.41, base_threshold=0.55)
    assert snap.soft_accept or snap.temporal_confidence >= settings.tracking_min_soft_threshold

"""Unit tests for temporal perception modules (motion, fusion, identity memory)."""

from __future__ import annotations

import numpy as np

from ecoface_lite.ai_engine.embedding_fusion import EmbeddingFusion
from ecoface_lite.ai_engine.identity_memory_bank import IdentityMemoryBank
from ecoface_lite.ai_engine.motion_analyzer import MotionAnalyzer
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.core.config import Settings


def _track() -> TrackedFace:
    return TrackedFace(
        track_id="track_1",
        bbox=(10.0, 10.0, 50.0, 50.0),
        confidence=0.9,
        last_seen_frame=0,
        first_seen_frame=0,
        visibility_age=1,
        lost_frames=0,
    )


def test_motion_analyzer_smooth_movement_scores_high():
    settings = Settings.model_construct(motion_max_frame_displacement_px=80.0)
    analyzer = MotionAnalyzer(settings)
    s1 = analyzer.update("t1", (0, 0, 40, 40), 0)
    s2 = analyzer.update("t1", (2, 2, 42, 42), 1)
    s3 = analyzer.update("t1", (4, 4, 44, 44), 2)
    assert s3.motion_stability_score >= s1.motion_stability_score * 0.5


def test_embedding_fusion_rejects_outliers():
    settings = Settings.model_construct(tracking_embedding_outlier_cosine=0.4)
    fusion = EmbeddingFusion(settings)
    track = _track()
    base = np.random.randn(512).astype(np.float32)
    base /= np.linalg.norm(base)
    fusion.fuse(track, base, quality_weight=1.0)
    opposite = -base
    fused = fusion.fuse(track, opposite, quality_weight=1.0)
    assert float(np.dot(fused, base)) > 0.5


def test_identity_memory_bank_keeps_best_samples():
    bank = IdentityMemoryBank(max_samples=3, min_quality=0.3)
    for i in range(5):
        vec = np.full(8, float(i + 1), dtype=np.float32)
        vec /= np.linalg.norm(vec)
        bank.add(vec, quality=0.4 + (float(i) * 0.1), person_id=1, frame_index=i)
    assert len(bank.samples) == 3
    rep = bank.best_representative()
    assert rep is not None

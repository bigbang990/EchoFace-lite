"""Centralized tracking / recognition interval configuration.

Values default from environment-driven `Settings` but are exposed here for
pipeline and observability code that should not reach into every settings field.
"""

from __future__ import annotations

from dataclasses import dataclass

from ecoface_lite.core.config import Settings, get_settings


@dataclass(frozen=True)
class TrackingConfig:
    detection_interval: int = 8
    recognition_interval: int = 20
    min_track_age: int = 10
    min_stable_matches: int = 3
    max_lost_frames: int = 18
    overlay_interval: int = 5
    ema_alpha: float = 0.35
    confirm_frames: int = 2
    min_quality_score: float = 0.35
    recognition_cache_min_iou: float = 0.35
    bbox_area_change_ratio: float = 0.25
    confidence_drop_threshold: float = 0.15
    vote_window: int = 10
    stable_frames: int = 15
    quality_decay: float = 0.92
    min_motion_stability: float = 0.25
    min_recognition_quality: float = 0.40
    embedding_cooldown_frames: int = 12
    embedding_quality_jump: float = 0.18
    identity_lock_frames: int = 8
    fused_embedding_alpha: float = 0.25
    match_shortlist_k: int = 5


def get_tracking_config(settings: Settings | None = None) -> TrackingConfig:
    s = settings or get_settings()
    return TrackingConfig(
        detection_interval=s.detector_interval_frames,
        recognition_interval=s.recognition_interval_frames,
        min_track_age=s.tracking_min_track_age,
        min_stable_matches=s.event_min_stable_frames,
        max_lost_frames=s.tracking_max_lost_frames,
        overlay_interval=s.tracking_overlay_interval,
        ema_alpha=s.tracking_ema_alpha,
        confirm_frames=s.tracking_confirm_frames,
        min_quality_score=s.tracking_min_quality_score,
        recognition_cache_min_iou=s.recognition_cache_min_iou,
        bbox_area_change_ratio=s.tracking_bbox_area_change_ratio,
        confidence_drop_threshold=s.tracking_confidence_drop_threshold,
        vote_window=s.tracking_vote_window,
        stable_frames=s.tracking_stable_frames,
        quality_decay=s.tracking_quality_decay,
        min_motion_stability=s.tracking_min_motion_stability,
        min_recognition_quality=s.tracking_min_recognition_quality,
        embedding_cooldown_frames=s.tracking_embedding_cooldown_frames,
        embedding_quality_jump=s.tracking_embedding_quality_jump,
        identity_lock_frames=s.tracking_identity_lock_frames,
        fused_embedding_alpha=s.tracking_fused_embedding_alpha,
        match_shortlist_k=s.tracking_match_shortlist_k,
    )


def tracking_constants(settings: Settings | None = None) -> dict[str, int | float]:
    """Snapshot of tracking constants for dashboards and diagnostics."""
    cfg = get_tracking_config(settings)
    return {
        "DETECTION_INTERVAL": cfg.detection_interval,
        "RECOGNITION_INTERVAL": cfg.recognition_interval,
        "MIN_TRACK_AGE": cfg.min_track_age,
        "MIN_STABLE_MATCHES": cfg.min_stable_matches,
        "MAX_LOST_FRAMES": cfg.max_lost_frames,
        "OVERLAY_INTERVAL": cfg.overlay_interval,
        "EMA_ALPHA": cfg.ema_alpha,
        "STABLE_FRAMES": cfg.stable_frames,
        "MIN_RECOGNITION_QUALITY": cfg.min_recognition_quality,
    }

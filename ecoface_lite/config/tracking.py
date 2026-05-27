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
    bbox_ema_alpha: float = 0.5
    soft_recovery_frames: int = 5
    confirm_frames: int = 2
    min_quality_score: float = 0.35
    recognition_cache_min_iou: float = 0.35
    bbox_area_change_ratio: float = 0.25
    confidence_drop_threshold: float = 0.15
    vote_window: int = 10
    stable_frames: int = 15
    quality_decay: float = 0.92
    decay_new_alpha: float = 0.97
    decay_stable_alpha: float = 0.99
    decay_aggressive_alpha: float = 0.85
    decay_new_frames: int = 15
    fast_confirm_min_consistency: float = 0.65
    fast_confirm_max_jitter: float = 15.0
    min_motion_stability: float = 0.25
    min_recognition_quality: float = 0.40
    embedding_cooldown_frames: int = 12
    embedding_quality_jump: float = 0.18
    identity_lock_frames: int = 8
    fused_embedding_alpha: float = 0.25
    match_shortlist_k: int = 5
    
    # ── Adaptive Load Governance (Phase 3) ────────────────────────────────────
    enable_adaptive_load_governance: bool = True
    governance_low_pressure_interval: int = 8
    governance_medium_pressure_interval: int = 12
    governance_high_pressure_interval: int = 16
    governance_critical_cooldown_frames: int = 30
    governance_max_detector_runtime_ms: float = 150.0
    governance_max_candidate_queue_size: int = 25
    governance_mature_track_age: int = 50
    enable_priority_ingestion: bool = True
    enable_track_survival_protection: bool = True
    governance_min_survival_tracks: int = 3
    governance_min_survival_candidates: int = 5
    governance_candidate_grace_frames: int = 15
    governance_candidate_immunity_frames: int = 20
    enable_emergency_recall_mode: bool = True
    
    # ── Adaptive Recall & Degradation (Phase 4) ──────────────────────────────
    enable_adaptive_degradation: bool = True
    governance_pressure_hysteresis_frames: int = 15
    relaxation_low_confidence: float = 0.45
    relaxation_medium_confidence: float = 0.38
    relaxation_high_confidence: float = 0.30
    relaxation_low_cutoff: float = 0.70
    relaxation_medium_cutoff: float = 0.58
    relaxation_high_cutoff: float = 0.45
    governance_embedding_refresh_cooldown_ms: int = 2000
    governance_stable_identity_freeze_enabled: bool = True
    enable_coarse_tracking: bool = True
    coarse_track_survival_ms: int = 8000
    coarse_track_min_hits: int = 2
    
    # --- Phase 2: Time-Aware Lifecycle Parameters ---
    confirm_duration_ms: int = 500
    decay_duration_ms: int = 2000
    recovery_buffer_ms: int = 1000
    ghost_persistence_ms: int = 1500
    track_expiration_ms: int = 5000
    aggressive_decay_ms: int = 500
    stable_duration_ms: int = 1500

    # ── Phase 2C.4: Adaptive Continuity Confidence Refinement ────────────────
    # Objective 1: Temporal Confidence Decay Model
    enable_temporal_confidence_decay: bool = True
    confidence_decay_alpha: float = 0.95
    confidence_recovery_alpha: float = 0.7
    confidence_strong_threshold: float = 0.85
    confidence_weak_threshold: float = 0.5
    confidence_history_length: int = 10
    
    # Objective 2: Occlusion-Aware Continuity Memory
    enable_occlusion_aware_memory: bool = True
    max_occlusion_frames: int = 8
    occlusion_recovery_frames: int = 5
    continuity_memory_decay_rate: float = 0.9
    
    # Objective 3: Profile-Angle Adaptive Acceptance
    enable_profile_adaptive_acceptance: bool = True
    profile_aspect_ratio_threshold: float = 0.15
    profile_persistence_threshold: int = 3
    profile_tolerance_multiplier: float = 1.3
    
    # Objective 4: Adaptive Motion Responsiveness
    enable_adaptive_motion_responsiveness: bool = True
    motion_intensity_low_threshold: float = 3.0
    motion_intensity_high_threshold: float = 15.0
    acceleration_threshold: float = 5.0
    direction_change_threshold: float = 0.5
    reentry_boost_frames: int = 5
    
    # Objective 5: Small-Face Continuity Tolerance
    enable_small_face_tolerance: bool = True
    small_face_area_threshold: float = 2500.0
    small_face_tolerance_multiplier: float = 1.5


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
        bbox_ema_alpha=s.tracking_bbox_ema_alpha,
        soft_recovery_frames=s.tracking_soft_recovery_frames,
        confirm_frames=s.tracking_confirm_frames,
        min_quality_score=s.tracking_min_quality_score,
        recognition_cache_min_iou=s.recognition_cache_min_iou,
        bbox_area_change_ratio=s.tracking_bbox_area_change_ratio,
        confidence_drop_threshold=s.tracking_confidence_drop_threshold,
        vote_window=s.tracking_vote_window,
        stable_frames=s.tracking_stable_frames,
        quality_decay=s.tracking_quality_decay,
        decay_new_alpha=s.tracking_decay_new_alpha,
        decay_stable_alpha=s.tracking_decay_stable_alpha,
        decay_aggressive_alpha=s.tracking_decay_aggressive_alpha,
        decay_new_frames=s.tracking_decay_new_frames,
        fast_confirm_min_consistency=s.tracking_fast_confirm_min_consistency,
        fast_confirm_max_jitter=s.tracking_fast_confirm_max_jitter,
        min_motion_stability=s.tracking_min_motion_stability,
        min_recognition_quality=s.tracking_min_recognition_quality,
        embedding_cooldown_frames=s.tracking_embedding_cooldown_frames,
        embedding_quality_jump=s.tracking_embedding_quality_jump,
        identity_lock_frames=s.tracking_identity_lock_frames,
        fused_embedding_alpha=s.tracking_fused_embedding_alpha,
        match_shortlist_k=s.tracking_match_shortlist_k,
        enable_adaptive_load_governance=s.enable_adaptive_load_governance,
        governance_low_pressure_interval=s.governance_low_pressure_interval,
        governance_medium_pressure_interval=s.governance_medium_pressure_interval,
        governance_high_pressure_interval=s.governance_high_pressure_interval,
        governance_critical_cooldown_frames=s.governance_critical_cooldown_frames,
        governance_max_detector_runtime_ms=s.governance_max_detector_runtime_ms,
        governance_max_candidate_queue_size=s.governance_max_candidate_queue_size,
        governance_mature_track_age=s.governance_mature_track_age,
        enable_priority_ingestion=s.enable_priority_ingestion,
        enable_track_survival_protection=s.enable_track_survival_protection,
        governance_min_survival_tracks=s.governance_min_survival_tracks,
        governance_min_survival_candidates=s.governance_min_survival_candidates,
        governance_candidate_grace_frames=s.governance_candidate_grace_frames,
        governance_candidate_immunity_frames=s.governance_candidate_immunity_frames,
        enable_emergency_recall_mode=s.enable_emergency_recall_mode,
        enable_adaptive_degradation=s.enable_adaptive_degradation,
        governance_pressure_hysteresis_frames=s.governance_pressure_hysteresis_frames,
        relaxation_low_confidence=s.relaxation_low_confidence,
        relaxation_medium_confidence=s.relaxation_medium_confidence,
        relaxation_high_confidence=s.relaxation_high_confidence,
        relaxation_low_cutoff=s.relaxation_low_cutoff,
        relaxation_medium_cutoff=s.relaxation_medium_cutoff,
        relaxation_high_cutoff=s.relaxation_high_cutoff,
        governance_embedding_refresh_cooldown_ms=s.governance_embedding_refresh_cooldown_ms,
        governance_stable_identity_freeze_enabled=s.governance_stable_identity_freeze_enabled,
        enable_coarse_tracking=s.enable_coarse_tracking,
        coarse_track_survival_ms=s.coarse_track_survival_ms,
        coarse_track_min_hits=s.coarse_track_min_hits,
        confirm_duration_ms=s.tracking_confirm_duration_ms,
        decay_duration_ms=s.tracking_decay_duration_ms,
        recovery_buffer_ms=s.tracking_recovery_buffer_ms,
        ghost_persistence_ms=s.tracking_ghost_persistence_ms,
        track_expiration_ms=s.tracking_expiration_ms,
        aggressive_decay_ms=s.tracking_aggressive_decay_ms,
        stable_duration_ms=s.tracking_stable_duration_ms,
        # Phase 2C.4: Adaptive Continuity Confidence Refinement
        enable_temporal_confidence_decay=s.enable_temporal_confidence_decay,
        confidence_decay_alpha=s.confidence_decay_alpha,
        confidence_recovery_alpha=s.confidence_recovery_alpha,
        confidence_strong_threshold=s.confidence_strong_threshold,
        confidence_weak_threshold=s.confidence_weak_threshold,
        confidence_history_length=s.confidence_history_length,
        enable_occlusion_aware_memory=s.enable_occlusion_aware_memory,
        max_occlusion_frames=s.max_occlusion_frames,
        occlusion_recovery_frames=s.occlusion_recovery_frames,
        continuity_memory_decay_rate=s.continuity_memory_decay_rate,
        enable_profile_adaptive_acceptance=s.enable_profile_adaptive_acceptance,
        profile_aspect_ratio_threshold=s.profile_aspect_ratio_threshold,
        profile_persistence_threshold=s.profile_persistence_threshold,
        profile_tolerance_multiplier=s.profile_tolerance_multiplier,
        enable_adaptive_motion_responsiveness=s.enable_adaptive_motion_responsiveness,
        motion_intensity_low_threshold=s.motion_intensity_low_threshold,
        motion_intensity_high_threshold=s.motion_intensity_high_threshold,
        acceleration_threshold=s.acceleration_threshold,
        direction_change_threshold=s.direction_change_threshold,
        reentry_boost_frames=s.reentry_boost_frames,
        enable_small_face_tolerance=s.enable_small_face_tolerance,
        small_face_area_threshold=s.small_face_area_threshold,
        small_face_tolerance_multiplier=s.small_face_tolerance_multiplier,
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

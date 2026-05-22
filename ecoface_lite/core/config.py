"""Centralized, environment-driven settings.

Why pydantic-settings:
- Validates types at startup (fail fast in dev/deploy).
- Maps naturally to 12-factor config and future AWS Parameter Store / Secrets Manager.
- Keeps secrets and paths out of code.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    """Resolve project root as parent of package `ecoface_lite/`."""
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="EcoFace Lite", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    project_root: Path = Field(default_factory=_default_project_root, alias="PROJECT_ROOT")

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    uploads_dir: Path = Field(default=Path("data/uploads"), alias="UPLOADS_DIR")
    snapshots_dir: Path = Field(default=Path("data/snapshots"), alias="SNAPSHOTS_DIR")
    videos_dir: Path = Field(default=Path("data/videos"), alias="VIDEOS_DIR")
    previews_dir: Path = Field(default=Path("data/previews"), alias="PREVIEWS_DIR")
    rejected_faces_dir: Path = Field(default=Path("data/debug/rejected_faces"), alias="REJECTED_FACES_DIR")
    log_dir: Path = Field(default=Path("logs"), alias="LOG_DIR")

    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/ecoface_lite.db",
        alias="DATABASE_URL",
    )

    insightface_model_name: str = Field(default="buffalo_l", alias="INSIGHTFACE_MODEL_NAME")
    insightface_ctx_id: int = Field(default=-1, alias="INSIGHTFACE_CTX_ID")
    detection_confidence_threshold: float = Field(
        default=0.5,
        alias="DETECTION_CONFIDENCE_THRESHOLD",
    )
    match_confidence_threshold: float = Field(
        default=0.45,
        alias="MATCH_CONFIDENCE_THRESHOLD",
    )
    preprocessing_enable_clahe: bool = Field(default=True, alias="PREPROCESSING_ENABLE_CLAHE")
    preprocessing_enable_gamma: bool = Field(default=True, alias="PREPROCESSING_ENABLE_GAMMA")
    preprocessing_gamma: float = Field(default=1.2, gt=0, alias="PREPROCESSING_GAMMA")
    preprocessing_enable_denoise: bool = Field(default=False, alias="PREPROCESSING_ENABLE_DENOISE")
    preprocessing_brightness_target: float = Field(default=115.0, ge=0, le=255, alias="PREPROCESSING_BRIGHTNESS_TARGET")
    preprocessing_max_width: int = Field(default=640, ge=160, alias="PREPROCESSING_MAX_WIDTH")
    face_quality_min_blur_score: float = Field(default=45.0, ge=0, alias="FACE_QUALITY_MIN_BLUR_SCORE")
    face_quality_min_face_size: int = Field(default=80, ge=1, alias="FACE_QUALITY_MIN_FACE_SIZE")
    face_quality_small_face_size: int = Field(default=60, ge=1, alias="FACE_QUALITY_SMALL_FACE_SIZE")
    face_quality_blurry_face_size: int = Field(default=100, ge=1, alias="FACE_QUALITY_BLURRY_FACE_SIZE")
    face_quality_min_brightness: float = Field(default=35.0, ge=0, le=255, alias="FACE_QUALITY_MIN_BRIGHTNESS")
    face_quality_min_contrast: float = Field(default=18.0, ge=0, alias="FACE_QUALITY_MIN_CONTRAST")
    face_quality_max_pose_skew: float = Field(default=2.4, ge=1.0, alias="FACE_QUALITY_MAX_POSE_SKEW")
    face_quality_max_aspect_ratio_skew: float = Field(default=1.9, ge=1.0, alias="FACE_QUALITY_MAX_ASPECT_RATIO_SKEW")
    detector_input_width: int = Field(default=320, ge=160, alias="DETECTOR_INPUT_WIDTH")
    detector_input_height: int = Field(default=320, ge=160, alias="DETECTOR_INPUT_HEIGHT")
    detector_medium_width: int = Field(default=416, ge=160, alias="DETECTOR_MEDIUM_WIDTH")
    detector_medium_height: int = Field(default=416, ge=160, alias="DETECTOR_MEDIUM_HEIGHT")
    detector_large_width: int = Field(default=512, ge=160, alias="DETECTOR_LARGE_WIDTH")
    detector_large_height: int = Field(default=512, ge=160, alias="DETECTOR_LARGE_HEIGHT")
    detector_medium_track_count: int = Field(default=4, ge=1, alias="DETECTOR_MEDIUM_TRACK_COUNT")
    detector_high_track_count: int = Field(default=8, ge=1, alias="DETECTOR_HIGH_TRACK_COUNT")
    detector_high_occupancy_ratio: float = Field(default=0.12, ge=0, le=1, alias="DETECTOR_HIGH_OCCUPANCY_RATIO")
    detector_interval_frames: int = Field(default=8, ge=1, alias="DETECTOR_INTERVAL_FRAMES")
    detector_min_score: float = Field(default=0.82, ge=0, le=1, alias="DETECTOR_MIN_SCORE")
    detector_high_quality_threshold: float = Field(default=0.82, ge=0, le=1, alias="DETECTOR_HIGH_QUALITY_THRESHOLD")
    detector_medium_quality_threshold: float = Field(default=0.88, ge=0, le=1, alias="DETECTOR_MEDIUM_QUALITY_THRESHOLD")
    detector_small_face_threshold: float = Field(default=0.93, ge=0, le=1, alias="DETECTOR_SMALL_FACE_THRESHOLD")
    detector_small_face_area_ratio: float = Field(default=0.004, ge=0, le=1, alias="DETECTOR_SMALL_FACE_AREA_RATIO")
    detector_medium_face_area_ratio: float = Field(default=0.012, ge=0, le=1, alias="DETECTOR_MEDIUM_FACE_AREA_RATIO")
    detector_min_face_width: int = Field(default=32, ge=1, alias="DETECTOR_MIN_FACE_WIDTH")
    detector_min_face_height: int = Field(default=32, ge=1, alias="DETECTOR_MIN_FACE_HEIGHT")
    detector_min_face_area: int = Field(default=1000, ge=1, alias="DETECTOR_MIN_FACE_AREA")
    detector_max_aspect_ratio: float = Field(default=1.45, ge=1.0, alias="DETECTOR_MAX_ASPECT_RATIO")
    detector_min_aspect_ratio: float = Field(default=0.65, ge=0.1, le=2.0, alias="DETECTOR_MIN_ASPECT_RATIO")
    detector_min_face_area_ratio: float = Field(default=0.0015, ge=0, le=1, alias="DETECTOR_MIN_FACE_AREA_RATIO")
    detector_edge_touch_ratio: float = Field(default=0.25, ge=0, le=1, alias="DETECTOR_EDGE_TOUCH_RATIO")
    detector_edge_high_confidence: float = Field(default=0.95, ge=0, le=1, alias="DETECTOR_EDGE_HIGH_CONFIDENCE")
    detector_max_landmark_asymmetry: float = Field(default=0.55, ge=0, le=2, alias="DETECTOR_MAX_LANDMARK_ASYMMETRY")
    detector_input_enable_enhancement: bool = Field(default=True, alias="DETECTOR_INPUT_ENABLE_ENHANCEMENT")
    detector_resolution_cap_enabled: bool = Field(default=True, alias="DETECTOR_RESOLUTION_CAP_ENABLED")
    detector_min_input_pixels: int = Field(default=90000, alias="DETECTOR_MIN_INPUT_PIXELS")
    detector_max_input_pixels: int = Field(default=120000, alias="DETECTOR_MAX_INPUT_PIXELS")
    detector_temporal_blend_current: float = Field(default=0.7, ge=0, le=1, alias="DETECTOR_TEMPORAL_BLEND_CURRENT")
    detector_temporal_blend_history: float = Field(default=0.3, ge=0, le=1, alias="DETECTOR_TEMPORAL_BLEND_HISTORY")
    detector_temporal_iou_match: float = Field(default=0.35, ge=0, le=1, alias="DETECTOR_TEMPORAL_IOU_MATCH")
    detector_edge_margin_ratio: float = Field(default=0.0, ge=0, le=0.25, alias="DETECTOR_EDGE_MARGIN_RATIO")
    detector_center_priority_enabled: bool = Field(default=False, alias="DETECTOR_CENTER_PRIORITY_ENABLED")
    detector_center_max_distance: float = Field(default=0.9, ge=0, le=2, alias="DETECTOR_CENTER_MAX_DISTANCE")
    detector_overload_face_count: int = Field(default=12, ge=1, alias="DETECTOR_OVERLOAD_FACE_COUNT")
    temporal_window_size: int = Field(default=8, ge=1, alias="TEMPORAL_WINDOW_SIZE")
    temporal_min_confirmations: int = Field(default=3, ge=1, alias="TEMPORAL_MIN_CONFIRMATIONS")
    temporal_min_average_confidence: float = Field(default=0.50, ge=0, le=1, alias="TEMPORAL_MIN_AVERAGE_CONFIDENCE")
    temporal_max_track_distance: float = Field(default=90.0, ge=1, alias="TEMPORAL_MAX_TRACK_DISTANCE")
    temporal_track_ttl_frames: int = Field(default=18, ge=1, alias="TEMPORAL_TRACK_TTL_FRAMES")
    temporal_min_track_iou: float = Field(default=0.08, ge=0, le=1, alias="TEMPORAL_MIN_TRACK_IOU")
    recognition_interval_frames: int = Field(default=20, ge=1, alias="RECOGNITION_INTERVAL_FRAMES")
    recognition_cache_min_iou: float = Field(default=0.35, ge=0, le=1, alias="RECOGNITION_CACHE_MIN_IOU")
    confidence_small_face_threshold: float = Field(default=0.60, ge=0, le=1, alias="CONFIDENCE_SMALL_FACE_THRESHOLD")
    confidence_large_face_threshold: float = Field(default=0.45, ge=0, le=1, alias="CONFIDENCE_LARGE_FACE_THRESHOLD")
    confidence_small_face_size: int = Field(default=100, ge=1, alias="CONFIDENCE_SMALL_FACE_SIZE")
    confidence_low_light_threshold: float = Field(default=70.0, ge=0, le=255, alias="CONFIDENCE_LOW_LIGHT_THRESHOLD")
    confidence_low_light_margin: float = Field(default=0.04, ge=0, le=1, alias="CONFIDENCE_LOW_LIGHT_MARGIN")
    confidence_blur_margin: float = Field(default=0.03, ge=0, le=1, alias="CONFIDENCE_BLUR_MARGIN")
    event_cooldown_frames: int = Field(default=90, ge=0, alias="EVENT_COOLDOWN_FRAMES")
    event_min_stable_frames: int = Field(default=3, ge=1, alias="EVENT_MIN_STABLE_FRAMES")
    tracking_min_track_age: int = Field(default=10, ge=1, alias="TRACKING_MIN_TRACK_AGE")
    tracking_max_lost_frames: int = Field(default=18, ge=1, alias="TRACKING_MAX_LOST_FRAMES")
    tracking_overlay_interval: int = Field(default=5, ge=1, alias="TRACKING_OVERLAY_INTERVAL")
    tracking_ema_alpha: float = Field(default=0.35, ge=0, le=1, alias="TRACKING_EMA_ALPHA")
    tracking_bbox_ema_alpha: float = Field(default=0.5, ge=0, le=1, alias="TRACKING_BBOX_EMA_ALPHA")
    tracking_soft_recovery_frames: int = Field(default=5, ge=1, alias="TRACKING_SOFT_RECOVERY_FRAMES")
    tracking_confirm_frames: int = Field(default=2, ge=1, alias="TRACKING_CONFIRM_FRAMES")
    tracking_min_quality_score: float = Field(default=0.35, ge=0, le=1, alias="TRACKING_MIN_QUALITY_SCORE")
    face_crop_min_eye_band_ratio: float = Field(default=0.12, ge=0, le=1, alias="FACE_CROP_MIN_EYE_BAND_RATIO")
    face_crop_max_forehead_ratio: float = Field(default=0.55, ge=0, le=1, alias="FACE_CROP_MAX_FOREHEAD_RATIO")
    face_crop_min_chin_ratio: float = Field(default=0.08, ge=0, le=1, alias="FACE_CROP_MIN_CHIN_RATIO")
    tracking_bbox_area_change_ratio: float = Field(default=0.25, ge=0, le=1, alias="TRACKING_BBOX_AREA_CHANGE_RATIO")
    tracking_confidence_drop_threshold: float = Field(default=0.15, ge=0, le=1, alias="TRACKING_CONFIDENCE_DROP_THRESHOLD")
    tracking_vote_window: int = Field(default=10, ge=1, alias="TRACKING_VOTE_WINDOW")
    tracking_stable_frames: int = Field(default=15, ge=1, alias="TRACKING_STABLE_FRAMES")
    tracking_quality_decay: float = Field(default=0.92, ge=0.5, le=1, alias="TRACKING_QUALITY_DECAY")
    tracking_min_motion_stability: float = Field(default=0.25, ge=0, le=1, alias="TRACKING_MIN_MOTION_STABILITY")
    tracking_min_recognition_quality: float = Field(default=0.40, ge=0, le=1, alias="TRACKING_MIN_RECOGNITION_QUALITY")
    tracking_embedding_cooldown_frames: int = Field(default=12, ge=1, alias="TRACKING_EMBEDDING_COOLDOWN_FRAMES")
    tracking_embedding_quality_jump: float = Field(default=0.18, ge=0.05, le=0.5, alias="TRACKING_EMBEDDING_QUALITY_JUMP")
    tracking_identity_lock_frames: int = Field(default=8, ge=1, alias="TRACKING_IDENTITY_LOCK_FRAMES")
    tracking_identity_lock_margin: float = Field(default=0.04, ge=0, le=0.2, alias="TRACKING_IDENTITY_LOCK_MARGIN")
    tracking_fused_embedding_alpha: float = Field(default=0.25, ge=0.05, le=1, alias="TRACKING_FUSED_EMBEDDING_ALPHA")
    tracking_embedding_outlier_cosine: float = Field(default=0.35, ge=0, le=1, alias="TRACKING_EMBEDDING_OUTLIER_COSINE")
    tracking_memory_max_samples: int = Field(default=12, ge=1, alias="TRACKING_MEMORY_MAX_SAMPLES")
    tracking_memory_min_quality: float = Field(default=0.45, ge=0, le=1, alias="TRACKING_MEMORY_MIN_QUALITY")
    tracking_match_shortlist_k: int = Field(default=5, ge=1, alias="TRACKING_MATCH_SHORTLIST_K")
    tracking_soft_match_margin: float = Field(default=0.08, ge=0, le=0.25, alias="TRACKING_SOFT_MATCH_MARGIN")
    tracking_min_soft_threshold: float = Field(default=0.38, ge=0, le=1, alias="TRACKING_MIN_SOFT_THRESHOLD")
    tracking_temporal_threshold_relief: float = Field(default=0.06, ge=0, le=0.2, alias="TRACKING_TEMPORAL_THRESHOLD_RELIEF")
    tracking_temporal_lock_min_agreement: int = Field(default=4, ge=1, alias="TRACKING_TEMPORAL_LOCK_MIN_AGREEMENT")
    tracking_temporal_lock_min_consistency: float = Field(
        default=0.35, ge=0, le=1, alias="TRACKING_TEMPORAL_LOCK_MIN_CONSISTENCY"
    )
    tracking_global_memory_ttl_frames: int = Field(default=90, ge=1, alias="TRACKING_GLOBAL_MEMORY_TTL_FRAMES")
    tracking_global_memory_max_lost: int = Field(default=32, ge=1, alias="TRACKING_GLOBAL_MEMORY_MAX_LOST")
    tracking_reid_min_similarity: float = Field(default=0.38, ge=0, le=1, alias="TRACKING_REID_MIN_SIMILARITY")
    tracking_blur_fusion_suppression: float = Field(default=0.12, ge=0.02, le=1, alias="TRACKING_BLUR_FUSION_SUPPRESSION")
    tracking_identity_hypothesis_decay: float = Field(default=0.94, ge=0.5, le=1, alias="TRACKING_IDENTITY_HYPOTHESIS_DECAY")
    motion_max_frame_displacement_px: float = Field(default=120.0, ge=1, alias="MOTION_MAX_FRAME_DISPLACEMENT_PX")
    motion_max_area_jitter_ratio: float = Field(default=0.35, ge=0.01, alias="MOTION_MAX_AREA_JITTER_RATIO")
    motion_max_speed_variance_ratio: float = Field(default=2.5, ge=0.1, alias="MOTION_MAX_SPEED_VARIANCE_RATIO")
    motion_high_threshold: float = Field(default=0.55, ge=0, le=1, alias="MOTION_HIGH_THRESHOLD")
    proposal_min_validation_score: float = Field(default=0.35, ge=0, le=1, alias="PROPOSAL_MIN_VALIDATION_SCORE")
    proposal_blur_weight: float = Field(default=0.20, ge=0, le=1, alias="PROPOSAL_BLUR_WEIGHT")
    proposal_illumination_weight: float = Field(default=0.15, ge=0, le=1, alias="PROPOSAL_ILLUMINATION_WEIGHT")
    proposal_pose_weight: float = Field(default=0.15, ge=0, le=1, alias="PROPOSAL_POSE_WEIGHT")
    proposal_geometry_weight: float = Field(default=0.50, ge=0, le=1, alias="PROPOSAL_GEOMETRY_WEIGHT")
    proposal_max_yaw_ratio: float = Field(default=0.65, ge=0, le=2, alias="PROPOSAL_MAX_YAW_RATIO")
    proposal_max_pitch_ratio: float = Field(default=0.75, ge=0, le=2, alias="PROPOSAL_MAX_PITCH_RATIO")
    detector_interval_min_frames: int = Field(default=4, ge=1, alias="DETECTOR_INTERVAL_MIN_FRAMES")
    detector_interval_max_frames: int = Field(default=16, ge=1, alias="DETECTOR_INTERVAL_MAX_FRAMES")
    detector_interval_stable_frames: int = Field(default=12, ge=1, alias="DETECTOR_INTERVAL_STABLE_FRAMES")
    detector_interval_motion_frames: int = Field(default=5, ge=1, alias="DETECTOR_INTERVAL_MOTION_FRAMES")

    # ── Unified Face Validator (Phase 2A) ──────────────────────────────────
    validator_min_aspect_ratio: float = Field(default=0.65, ge=0.1, le=2.0, alias="VALIDATOR_MIN_ASPECT_RATIO")
    validator_max_aspect_ratio: float = Field(default=1.45, ge=1.0, alias="VALIDATOR_MAX_ASPECT_RATIO")
    validator_edge_margin_ratio: float = Field(default=0.0, ge=0, le=0.25, alias="VALIDATOR_EDGE_MARGIN_RATIO")
    validator_min_face_area_ratio: float = Field(default=0.0015, ge=0, le=1, alias="VALIDATOR_MIN_FACE_AREA_RATIO")
    validator_max_face_area_ratio: float = Field(default=0.45, ge=0, le=1, alias="VALIDATOR_MAX_FACE_AREA_RATIO")
    validator_min_blur_var: float = Field(default=45.0, ge=0, alias="VALIDATOR_MIN_BLUR_VAR")
    validator_min_brightness: float = Field(default=35.0, ge=0, le=255, alias="VALIDATOR_MIN_BRIGHTNESS")
    validator_max_brightness: float = Field(default=230.0, ge=0, le=255, alias="VALIDATOR_MAX_BRIGHTNESS")
    validator_max_landmark_asymmetry: float = Field(default=0.55, ge=0, le=2, alias="VALIDATOR_MAX_LANDMARK_ASYMMETRY")
    validator_min_detector_confidence: float = Field(default=0.45, ge=0, le=1, alias="VALIDATOR_MIN_DETECTOR_CONFIDENCE")
    validator_blur_weight: float = Field(default=0.30, ge=0, le=1, alias="VALIDATOR_BLUR_WEIGHT")
    validator_brightness_weight: float = Field(default=0.15, ge=0, le=1, alias="VALIDATOR_BRIGHTNESS_WEIGHT")
    validator_geometry_weight: float = Field(default=0.25, ge=0, le=1, alias="VALIDATOR_GEOMETRY_WEIGHT")
    validator_landmark_weight: float = Field(default=0.15, ge=0, le=1, alias="VALIDATOR_LANDMARK_WEIGHT")
    validator_size_weight: float = Field(default=0.15, ge=0, le=1, alias="VALIDATOR_SIZE_WEIGHT")
    validator_quality_cutoff: float = Field(default=0.35, ge=0, le=1, alias="VALIDATOR_QUALITY_CUTOFF")
    validator_strict_cutoff: float = Field(default=0.70, ge=0, le=1, alias="VALIDATOR_STRICT_CUTOFF")
    validator_snapshot_enabled: bool = Field(default=False, alias="VALIDATOR_SNAPSHOT_ENABLED")
    validator_snapshot_max_count: int = Field(default=500, ge=1, alias="VALIDATOR_SNAPSHOT_MAX_COUNT")
    validator_snapshot_sampling_rate: float = Field(default=0.15, ge=0, le=1, alias="VALIDATOR_SNAPSHOT_SAMPLING_RATE")
    validator_weak_pass_max_age_frames: int = Field(default=90, ge=1, alias="VALIDATOR_WEAK_PASS_MAX_AGE_FRAMES")
    validator_weak_pass_retry_limit: int = Field(default=3, ge=1, alias="VALIDATOR_WEAK_PASS_RETRY_LIMIT")
    validator_weak_pass_temporal_min: float = Field(default=0.65, ge=0, le=1, alias="VALIDATOR_WEAK_PASS_TEMPORAL_MIN")
    validator_weak_pass_landmark_min: float = Field(default=0.5, ge=0, le=1, alias="VALIDATOR_WEAK_PASS_LANDMARK_MIN")
    validator_weak_pass_confidence_boost: float = Field(default=0.15, ge=0, le=0.5, alias="VALIDATOR_WEAK_PASS_CONFIDENCE_BOOST")
    validator_weak_pass_soft_temporal: float = Field(default=0.55, ge=0, le=1, alias="VALIDATOR_WEAK_PASS_SOFT_TEMPORAL")
    validator_weak_pass_motion_min: float = Field(default=0.6, ge=0, le=1, alias="VALIDATOR_WEAK_PASS_MOTION_MIN")
    validator_weak_pass_persistence_min: float = Field(default=0.3, ge=0, le=1, alias="VALIDATOR_WEAK_PASS_PERSISTENCE_MIN")
    validator_track_only_max_age_frames: int = Field(default=45, ge=1, alias="VALIDATOR_TRACK_ONLY_MAX_AGE_FRAMES")
    validator_temporal_decay: float = Field(default=0.92, ge=0.5, le=1, alias="VALIDATOR_TEMPORAL_DECAY")
    validator_adaptive_brightness: bool = Field(default=False, alias="VALIDATOR_ADAPTIVE_BRIGHTNESS")
    validator_min_quality_for_embedding: float = Field(default=0.55, ge=0, le=1, alias="VALIDATOR_MIN_QUALITY_FOR_EMBEDDING")
    validator_max_faces_per_frame: int = Field(default=25, ge=1, alias="VALIDATOR_MAX_FACES_PER_FRAME")

    # ── Phase 2A rollout flags (default True = legacy validators still active) ─
    enable_legacy_face_validation: bool = Field(default=True, alias="ENABLE_LEGACY_FACE_VALIDATION")
    enable_legacy_quality_checks: bool = Field(default=True, alias="ENABLE_LEGACY_QUALITY_CHECKS")

    # ── Phase 2A.1: Detection Observability Foundation ───────────────────────
    detection_metrics_enabled: bool = Field(default=True, alias="DETECTION_METRICS_ENABLED")
    detection_metrics_export_interval: int = Field(default=100, ge=1, alias="DETECTION_METRICS_EXPORT_INTERVAL")
    detection_metrics_log_dir: Path = Field(default=Path("logs/detection_metrics"), alias="DETECTION_METRICS_LOG_DIR")
    false_positive_snapshot_enabled: bool = Field(default=True, alias="FALSE_POSITIVE_SNAPSHOT_ENABLED")
    false_positive_max_snapshots: int = Field(default=1000, ge=1, alias="FALSE_POSITIVE_MAX_SNAPSHOTS")
    false_positive_sampling_rate: float = Field(default=0.10, ge=0.0, le=1.0, alias="FALSE_POSITIVE_SAMPLING_RATE")
    false_positive_min_confidence: float = Field(default=0.60, ge=0.0, le=1.0, alias="FALSE_POSITIVE_MIN_CONFIDENCE")
    false_positive_dataset_dir: Path = Field(default=Path("data/hard_negatives"), alias="FALSE_POSITIVE_DATASET_DIR")

    # ── Phase 2A.2: Multi-Scale Detection ─────────────────────────────────────
    enable_multiscale_detection: bool = Field(default=False, alias="ENABLE_MULTISCALE_DETECTION")
    multiscale_scales: list[float] = Field(default=[1.0, 1.5, 2.0], alias="MULTISCALE_SCALES")
    multiscale_adaptive_activation: bool = Field(default=True, alias="MULTISCALE_ADAPTIVE_ACTIVATION")
    multiscale_tiny_face_threshold: int = Field(default=30, ge=1, alias="MULTISCALE_TINY_FACE_THRESHOLD")
    multiscale_small_face_threshold: int = Field(default=60, ge=1, alias="MULTISCALE_SMALL_FACE_THRESHOLD")
    multiscale_max_scales_per_frame: int = Field(default=2, ge=1, le=3, alias="MULTISCALE_MAX_SCALES_PER_FRAME")
    multiscale_gpu_batching: bool = Field(default=True, alias="MULTISCALE_GPU_BATCHING")

    # ── Phase 2A.4: Proposal Fusion Engine ─────────────────────────────────────
    enable_confidence_normalization: bool = Field(default=False, alias="ENABLE_CONFIDENCE_NORMALIZATION")
    fusion_wbf_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0, alias="FUSION_WBF_IOU_THRESHOLD")
    fusion_crowd_iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0, alias="FUSION_CROWD_IOU_THRESHOLD")
    fusion_scale_weight_tiny: float = Field(default=1.3, ge=1.0, alias="FUSION_SCALE_WEIGHT_TINY")
    fusion_scale_weight_small: float = Field(default=1.1, ge=1.0, alias="FUSION_SCALE_WEIGHT_SMALL")
    fusion_scale_weight_baseline: float = Field(default=1.0, ge=1.0, alias="FUSION_SCALE_WEIGHT_BASELINE")
    fusion_max_proposals_per_frame: int = Field(default=50, ge=1, alias="FUSION_MAX_PROPOSALS_PER_FRAME")

    # ── Phase 2A.5: Temporal Weak Detection Recovery ───────────────────────────
    enable_weak_detection_memory: bool = Field(default=False, alias="ENABLE_WEAK_DETECTION_MEMORY")
    weak_memory_max_frames: int = Field(default=32, ge=1, alias="WEAK_MEMORY_MAX_FRAMES")
    weak_memory_cluster_iou: float = Field(default=0.4, ge=0.0, le=1.0, alias="WEAK_MEMORY_CLUSTER_IOU")
    weak_memory_min_recurrence: int = Field(default=3, ge=1, alias="WEAK_MEMORY_MIN_RECURRENCE")
    weak_memory_promotion_boost: float = Field(default=0.15, ge=0.0, le=1.0, alias="WEAK_MEMORY_PROMOTION_BOOST")

    # ── Phase 2A.3: Tile-Based Crowd Recovery ───────────────────────────────────
    enable_tile_detection: bool = Field(default=False, alias="ENABLE_TILE_DETECTION")
    tile_size: int = Field(default=640, ge=320, alias="TILE_SIZE")
    tile_overlap: float = Field(default=0.20, ge=0.0, le=0.5, alias="TILE_OVERLAP")
    tile_crowd_threshold: int = Field(default=8, ge=1, alias="TILE_CROWD_THRESHOLD")
    tile_max_tiles: int = Field(default=9, ge=1, alias="TILE_MAX_TILES")
    tile_edge_padding: int = Field(default=32, ge=0, alias="TILE_EDGE_PADDING")
    tile_priority_center: bool = Field(default=True, alias="TILE_PRIORITY_CENTER")

    video_frame_skip: int = Field(default=1, ge=1, alias="VIDEO_FRAME_SKIP")
    video_inference_width: int = Field(default=640, ge=160, alias="VIDEO_INFERENCE_WIDTH")
    video_progress_interval: int = Field(default=10, ge=1, alias="VIDEO_PROGRESS_INTERVAL")
    video_preview_interval: int = Field(default=5, ge=1, alias="VIDEO_PREVIEW_INTERVAL")
    rejected_face_snapshot_interval: int = Field(default=10, ge=1, alias="REJECTED_FACE_SNAPSHOT_INTERVAL")
    video_event_dedupe_frames: int = Field(default=30, ge=0, alias="VIDEO_EVENT_DEDUPE_FRAMES")
    video_worker_queue_size: int = Field(default=8, ge=1, alias="VIDEO_WORKER_QUEUE_SIZE")
    live_event_dedupe_seconds: int = Field(default=10, ge=0, alias="LIVE_EVENT_DEDUPE_SECONDS")
    max_image_mb: int = Field(default=10, ge=1, alias="MAX_IMAGE_MB")

    @field_validator(
        "data_dir",
        "uploads_dir",
        "snapshots_dir",
        "videos_dir",
        "previews_dir",
        "rejected_faces_dir",
        "log_dir",
        mode="before",
    )
    @classmethod
    def _coerce_path(cls, v: str | Path) -> Path:
        return Path(v)

    def resolved_uploads_dir(self) -> Path:
        p = self.uploads_dir if self.uploads_dir.is_absolute() else self.project_root / self.uploads_dir
        return p.resolve()

    def resolved_snapshots_dir(self) -> Path:
        p = (
            self.snapshots_dir
            if self.snapshots_dir.is_absolute()
            else self.project_root / self.snapshots_dir
        )
        return p.resolve()

    def resolved_videos_dir(self) -> Path:
        p = self.videos_dir if self.videos_dir.is_absolute() else self.project_root / self.videos_dir
        return p.resolve()

    def resolved_previews_dir(self) -> Path:
        p = self.previews_dir if self.previews_dir.is_absolute() else self.project_root / self.previews_dir
        return p.resolve()

    def resolved_rejected_faces_dir(self) -> Path:
        p = self.rejected_faces_dir if self.rejected_faces_dir.is_absolute() else self.project_root / self.rejected_faces_dir
        return p.resolve()

    def resolved_log_dir(self) -> Path:
        p = self.log_dir if self.log_dir.is_absolute() else self.project_root / self.log_dir
        return p.resolve()

    def resolved_detection_metrics_log_dir(self) -> Path:
        p = (
            self.detection_metrics_log_dir
            if self.detection_metrics_log_dir.is_absolute()
            else self.project_root / self.detection_metrics_log_dir
        )
        return p.resolve()

    def resolved_false_positive_dataset_dir(self) -> Path:
        """Resolve false positive dataset directory relative to workspace."""
        if self.false_positive_dataset_dir.is_absolute():
            return self.false_positive_dataset_dir
        return self.workspace_dir / self.false_positive_dataset_dir


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton for import-time reuse (e.g. logging, DB)."""
    return Settings()

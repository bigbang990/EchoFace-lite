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
    face_quality_min_face_size: int = Field(default=40, ge=1, alias="FACE_QUALITY_MIN_FACE_SIZE")
    face_quality_max_aspect_ratio_skew: float = Field(default=1.9, ge=1.0, alias="FACE_QUALITY_MAX_ASPECT_RATIO_SKEW")
    detector_input_width: int = Field(default=512, ge=160, alias="DETECTOR_INPUT_WIDTH")
    detector_interval_frames: int = Field(default=2, ge=1, alias="DETECTOR_INTERVAL_FRAMES")
    detector_min_score: float = Field(default=0.50, ge=0, le=1, alias="DETECTOR_MIN_SCORE")
    detector_min_face_width: int = Field(default=32, ge=1, alias="DETECTOR_MIN_FACE_WIDTH")
    detector_min_face_height: int = Field(default=32, ge=1, alias="DETECTOR_MIN_FACE_HEIGHT")
    detector_min_face_area: int = Field(default=1000, ge=1, alias="DETECTOR_MIN_FACE_AREA")
    detector_max_aspect_ratio: float = Field(default=2.3, ge=1.0, alias="DETECTOR_MAX_ASPECT_RATIO")
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
    confidence_low_light_threshold: float = Field(default=70.0, ge=0, le=255, alias="CONFIDENCE_LOW_LIGHT_THRESHOLD")
    confidence_low_light_margin: float = Field(default=0.04, ge=0, le=1, alias="CONFIDENCE_LOW_LIGHT_MARGIN")
    confidence_blur_margin: float = Field(default=0.03, ge=0, le=1, alias="CONFIDENCE_BLUR_MARGIN")
    event_cooldown_frames: int = Field(default=90, ge=0, alias="EVENT_COOLDOWN_FRAMES")
    event_min_stable_frames: int = Field(default=3, ge=1, alias="EVENT_MIN_STABLE_FRAMES")

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


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton for import-time reuse (e.g. logging, DB)."""
    return Settings()

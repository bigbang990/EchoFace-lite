"""EffectiveRuntimeConfig compilation layer - FULLY RESOLVED VALUES ONLY.

Stage 2 (Module M2): Compile immutable runtime configuration snapshot.
Flattened values only - no Settings exposure to prevent bypass of override logic.
"""

from dataclasses import dataclass
from typing import Any, Dict

from ecoface_lite.core.config import Settings
from ecoface_lite.core.experimental_config import BackendType, PipelineMode
from ecoface_lite.core.runtime_state import CpuProtectionState


# Explicit mapping from override keys to Settings attributes
# Verbose > implicit magic in production systems
CONFIG_ATTRIBUTE_MAP = {
    "pipeline_mode": "pipeline_mode",  # Not in Settings, handled separately
    "validator_quality_cutoff": "validator_quality_cutoff",
    "validator_strict_cutoff": "validator_strict_cutoff",
    "validator_min_detector_confidence": "validator_min_detector_confidence",
    "validator_min_blur_var": "validator_min_blur_var",
    "validator_max_faces_per_frame": "validator_max_faces_per_frame",
    "validator_min_quality_for_embedding": "validator_min_quality_for_embedding",
    "tracker_detection_interval": "detector_interval_frames",  # Note: mapping difference
    "track_confirmation_frames": "tracking_confirm_frames",  # Note: mapping difference
    "track_lost_buffer": "tracking_max_lost_frames",  # Note: mapping difference
    "identity_match_threshold": "match_confidence_threshold",  # Note: mapping difference
}


@dataclass
class EffectiveRuntimeConfig:
    """Compiled runtime configuration - FULLY RESOLVED VALUES ONLY.
    
    This is an immutable snapshot captured at video job initialization.
    All pipeline components read from this flattened copy.
    NEVER exposes Settings downstream to prevent accidental bypass of override logic.
    
    Architecture decision: Flatten everything. No base_settings reference.
    Pipeline reads: effective_config.video_frame_skip (not effective_config.base_settings.video_frame_skip)
    """
    # Pipeline mode (from experimental config)
    pipeline_mode: PipelineMode
    
    # Video processing parameters (from Settings)
    video_frame_skip: int
    video_inference_width: int
    video_progress_interval: int
    video_preview_interval: int
    video_event_dedupe_frames: int
    
    # Validator parameters (from Settings + overrides)
    validator_quality_cutoff: float
    validator_strict_cutoff: float
    validator_min_detector_confidence: float
    validator_min_blur_var: float
    validator_max_faces_per_frame: int
    validator_min_quality_for_embedding: float
    
    # Tracker parameters (from Settings + overrides)
    tracker_detection_interval: int
    track_confirmation_frames: int
    track_lost_buffer: int
    tracking_min_track_age: int
    tracking_max_lost_frames: int
    tracking_stable_frames: int
    tracking_ema_alpha: float
    
    # Recognition parameters (from Settings + overrides)
    identity_match_threshold: float
    recognition_interval_frames: int
    
    # Detection parameters (from Settings)
    detector_interval_frames: int
    detector_min_score: float
    detector_overload_face_count: int
    
    # Temporal parameters (from Settings)
    temporal_window_size: int
    temporal_min_confirmations: int
    temporal_min_average_confidence: float
    
    # Preprocessing parameters (from Settings)
    preprocessing_enable_clahe: bool
    preprocessing_enable_gamma: bool
    preprocessing_gamma: float
    preprocessing_brightness_target: float
    preprocessing_max_width: int
    
    # CPU protection state (from RuntimeOverrideState)
    cpu_protection_active: bool
    current_detection_interval: int  # May be elevated by protection
    
    # Backend type (for gating)
    backend_type: BackendType
    
    # Experiment session ID (for lineage tracking)
    experiment_session_id: str
    
    @classmethod
    def compile(
        cls,
        settings: Settings,
        overrides: Dict[str, Any],
        cpu_protection_state: CpuProtectionState,
        backend_type: BackendType,
        experiment_session_id: str,
    ) -> "EffectiveRuntimeConfig":
        """Compile effective config from base settings and overrides.
        
        Uses explicit mapping, no getattr magic.
        Flattened values only, no Settings exposure.
        """
        def get_value(override_key: str, settings_attr: str, default: Any) -> Any:
            """Get value with override priority, using explicit mapping."""
            if override_key in overrides:
                return overrides[override_key]
            return getattr(settings, settings_attr, default)
        
        # Handle CPU protection elevation of detection interval
        base_interval = get_value(
            "tracker_detection_interval",
            "detector_interval_frames",
            8
        )
        if cpu_protection_state.protection_active:
            effective_interval = max(
                base_interval,
                cpu_protection_state.current_detection_interval
            )
        else:
            effective_interval = base_interval
        
        # Parse pipeline mode from override or default
        pipeline_mode_str = overrides.get("pipeline_mode", "HYBRID")
        try:
            pipeline_mode = PipelineMode(pipeline_mode_str)
        except ValueError:
            pipeline_mode = PipelineMode.HYBRID
        
        return cls(
            # Pipeline mode
            pipeline_mode=pipeline_mode,
            
            # Video processing parameters (from Settings only)
            video_frame_skip=settings.video_frame_skip,
            video_inference_width=settings.video_inference_width,
            video_progress_interval=settings.video_progress_interval,
            video_preview_interval=settings.video_preview_interval,
            video_event_dedupe_frames=settings.video_event_dedupe_frames,
            
            # Validator parameters (from Settings + overrides)
            validator_quality_cutoff=get_value(
                "validator_quality_cutoff", "validator_quality_cutoff", 0.35
            ),
            validator_strict_cutoff=get_value(
                "validator_strict_cutoff", "validator_strict_cutoff", 0.70
            ),
            validator_min_detector_confidence=get_value(
                "validator_min_detector_confidence", "validator_min_detector_confidence", 0.45
            ),
            validator_min_blur_var=get_value(
                "validator_min_blur_var", "validator_min_blur_var", 45.0
            ),
            validator_max_faces_per_frame=get_value(
                "validator_max_faces_per_frame", "validator_max_faces_per_frame", 8
            ),
            validator_min_quality_for_embedding=get_value(
                "validator_min_quality_for_embedding", "validator_min_quality_for_embedding", 0.55
            ),
            
            # Tracker parameters (from Settings + overrides with explicit mapping)
            tracker_detection_interval=effective_interval,
            track_confirmation_frames=get_value(
                "track_confirmation_frames", "tracking_confirm_frames", 2
            ),
            track_lost_buffer=get_value(
                "track_lost_buffer", "tracking_max_lost_frames", 18
            ),
            tracking_min_track_age=settings.tracking_min_track_age,
            tracking_max_lost_frames=settings.tracking_max_lost_frames,
            tracking_stable_frames=settings.tracking_stable_frames,
            tracking_ema_alpha=settings.tracking_ema_alpha,
            
            # Recognition parameters (from Settings + overrides with explicit mapping)
            identity_match_threshold=get_value(
                "identity_match_threshold", "match_confidence_threshold", 0.38
            ),
            recognition_interval_frames=settings.recognition_interval_frames,
            
            # Detection parameters (from Settings)
            detector_interval_frames=settings.detector_interval_frames,
            detector_min_score=settings.detector_min_score,
            detector_overload_face_count=settings.detector_overload_face_count,
            
            # Temporal parameters (from Settings)
            temporal_window_size=settings.temporal_window_size,
            temporal_min_confirmations=settings.temporal_min_confirmations,
            temporal_min_average_confidence=settings.temporal_min_average_confidence,
            
            # Preprocessing parameters (from Settings)
            preprocessing_enable_clahe=settings.preprocessing_enable_clahe,
            preprocessing_enable_gamma=settings.preprocessing_enable_gamma,
            preprocessing_gamma=settings.preprocessing_gamma,
            preprocessing_brightness_target=settings.preprocessing_brightness_target,
            preprocessing_max_width=settings.preprocessing_max_width,
            
            # CPU protection state
            cpu_protection_active=cpu_protection_state.protection_active,
            current_detection_interval=effective_interval,
            
            # Backend type
            backend_type=backend_type,
            
            # Experiment session ID
            experiment_session_id=experiment_session_id,
        )

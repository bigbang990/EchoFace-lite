"""EffectiveRuntimeConfig compilation layer - FULLY RESOLVED VALUES ONLY.

Stage 2 (Module M2): Compile immutable runtime configuration snapshot.
Flattened values only - no Settings exposure to prevent bypass of override logic.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

from ecoface_lite.core.config import Settings
from ecoface_lite.core.experimental_config import BackendType, PipelineMode, ConfigKey
from ecoface_lite.core.runtime_state import CpuProtectionState
from ecoface_lite.core.metrics import metrics

logger = logging.getLogger(__name__)


class ConfigSource(str, Enum):
    """Source of a configuration value for auditing."""
    DEFAULT = "default"
    ENV_OVERRIDE = "env"
    CONSOLE_OVERRIDE = "console"
    BENCHMARK_OVERRIDE = "benchmark"
    RUNTIME_MUTATION = "runtime"


@dataclass
class ConfigValue:
    """A configuration value with its source for auditing."""
    value: Any
    source: ConfigSource
    original_value: Any = None


# Explicit mapping from override keys to Settings attributes
CONFIG_ATTRIBUTE_MAP = {
    ConfigKey.PIPELINE_MODE.value: "pipeline_mode",
    ConfigKey.VALIDATOR_QUALITY_CUTOFF.value: "validator_quality_cutoff",
    ConfigKey.VALIDATOR_STRICT_CUTOFF.value: "validator_strict_cutoff",
    ConfigKey.VALIDATOR_MIN_DETECTOR_CONFIDENCE.value: "validator_min_detector_confidence",
    ConfigKey.VALIDATOR_MIN_BLUR_VAR.value: "validator_min_blur_var",
    ConfigKey.VALIDATOR_MAX_FACES_PER_FRAME.value: "validator_max_faces_per_frame",
    ConfigKey.VALIDATOR_MIN_QUALITY_FOR_EMBEDDING.value: "validator_min_quality_for_embedding",
    ConfigKey.TRACKER_DETECTION_INTERVAL.value: "detector_interval_frames",
    ConfigKey.TRACK_CONFIRMATION_FRAMES.value: "tracking_confirm_frames",
    ConfigKey.TRACK_LOST_BUFFER.value: "tracking_max_lost_frames",
    ConfigKey.IDENTITY_MATCH_THRESHOLD.value: "match_confidence_threshold",
}


@dataclass
class EffectiveRuntimeConfig:
    """Compiled runtime configuration - FULLY RESOLVED VALUES ONLY.
    
    This is an immutable snapshot captured at video job initialization.
    All pipeline components read from this flattened copy.
    
    Architecture decision: Flatten everything. No base_settings reference.
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

    # --- Phase 1: Integrity Audit Additions ---
    sources: Dict[str, ConfigValue] = field(default_factory=dict)
    integrity_warnings: List[str] = field(default_factory=list)
    effective_profiles: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def compile(
        cls,
        settings: Settings,
        overrides: Dict[str, Any],
        cpu_protection_state: CpuProtectionState,
        backend_type: BackendType,
        experiment_session_id: str,
    ) -> "EffectiveRuntimeConfig":
        """Compile effective config from base settings and overrides."""
        sources = {}
        warnings = []

        def get_resolved_value(override_key: str, settings_attr: str, default: Any) -> Any:
            """Get value with override priority and track source."""
            if override_key in overrides:
                val = overrides[override_key]
                sources[override_key] = ConfigValue(value=val, source=ConfigSource.CONSOLE_OVERRIDE)
                return val
            
            # Check for ENV overrides in Settings (Pydantic handles this, but we want to know if it's non-default)
            # For simplicity, we assume anything in settings that differs from Pydantic default might be ENV or manually set
            val = getattr(settings, settings_attr, default)
            sources[override_key] = ConfigValue(value=val, source=ConfigSource.DEFAULT)
            return val

        # Handle CPU protection elevation of detection interval
        base_interval = get_resolved_value(
            "tracker_detection_interval",
            "detector_interval_frames",
            8
        )
        if cpu_protection_state.protection_active:
            effective_interval = max(
                base_interval,
                cpu_protection_state.current_detection_interval
            )
            sources["tracker_detection_interval"] = ConfigValue(
                value=effective_interval,
                source=ConfigSource.RUNTIME_MUTATION,
                original_value=base_interval
            )
        else:
            effective_interval = base_interval
        
        # Parse pipeline mode from override or default
        pipeline_mode_str = overrides.get("pipeline_mode", "HYBRID")
        try:
            pipeline_mode = PipelineMode(pipeline_mode_str)
        except ValueError:
            pipeline_mode = PipelineMode.HYBRID
            warnings.append(f"Invalid pipeline_mode: {pipeline_mode_str}. Falling back to HYBRID.")
        
        sources["pipeline_mode"] = ConfigValue(
            value=pipeline_mode.value,
            source=ConfigSource.CONSOLE_OVERRIDE if "pipeline_mode" in overrides else ConfigSource.DEFAULT
        )

        config = cls(
            pipeline_mode=pipeline_mode,
            video_frame_skip=settings.video_frame_skip,
            video_inference_width=settings.video_inference_width,
            video_progress_interval=settings.video_progress_interval,
            video_preview_interval=settings.video_preview_interval,
            video_event_dedupe_frames=settings.video_event_dedupe_frames,
            
            validator_quality_cutoff=get_resolved_value("validator_quality_cutoff", "validator_quality_cutoff", 0.35),
            validator_strict_cutoff=get_resolved_value("validator_strict_cutoff", "validator_strict_cutoff", 0.70),
            validator_min_detector_confidence=get_resolved_value("validator_min_detector_confidence", "validator_min_detector_confidence", 0.45),
            validator_min_blur_var=get_resolved_value("validator_min_blur_var", "validator_min_blur_var", 45.0),
            validator_max_faces_per_frame=get_resolved_value("validator_max_faces_per_frame", "validator_max_faces_per_frame", 8),
            validator_min_quality_for_embedding=get_resolved_value("validator_min_quality_for_embedding", "validator_min_quality_for_embedding", 0.55),
            
            tracker_detection_interval=effective_interval,
            track_confirmation_frames=get_resolved_value("track_confirmation_frames", "tracking_confirm_frames", 2),
            track_lost_buffer=get_resolved_value("track_lost_buffer", "tracking_max_lost_frames", 18),
            tracking_min_track_age=settings.tracking_min_track_age,
            tracking_max_lost_frames=settings.tracking_max_lost_frames,
            tracking_stable_frames=settings.tracking_stable_frames,
            tracking_ema_alpha=settings.tracking_ema_alpha,
            
            identity_match_threshold=get_resolved_value("identity_match_threshold", "match_confidence_threshold", 0.38),
            recognition_interval_frames=settings.recognition_interval_frames,
            
            detector_interval_frames=settings.detector_interval_frames,
            detector_min_score=settings.detector_min_score,
            detector_overload_face_count=settings.detector_overload_face_count,
            
            temporal_window_size=settings.temporal_window_size,
            temporal_min_confirmations=settings.temporal_min_confirmations,
            temporal_min_average_confidence=settings.temporal_min_average_confidence,
            
            preprocessing_enable_clahe=settings.preprocessing_enable_clahe,
            preprocessing_enable_gamma=settings.preprocessing_enable_gamma,
            preprocessing_gamma=settings.preprocessing_gamma,
            preprocessing_brightness_target=settings.preprocessing_brightness_target,
            preprocessing_max_width=settings.preprocessing_max_width,
            
            cpu_protection_active=cpu_protection_state.protection_active,
            current_detection_interval=effective_interval,
            backend_type=backend_type,
            experiment_session_id=experiment_session_id,
            
            sources=sources,
            integrity_warnings=warnings,
            effective_profiles={
                "runtime": "experimental" if overrides else "default",
                "tracking": "standard",
                "detector": "stable"
            }
        )
        
        config._audit_integrity()
        return config

    def _audit_integrity(self) -> None:
        """Detect stale overrides and conflicting parameters."""
        # 1. Detect stale overrides (if session ID is old, etc. - simplified for now)
        if len(self.sources) > 10: # Arbitrary threshold for high override count
            metrics.observe("runtime_override_count", len(self.sources))
            
        # 2. Check for conflicting benchmarks (example: low track_lost_buffer with high detector_interval)
        if self.track_lost_buffer < self.tracker_detection_interval:
            self.integrity_warnings.append(
                f"Conflicting: track_lost_buffer({self.track_lost_buffer}) < tracker_detection_interval({self.tracker_detection_interval}). "
                "Tracks may expire before next detection."
            )

        # 3. Detect mismatched resolution settings
        if self.video_inference_width > 1280:
            self.integrity_warnings.append("High inference resolution detected: performance may degrade.")

        # Update telemetry
        metrics.observe("config_integrity_warnings", len(self.integrity_warnings))

    def log_integrity(self) -> None:
        """Log all effective settings and their sources at startup."""
        logger.info("=== RUNTIME CONFIGURATION INTEGRITY AUDIT ===")
        logger.info(f"Experiment Session ID: {self.experiment_session_id}")
        logger.info(f"Backend Type: {self.backend_type.value}")
        
        for key, val_obj in sorted(self.sources.items()):
            orig = f" (original: {val_obj.original_value})" if val_obj.original_value is not None else ""
            logger.info(f"  {key:30} = {str(val_obj.value):10} [source: {val_obj.source.value}]{orig}")
            
        if self.integrity_warnings:
            logger.warning("Configuration Integrity Warnings:")
            for warning in self.integrity_warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.info("Configuration Integrity: OK")
        logger.info("==============================================")

    def to_observability_json(self) -> Dict[str, Any]:
        """Export effective runtime configuration for observability."""
        return {
            "experiment_session_id": self.experiment_session_id,
            "pipeline_mode": self.pipeline_mode.value,
            "backend_type": self.backend_type.value,
            "effective_profiles": self.effective_profiles,
            "runtime_override_count": len([s for s in self.sources.values() if s.source != ConfigSource.DEFAULT]),
            "integrity_warnings": self.integrity_warnings,
            "settings": {k: v.value for k, v in self.sources.items()}
        }


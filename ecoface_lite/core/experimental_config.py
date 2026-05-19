"""Experimental Configuration Layer - Runtime Tuning Overrides.

Stage 1 (Module M1): Strict validation with 1:1 mapping to actual pipeline variables.
Primitive-only validation, safe clamping, never throws.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal
import logging

logger = logging.getLogger(__name__)


class PipelineMode(str, Enum):
    """Pipeline execution mode for Phase 2 dynamic toggling."""
    
    LEGACY_ONLY = "LEGACY_ONLY"   # Legacy validators ON, Unified OFF
    HYBRID = "HYBRID"             # Legacy validators ON, Unified ON
    UNIFIED_ONLY = "UNIFIED_ONLY" # Legacy validators OFF, Unified ON


class BackendType(str, Enum):
    """Backend execution environment for CPU protection logic."""
    
    LOCAL_CPU = "LOCAL_CPU"
    COLAB_GPU = "COLAB_GPU"
    REMOTE_GPU = "REMOTE_GPU"


class ConfigKey(str, Enum):
    """Strict whitelist of allowable runtime tuning override keys (1:1 with pipeline)."""
    
    # Pipeline mode
    PIPELINE_MODE = "pipeline_mode"
    
    # Validator parameters
    VALIDATOR_QUALITY_CUTOFF = "validator_quality_cutoff"
    VALIDATOR_STRICT_CUTOFF = "validator_strict_cutoff"
    VALIDATOR_MIN_DETECTOR_CONFIDENCE = "validator_min_detector_confidence"
    VALIDATOR_MIN_BLUR_VAR = "validator_min_blur_var"
    VALIDATOR_MAX_FACES_PER_FRAME = "validator_max_faces_per_frame"
    VALIDATOR_MIN_QUALITY_FOR_EMBEDDING = "validator_min_quality_for_embedding"
    
    # Tracker parameters
    TRACKER_DETECTION_INTERVAL = "tracker_detection_interval"
    TRACK_CONFIRMATION_FRAMES = "track_confirmation_frames"
    TRACK_LOST_BUFFER = "track_lost_buffer"
    
    # Recognition parameters
    IDENTITY_MATCH_THRESHOLD = "identity_match_threshold"


ALLOWED_RUNTIME_KEYS = {key.value for key in ConfigKey}


# Safety limits based on observed pipeline behavior
SAFETY_LIMITS: Dict[str, Dict[str, Any]] = {
    ConfigKey.VALIDATOR_QUALITY_CUTOFF.value: {"min": 0.0, "max": 1.0, "type": float},
    ConfigKey.VALIDATOR_STRICT_CUTOFF.value: {"min": 0.0, "max": 1.0, "type": float},
    ConfigKey.VALIDATOR_MIN_DETECTOR_CONFIDENCE.value: {"min": 0.0, "max": 1.0, "type": float},
    ConfigKey.VALIDATOR_MIN_BLUR_VAR.value: {"min": 0.0, "max": 255.0, "type": float},
    ConfigKey.VALIDATOR_MAX_FACES_PER_FRAME.value: {"min": 1, "max": 12, "type": int},
    ConfigKey.VALIDATOR_MIN_QUALITY_FOR_EMBEDDING.value: {"min": 0.0, "max": 1.0, "type": float},
    ConfigKey.TRACKER_DETECTION_INTERVAL.value: {"min": 1, "max": 18, "type": int},
    ConfigKey.TRACK_CONFIRMATION_FRAMES.value: {"min": 1, "max": 10, "type": int},
    ConfigKey.TRACK_LOST_BUFFER.value: {"min": 1, "max": 30, "type": int},
    ConfigKey.IDENTITY_MATCH_THRESHOLD.value: {"min": 0.0, "max": 1.0, "type": float},
}


@dataclass
class ExperimentalRuntimeConfig:
    """Runtime configuration container with 1:1 mapping to pipeline variables."""
    
    # Pipeline mode
    pipeline_mode: PipelineMode = PipelineMode.HYBRID
    
    # Validator parameters
    validator_quality_cutoff: float = 0.35
    validator_strict_cutoff: float = 0.70
    validator_min_detector_confidence: float = 0.45
    validator_min_blur_var: float = 45.0
    validator_max_faces_per_frame: int = 8  # Fixed: default must respect safety max of 12
    validator_min_quality_for_embedding: float = 0.55
    
    # Tracker parameters
    tracker_detection_interval: int = 8
    track_confirmation_frames: int = 2
    track_lost_buffer: int = 18
    
    # Recognition parameters
    identity_match_threshold: float = 0.38
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with whitelisted keys only."""
        return {
            ConfigKey.PIPELINE_MODE.value: self.pipeline_mode.value,
            ConfigKey.VALIDATOR_QUALITY_CUTOFF.value: self.validator_quality_cutoff,
            ConfigKey.VALIDATOR_STRICT_CUTOFF.value: self.validator_strict_cutoff,
            ConfigKey.VALIDATOR_MIN_DETECTOR_CONFIDENCE.value: self.validator_min_detector_confidence,
            ConfigKey.VALIDATOR_MIN_BLUR_VAR.value: self.validator_min_blur_var,
            ConfigKey.VALIDATOR_MAX_FACES_PER_FRAME.value: self.validator_max_faces_per_frame,
            ConfigKey.VALIDATOR_MIN_QUALITY_FOR_EMBEDDING.value: self.validator_min_quality_for_embedding,
            ConfigKey.TRACKER_DETECTION_INTERVAL.value: self.tracker_detection_interval,
            ConfigKey.TRACK_CONFIRMATION_FRAMES.value: self.track_confirmation_frames,
            ConfigKey.TRACK_LOST_BUFFER.value: self.track_lost_buffer,
            ConfigKey.IDENTITY_MATCH_THRESHOLD.value: self.identity_match_threshold,
        }


DEFAULT_EXPERIMENTAL_CONFIG = ExperimentalRuntimeConfig()


def get_default_runtime_config() -> Dict[str, Any]:
    """Get canonical default runtime configuration dictionary.
    
    Used by:
    - Reset endpoint
    - UI reset sync
    - Experiment snapshots
    - Backend runtime restore
    
    Returns:
        Dictionary with all default runtime configuration values.
    """
    return DEFAULT_EXPERIMENTAL_CONFIG.to_dict()


def _is_primitive(value: Any) -> bool:
    """Check if value is an allowed primitive type (int, float, bool, str, enum)."""
    if isinstance(value, (int, float, bool, str)):
        return True
    if isinstance(value, Enum):
        return True
    return False


def validate_and_clamp_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clamp configuration dictionary with strict key whitelisting.
    
    Args:
        config_dict: Partial or full configuration update dictionary.
        
    Returns:
        Validated and clamped configuration dictionary containing only
        whitelisted keys with properly bounded values.
        
    Notes:
        - Unknown keys are ignored, logged as warnings, and NOT persisted.
        - Only primitive types allowed (int, float, bool, str, enum).
        - Nested dicts, arrays, complex objects are rejected.
        - PipelineMode enum validated immediately at ingress.
        - 100% stable, never crashes on partial updates.
    """
    validated = {}
    unknown_keys = []
    rejected_keys = []
    
    for key, value in config_dict.items():
        # Check whitelist
        if key not in ALLOWED_RUNTIME_KEYS:
            unknown_keys.append(key)
            continue
        
        # Reject non-primitive types
        if not _is_primitive(value):
            rejected_keys.append(key)
            logger.warning(f"Rejected non-primitive value for key '{key}': {type(value)}")
            continue
        
        try:
            # Apply key-specific clamping rules
            if key == ConfigKey.PIPELINE_MODE.value:
                # Validate PipelineMode enum immediately at ingress
                try:
                    validated[key] = PipelineMode(value).value
                except ValueError:
                    rejected_keys.append(key)
                    logger.warning(
                        f"Rejected invalid pipeline_mode value: {value}. "
                        f"Must be one of: {[m.value for m in PipelineMode]}"
                    )
            else:
                limits = SAFETY_LIMITS.get(key)
                if limits:
                    target_type = limits["type"]
                    min_val = limits["min"]
                    max_val = limits["max"]
                    
                    # Convert to target type
                    if target_type == int:
                        clamped = max(min_val, min(max_val, int(float(value))))
                        validated[key] = clamped
                    elif target_type == float:
                        clamped = max(min_val, min(max_val, float(value)))
                        validated[key] = clamped
                else:
                    # No limits defined, pass through as-is
                    validated[key] = value
                    
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse config key '{key}' with value {value}: {e}")
            continue
    
    # Log unknown keys as warnings
    if unknown_keys:
        logger.warning(
            f"Ignored {len(unknown_keys)} unknown config key(s): {unknown_keys}. "
            f"Only whitelisted keys are allowed: {sorted(ALLOWED_RUNTIME_KEYS)}"
        )
    
    # Log rejected keys
    if rejected_keys:
        logger.warning(
            f"Rejected {len(rejected_keys)} config key(s) with invalid values: {rejected_keys}"
        )
    
    return validated

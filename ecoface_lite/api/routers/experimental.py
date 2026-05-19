"""Experimental configuration API endpoints.

Stage 2 (Module M2): Runtime override management and reset controller.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from ecoface_lite.core.experimental_config import (
    validate_and_clamp_config,
    get_default_runtime_config,
)
from ecoface_lite.core.runtime_state import get_runtime_state
from ecoface_lite.core.action_logging import DashboardActionLog, DashboardAction
from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/experimental", tags=["experimental"])


class OverrideRequest(BaseModel):
    """Request model for applying runtime overrides."""
    config: dict


class OverrideResponse(BaseModel):
    """Response model for override application."""
    applied: dict
    warnings: list[str] = []


class ResetResponse(BaseModel):
    """Response model for reset operation."""
    status: str
    previous_session_id: str
    new_session_id: str


@router.post("/overrides", response_model=OverrideResponse)
async def apply_overrides(request: OverrideRequest) -> OverrideResponse:
    """Apply experimental runtime overrides with validation.
    
    Validates against strict whitelist, clamps parameter bounds,
    and returns applied configuration with any warnings.
    """
    runtime_state = get_runtime_state()
    
    # Validate and clamp configuration
    validated = validate_and_clamp_config(request.config)
    
    # Apply to runtime state
    applied = runtime_state.apply_overrides(validated)
    
    # Log action
    logger.info(f"Applied runtime overrides: {list(applied.keys())}")
    
    return OverrideResponse(applied=applied)


@router.get("/overrides")
async def get_current_overrides() -> dict:
    """Get current experimental runtime overrides."""
    runtime_state = get_runtime_state()
    return {
        "overrides": runtime_state.get_overrides(),
        "experiment_session_id": runtime_state.get_experiment_session_id(),
        "last_override_timestamp": runtime_state.get_last_override_timestamp(),
        "backend_type": runtime_state.get_backend_type().value,
    }


@router.post("/reset", response_model=ResetResponse)
async def reset_experimental_settings() -> ResetResponse:
    """Reset all experimental overrides and CPU protection state.
    
    Clears:
    - All runtime override parameters
    - CPU protection state metrics
    - Adaptive detection interval elevation
    - Embedding suppression flags
    - Debug truncation flags
    - Retry counter logs
    - Overload event counters
    - Recovery stable frame counters
    - Last protection trigger time
    - Last override timestamp
    - Experiment session ID (regenerated)
    """
    runtime_state = get_runtime_state()
    
    # Capture previous session ID for response
    old_session_id = runtime_state.get_experiment_session_id()
    
    # Clear all state (includes session ID regeneration)
    runtime_state.clear_overrides()
    
    # Log reset action
    logger.info(
        f"Reset experimental settings. Previous session: {old_session_id}, "
        f"New session: {runtime_state.get_experiment_session_id()}"
    )
    
    return ResetResponse(
        status="reset_complete",
        previous_session_id=old_session_id,
        new_session_id=runtime_state.get_experiment_session_id(),
    )


@router.get("/cpu-protection")
async def get_cpu_protection_state() -> dict:
    """Get current CPU protection state."""
    runtime_state = get_runtime_state()
    protection = runtime_state.get_cpu_protection_state()
    return {
        "protection_active": protection.protection_active,
        "current_detection_interval": protection.current_detection_interval,
        "embedding_suppression_active": protection.embedding_suppression_active,
        "debug_truncation_active": protection.debug_truncation_active,
        "overload_event_count": protection.overload_event_count,
        "recovery_stable_frames": protection.recovery_stable_frames,
        "last_protection_trigger_time": protection.last_protection_trigger_time.isoformat() if protection.last_protection_trigger_time else None,
    }


@router.post("/backend-type")
async def set_backend_type(backend_type: str) -> dict:
    """Set backend type for CPU protection gating."""
    from ecoface_lite.core.experimental_config import BackendType
    
    runtime_state = get_runtime_state()
    
    try:
        backend_enum = BackendType(backend_type.upper())
        runtime_state.set_backend_type(backend_enum)
        logger.info(f"Set backend type to: {backend_enum.value}")
        return {"backend_type": backend_enum.value, "status": "set"}
    except ValueError:
        logger.warning(f"Invalid backend type requested: {backend_type}")
        return {"backend_type": runtime_state.get_backend_type().value, "status": "invalid", "error": f"Invalid backend type: {backend_type}"}


@router.get("/defaults")
async def get_default_config() -> dict:
    """Get default runtime configuration for reference."""
    return get_default_runtime_config()


@router.get("/snapshots")
async def get_experiment_snapshots(limit: int = 20) -> dict:
    """Get recent experiment snapshots from storage.
    
    Args:
        limit: Maximum number of recent snapshots to return (default 20).
    
    Returns:
        List of experiment snapshot dictionaries sorted by timestamp (newest first).
    """
    from ecoface_lite.core.experiment_storage import ExperimentStorage
    from ecoface_lite.core.config import get_settings
    
    settings = get_settings()
    storage = ExperimentStorage(settings.resolved_experiments_dir())
    
    snapshots = storage.load_recent_snapshots(limit=limit)
    
    return {
        "snapshots": snapshots,
        "count": len(snapshots),
    }


@router.get("/actions")
async def get_action_log(limit: int = 100) -> dict:
    """Get recent action log entries.
    
    Args:
        limit: Maximum number of recent action entries to return (default 100).
    
    Returns:
        List of action log entries sorted by timestamp (newest first).
    """
    from ecoface_lite.core.action_logging import DashboardActionLog
    from ecoface_lite.core.config import get_settings
    
    settings = get_settings()
    action_log = DashboardActionLog(settings.resolved_log_dir())
    
    actions = action_log.load_recent_actions(limit=limit)
    
    return {
        "actions": actions,
        "count": len(actions),
    }

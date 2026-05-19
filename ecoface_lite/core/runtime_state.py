"""Thread-safe runtime override state buffer.

Stage 2 (Module M2): Runtime override storage, CPU protection state,
and experiment session tracking. Single-responsibility: state management only.
"""

import dataclasses
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid

from ecoface_lite.core.experimental_config import (
    validate_and_clamp_config,
    BackendType,
)

# Singleton instance for dependency injection
_runtime_override_state_instance: Optional["RuntimeOverrideState"] = None
_state_lock = threading.Lock()


def get_runtime_state() -> "RuntimeOverrideState":
    """Get singleton RuntimeOverrideState instance (thread-safe)."""
    global _runtime_override_state_instance
    if _runtime_override_state_instance is None:
        with _state_lock:
            if _runtime_override_state_instance is None:
                _runtime_override_state_instance = RuntimeOverrideState()
    return _runtime_override_state_instance


@dataclass
class CpuProtectionState:
    """Hysteresis CPU protection state (LOCAL_CPU only)."""
    protection_active: bool = False
    current_detection_interval: int = 8
    embedding_suppression_active: bool = False
    debug_truncation_active: bool = False
    overload_event_count: int = 0
    recovery_stable_frames: int = 0
    last_protection_trigger_time: Optional[datetime] = None


@dataclass
class RuntimeOverrideState:
    """Thread-safe memory buffer for experimental runtime overrides.
    
    Responsibilities:
    - Override storage and validation
    - CPU protection state management
    - Experiment session tracking
    
    Does NOT handle:
    - FPS calculations (handled by diagnostics)
    - Metrics aggregation (handled by metrics system)
    - Retry logic (handled by pipeline)
    - Diagnostics persistence (handled by storage)
    """
    _overrides: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _cpu_protection: CpuProtectionState = field(default_factory=CpuProtectionState)
    _backend_type: BackendType = BackendType.LOCAL_CPU
    _experiment_session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _last_override_timestamp: Optional[datetime] = None
    
    def apply_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and apply experimental overrides (thread-safe)."""
        validated = validate_and_clamp_config(config_dict)
        with self._lock:
            self._overrides.update(validated)
            self._last_override_timestamp = datetime.utcnow()
        return validated
    
    def get_overrides(self) -> Dict[str, Any]:
        """Get current overrides (thread-safe)."""
        with self._lock:
            return self._overrides.copy()
    
    def clear_overrides(self) -> None:
        """Clear all experimental overrides (thread-safe).
        
        Clears:
        - All runtime override parameters
        - CPU protection state
        - Experiment session ID (regenerated)
        - Override timestamp
        """
        with self._lock:
            self._overrides.clear()
            self._cpu_protection = CpuProtectionState()
            self._experiment_session_id = str(uuid.uuid4())
            self._last_override_timestamp = None
    
    def set_backend_type(self, backend_type: BackendType) -> None:
        """Set backend type for CPU protection gating."""
        with self._lock:
            self._backend_type = backend_type
    
    def get_backend_type(self) -> BackendType:
        """Get current backend type."""
        with self._lock:
            return self._backend_type
    
    def get_experiment_session_id(self) -> str:
        """Get current experiment session ID."""
        with self._lock:
            return self._experiment_session_id
    
    def get_last_override_timestamp(self) -> Optional[datetime]:
        """Get last override timestamp."""
        with self._lock:
            return self._last_override_timestamp
    
    def get_cpu_protection_state(self) -> CpuProtectionState:
        """Get CPU protection state (thread-safe, returns COPY)."""
        with self._lock:
            return dataclasses.replace(self._cpu_protection)
    
    def update_cpu_protection(
        self,
        current_fps: float,
        frame_stable: bool = True,
        configured_baseline_interval: int = 8,
    ) -> CpuProtectionState:
        """Update hysteresis CPU protection based on FPS (LOCAL_CPU only).
        
        Args:
            current_fps: Current frames per second.
            frame_stable: Whether current frame is stable.
            configured_baseline_interval: Configured baseline detection interval
                from effective config (recovery decays toward this, not hardcoded).
        
        Returns:
            Updated CPU protection state (copy).
        
        Hysteresis logic:
        - Enable protection below PROTECTION_ENABLE_FPS (1.0)
        - Disable protection above PROTECTION_DISABLE_FPS (1.8) after sustained stability
        - Gradual recovery: decrement interval step by step toward baseline
        """
        PROTECTION_ENABLE_FPS = 1.0
        PROTECTION_DISABLE_FPS = 1.8
        MAX_DETECTION_INTERVAL = 18
        RECOVERY_STABLE_FRAMES_THRESHOLD = 30
        
        with self._lock:
            if self._backend_type != BackendType.LOCAL_CPU:
                return dataclasses.replace(self._cpu_protection)
            
            if self._cpu_protection.protection_active:
                # Protection active: track recovery progress
                if frame_stable:
                    self._cpu_protection.recovery_stable_frames += 1
                else:
                    self._cpu_protection.recovery_stable_frames = 0
                
                # Only disable after sustained stability above threshold
                if (current_fps >= PROTECTION_DISABLE_FPS and 
                    self._cpu_protection.recovery_stable_frames >= RECOVERY_STABLE_FRAMES_THRESHOLD):
                    # Gradual recovery: decay toward configured baseline
                    if self._cpu_protection.current_detection_interval > configured_baseline_interval:
                        self._cpu_protection.current_detection_interval -= 1
                        # Stay in protection mode until fully recovered
                    else:
                        # Fully recovered, disable protection
                        self._cpu_protection.protection_active = False
                        self._cpu_protection.embedding_suppression_active = False
                        self._cpu_protection.debug_truncation_active = False
                        self._cpu_protection.recovery_stable_frames = 0
            else:
                # Protection inactive: enable below threshold
                if current_fps < PROTECTION_ENABLE_FPS:
                    self._cpu_protection.protection_active = True
                    self._cpu_protection.current_detection_interval = min(
                        self._cpu_protection.current_detection_interval + 1,
                        MAX_DETECTION_INTERVAL
                    )
                    self._cpu_protection.embedding_suppression_active = True
                    self._cpu_protection.debug_truncation_active = True
                    self._cpu_protection.overload_event_count += 1
                    self._cpu_protection.last_protection_trigger_time = datetime.utcnow()
                    self._cpu_protection.recovery_stable_frames = 0
            
            return dataclasses.replace(self._cpu_protection)

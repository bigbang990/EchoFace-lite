"""Pipeline execution context and adaptive runtime state.

Stage 2 (Module M2): Thread-safe execution context for pipeline processing.
Explicit context passing - no mutable state stored on pipeline instances.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecoface_lite.core.runtime_config import EffectiveRuntimeConfig


@dataclass
class CpuAdaptiveRuntimeState:
    """Mutable lightweight runtime overlay for CPU protection only.
    
    This contains only the dynamic values that change during frame processing
    due to CPU protection. All other configuration is immutable in EffectiveRuntimeConfig.
    
    Responsibilities:
    - Hold current adaptive detection interval (may be elevated by protection)
    - Track embedding suppression state
    - Track debug truncation state
    
    Does NOT hold:
    - FPS calculations (handled by diagnostics)
    - Metrics (handled by metrics system)
    - Configuration values (in EffectiveRuntimeConfig)
    """
    current_detection_interval: int
    embedding_suppression_active: bool
    debug_truncation_active: bool
    
    def update_from_protection(self, protection_interval: int, suppression: bool, truncation: bool) -> None:
        """Update adaptive state from protection state (mutable, lightweight)."""
        self.current_detection_interval = protection_interval
        self.embedding_suppression_active = suppression
        self.debug_truncation_active = truncation


@dataclass
class PipelineExecutionContext:
    """Execution context for pipeline with compiled config and adaptive state.
    
    This context is passed explicitly through all pipeline methods.
    It is NOT stored on self to ensure thread-safety for concurrent jobs.
    
    Architecture:
    - config: Immutable EffectiveRuntimeConfig (compiled once per job)
    - cpu_adaptive: Mutable CpuAdaptiveRuntimeState (updates during processing)
    
    Thread-safety:
    - Each job gets its own context instance
    - Context is passed explicitly, never stored on pipeline
    - No shared mutable state between concurrent jobs
    """
    config: "EffectiveRuntimeConfig"
    cpu_adaptive: CpuAdaptiveRuntimeState

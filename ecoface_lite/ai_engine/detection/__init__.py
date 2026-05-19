"""Detection layer for Phase 2A Enterprise Detection Upgrade."""

from ecoface_lite.ai_engine.detection.detectors import (
    BaseDetector,
    SCRFDDetector,
    MultiScaleDetector,
)

__all__ = [
    "BaseDetector",
    "SCRFDDetector",
    "MultiScaleDetector",
]

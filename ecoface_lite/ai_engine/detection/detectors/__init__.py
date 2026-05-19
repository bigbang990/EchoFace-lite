"""Detection layer for Phase 2A Enterprise Detection Upgrade."""

from ecoface_lite.ai_engine.detection.detectors.base_detector import BaseDetector
from ecoface_lite.ai_engine.detection.detectors.scrfd_detector import SCRFDDetector
from ecoface_lite.ai_engine.detection.detectors.multiscale_detector import MultiScaleDetector

__all__ = [
    "BaseDetector",
    "SCRFDDetector",
    "MultiScaleDetector",
]

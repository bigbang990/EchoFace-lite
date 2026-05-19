"""Proposal fusion engine for Phase 2A Enterprise Detection Upgrade."""

from ecoface_lite.ai_engine.detection.fusion.weighted_box_fusion import WeightedBoxFusion
from ecoface_lite.ai_engine.detection.fusion.duplicate_filter import DuplicateFilter
from ecoface_lite.ai_engine.detection.fusion.confidence_normalizer import ConfidenceNormalizer

__all__ = [
    "WeightedBoxFusion",
    "DuplicateFilter",
    "ConfidenceNormalizer",
]

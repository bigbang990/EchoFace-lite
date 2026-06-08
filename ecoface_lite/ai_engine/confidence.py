from __future__ import annotations

from dataclasses import dataclass

from ecoface_lite.ai_engine.face_quality import FaceQualityResult
from ecoface_lite.ai_engine.preprocessing import FrameDiagnostics
from ecoface_lite.core.config import Settings


@dataclass(frozen=True)
class ConfidenceDecision:
    accepted: bool
    adjusted_threshold: float
    raw_confidence: float


class ConfidencePolicy:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def threshold_for(self, diagnostics: FrameDiagnostics | None, quality: FaceQualityResult) -> float:
        threshold = self._settings.confidence_large_face_threshold
        if quality.quality_score < 0.55:
            threshold = self._settings.confidence_small_face_threshold
        
        # Defensive check for brightness access
        if diagnostics is not None and hasattr(diagnostics, "brightness"):
            if diagnostics.brightness < self._settings.confidence_low_light_threshold:
                threshold += self._settings.confidence_low_light_margin
                
        if quality.blur_score < self._settings.face_quality_min_blur_score * 1.5:
            threshold += self._settings.confidence_blur_margin
        return min(0.99, threshold)

    def decide(self, confidence: float, diagnostics: FrameDiagnostics, quality: FaceQualityResult) -> ConfidenceDecision:
        threshold = self.threshold_for(diagnostics, quality)
        return ConfidenceDecision(confidence >= threshold, threshold, confidence)

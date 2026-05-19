"""Confidence normalizer for multi-scale proposals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration for confidence normalization."""
    enable_scale_boost: bool = True
    tiny_scale_boost: float = 0.15
    small_scale_boost: float = 0.10
    baseline_boost: float = 0.0
    max_confidence: float = 1.0
    min_confidence: float = 0.0
    tiny_threshold: int = 30
    small_threshold: int = 60


class ConfidenceNormalizer:
    """Normalize confidence scores across different scales and sources."""

    def __init__(self, config: NormalizationConfig | None = None) -> None:
        self._config = config or NormalizationConfig()

    def normalize(
        self,
        faces: list[DetectedFace],
    ) -> list[DetectedFace]:
        """Normalize confidence scores across all faces.

        Args:
            faces: List of detected faces

        Returns:
            List of faces with normalized confidence
        """
        if not faces:
            return []

        if not self._config.enable_scale_boost:
            return faces

        normalized_faces = []
        for face in faces:
            normalized = self._normalize_face(face)
            normalized_faces.append(normalized)

        return normalized_faces

    def _normalize_face(self, face: DetectedFace) -> DetectedFace:
        """Normalize confidence for a single face.

        Args:
            face: Face to normalize

        Returns:
            Face with normalized confidence
        """
        # Calculate face size
        face_size = max(
            face.bbox.x2 - face.bbox.x1,
            face.bbox.y2 - face.bbox.y1,
        )

        # Apply scale-based boost
        boost = 0.0
        if face_size < self._config.tiny_threshold:
            boost = self._config.tiny_scale_boost
        elif face_size < self._config.small_threshold:
            boost = self._config.small_scale_boost
        else:
            boost = self._config.baseline_boost

        # Apply boost to confidence
        normalized_confidence = face.det_score + boost

        # Clamp to valid range
        normalized_confidence = max(
            self._config.min_confidence,
            min(self._config.max_confidence, normalized_confidence),
        )

        # Create normalized face
        normalized_face = DetectedFace(
            bbox=face.bbox,
            det_score=normalized_confidence,
            aligned_face=face.aligned_face,
            embedding=face.embedding,
            landmarks=face.landmarks,
            temporal_score=face.temporal_score,
        )

        return normalized_face

    def normalize_temporal_scores(
        self,
        faces: list[DetectedFace],
    ) -> list[DetectedFace]:
        """Normalize temporal scores to match detector confidence range.

        Args:
            faces: List of faces with temporal scores

        Returns:
            List of faces with normalized temporal scores
        """
        normalized_faces = []
        for face in faces:
            if face.temporal_score is not None:
                # Temporal scores are already in 0-1 range, just ensure they're valid
                normalized_temporal = max(
                    self._config.min_confidence,
                    min(self._config.max_confidence, face.temporal_score),
                )

                normalized_face = DetectedFace(
                    bbox=face.bbox,
                    det_score=face.det_score,
                    aligned_face=face.aligned_face,
                    embedding=face.embedding,
                    landmarks=face.landmarks,
                    temporal_score=normalized_temporal,
                )
                normalized_faces.append(normalized_face)
            else:
                normalized_faces.append(face)

        return normalized_faces

    def get_confidence_statistics(
        self,
        faces: list[DetectedFace],
    ) -> dict[str, float]:
        """Get statistics about confidence scores.

        Args:
            faces: List of faces

        Returns:
            Dictionary with confidence statistics
        """
        if not faces:
            return {}

        confidences = [f.det_score for f in faces]

        return {
            "mean_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
            "median_confidence": float(np.median(confidences)),
        }

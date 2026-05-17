"""Temporal agreement for detector proposals across refresh cycles."""

from __future__ import annotations

from dataclasses import dataclass, field

from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace
from ecoface_lite.ai_engine.geometry import bbox_iou
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


@dataclass
class _TemporalHit:
    bbox: BoundingBox
    score: float
    frame_index: int


class TemporalDetectorFilter:
    """Blend current detection confidence with recent spatially-matched history."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._history: list[_TemporalHit] = []
        self._max_history = 32

    def apply(self, faces: list[DetectedFace], frame_index: int) -> list[DetectedFace]:
        alpha = self._settings.detector_temporal_blend_current
        beta = self._settings.detector_temporal_blend_history
        out: list[DetectedFace] = []
        for face in faces:
            historical = self._match_history(face.bbox)
            blended = face.det_score
            if historical is not None:
                blended = (alpha * face.det_score) + (beta * historical)
                metrics.increment("temporal_detector_agreements")
            temporal_score = float(min(1.0, max(0.0, blended)))
            out.append(
                DetectedFace(
                    bbox=face.bbox,
                    det_score=face.det_score,
                    aligned_face=face.aligned_face,
                    embedding=face.embedding,
                    landmarks=face.landmarks,
                    temporal_score=temporal_score,
                )
            )
            self._history.append(_TemporalHit(bbox=face.bbox, score=temporal_score, frame_index=frame_index))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        return out

    def _match_history(self, bbox: BoundingBox) -> float | None:
        best_score: float | None = None
        best_iou = 0.0
        iou_thresh = self._settings.detector_temporal_iou_match
        for hit in reversed(self._history):
            iou = bbox_iou(bbox, hit.bbox)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_score = hit.score
        return best_score

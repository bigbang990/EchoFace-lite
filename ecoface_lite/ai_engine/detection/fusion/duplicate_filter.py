"""Duplicate filter for removing redundant proposals."""

from __future__ import annotations

from dataclasses import dataclass

from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace
from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for duplicate filtering."""
    iou_threshold: float = 0.5
    crowd_iou_threshold: float = 0.3
    confidence_threshold: float = 0.3
    prefer_higher_confidence: bool = True
    prefer_larger_scale: bool = True


class DuplicateFilter:
    """Remove duplicate proposals while preserving crowd detections."""

    def __init__(self, config: FilterConfig | None = None) -> None:
        self._config = config or FilterConfig()

    def filter(
        self,
        faces: list[DetectedFace],
        is_crowd_scene: bool = False,
    ) -> list[DetectedFace]:
        """Filter duplicate proposals using IoU-based suppression.

        Args:
            faces: List of detected faces
            is_crowd_scene: Whether this is a crowded scene (uses lower IoU threshold)

        Returns:
            List of filtered faces
        """
        if not faces:
            return []

        if len(faces) == 1:
            return faces

        # Use crowd-aware IoU threshold
        iou_threshold = (
            self._config.crowd_iou_threshold
            if is_crowd_scene
            else self._config.iou_threshold
        )

        # Sort by confidence (highest first)
        sorted_faces = sorted(faces, key=lambda f: f.det_score, reverse=True)

        filtered: list[DetectedFace] = []
        suppressed = [False] * len(sorted_faces)

        for i, face in enumerate(sorted_faces):
            if suppressed[i]:
                continue

            # Keep this face
            filtered.append(face)

            # Suppress overlapping faces with lower confidence
            for j in range(i + 1, len(sorted_faces)):
                if suppressed[j]:
                    continue

                iou = self._calculate_iou(face.bbox, sorted_faces[j].bbox)

                if iou >= iou_threshold:
                    # Suppress if current face is better
                    if self._should_suppress(face, sorted_faces[j]):
                        suppressed[j] = True
                        logger.debug(
                            "Suppressed duplicate: iou=%.3f, conf=%.3f vs %.3f",
                            iou,
                            face.det_score,
                            sorted_faces[j].det_score,
                        )

        return filtered

    def _should_suppress(
        self,
        keep_face: DetectedFace,
        suppress_face: DetectedFace,
    ) -> bool:
        """Determine whether to suppress a face based on configuration.

        Args:
            keep_face: The face to keep
            suppress_face: The face to potentially suppress

        Returns:
            True if suppress_face should be suppressed
        """
        # Confidence threshold check
        if suppress_face.det_score < self._config.confidence_threshold:
            return True

        # Prefer higher confidence
        if self._config.prefer_higher_confidence:
            if keep_face.det_score > suppress_face.det_score:
                return True

        # Prefer larger scale (if size information available)
        if self._config.prefer_larger_scale:
            keep_size = max(
                keep_face.bbox.x2 - keep_face.bbox.x1,
                keep_face.bbox.y2 - keep_face.bbox.y1,
            )
            suppress_size = max(
                suppress_face.bbox.x2 - suppress_face.bbox.x1,
                suppress_face.bbox.y2 - suppress_face.bbox.y1,
            )
            if keep_size > suppress_size * 1.2:  # Keep if 20% larger
                return True

        return False

    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate IoU between two bounding boxes.

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            IoU score
        """
        # Calculate intersection
        x1_inter = max(bbox1.x1, bbox2.x1)
        y1_inter = max(bbox1.y1, bbox2.y1)
        x2_inter = min(bbox1.x2, bbox2.x2)
        y2_inter = min(bbox1.y2, bbox2.y2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union
        area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

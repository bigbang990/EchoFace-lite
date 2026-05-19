"""Weighted Box Fusion for merging multi-scale and multi-source proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace
from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FusionConfig:
    """Configuration for proposal fusion."""
    iou_threshold: float = 0.5
    crowd_iou_threshold: float = 0.3
    scale_weight_tiny: float = 1.3
    scale_weight_small: float = 1.1
    scale_weight_baseline: float = 1.0
    skip_weight: float = 0.5  # Weight for boxes that would be NMS'd


class WeightedBoxFusion:
    """Merge overlapping proposals from multiple scales/sources using weighted averaging."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self._config = config or FusionConfig()

    def fuse(
        self,
        faces: list[DetectedFace],
        frame_shape: tuple[int, int],
        is_crowd_scene: bool = False,
    ) -> list[DetectedFace]:
        """Fuse overlapping face proposals using weighted box fusion.

        Args:
            faces: List of detected faces from multiple scales/sources
            frame_shape: Frame dimensions (height, width)
            is_crowd_scene: Whether this is a crowded scene (uses lower IoU threshold)

        Returns:
            List of fused faces
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

        # Group faces by spatial overlap
        clusters = self._cluster_by_iou(faces, iou_threshold)

        # Fuse each cluster
        fused_faces = []
        for cluster in clusters:
            if len(cluster) == 1:
                fused_faces.append(cluster[0])
            else:
                fused = self._fuse_cluster(cluster)
                fused_faces.append(fused)

        return fused_faces

    def _cluster_by_iou(
        self,
        faces: list[DetectedFace],
        iou_threshold: float,
    ) -> list[list[DetectedFace]]:
        """Cluster faces by IoU overlap.

        Args:
            faces: List of faces to cluster
            iou_threshold: IoU threshold for clustering

        Returns:
            List of clusters (each cluster is a list of faces)
        """
        if not faces:
            return []

        # Sort by confidence (highest first)
        sorted_faces = sorted(faces, key=lambda f: f.det_score, reverse=True)

        clusters: list[list[DetectedFace]] = []
        assigned = [False] * len(sorted_faces)

        for i, face in enumerate(sorted_faces):
            if assigned[i]:
                continue

            # Start new cluster
            cluster = [face]
            assigned[i] = True

            # Find overlapping faces
            for j in range(i + 1, len(sorted_faces)):
                if assigned[j]:
                    continue

                iou = self._calculate_iou(face.bbox, sorted_faces[j].bbox)
                if iou >= iou_threshold:
                    cluster.append(sorted_faces[j])
                    assigned[j] = True

            clusters.append(cluster)

        return clusters

    def _fuse_cluster(self, cluster: list[DetectedFace]) -> DetectedFace:
        """Fuse a cluster of overlapping faces using weighted averaging.

        Args:
            cluster: List of overlapping faces

        Returns:
            Fused face
        """
        if len(cluster) == 1:
            return cluster[0]

        # Calculate weights based on confidence and scale
        weights = []
        for face in cluster:
            # Base weight from confidence
            weight = face.det_score

            # Scale-based weight boost (if metadata available)
            if hasattr(face, 'bbox'):
                face_size = max(face.bbox.x2 - face.bbox.x1, face.bbox.y2 - face.bbox.y1)
                if face_size < 30:
                    weight *= self._config.scale_weight_tiny
                elif face_size < 60:
                    weight *= self._config.scale_weight_small
                else:
                    weight *= self._config.scale_weight_baseline

            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(cluster)] * len(cluster)
        else:
            weights = [w / total_weight for w in weights]

        # Weighted average of bbox coordinates
        x1 = sum(f.bbox.x1 * w for f, w in zip(cluster, weights))
        y1 = sum(f.bbox.y1 * w for f, w in zip(cluster, weights))
        x2 = sum(f.bbox.x2 * w for f, w in zip(cluster, weights))
        y2 = sum(f.bbox.y2 * w for f, w in zip(cluster, weights))

        # Weighted average of confidence
        fused_confidence = sum(f.det_score * w for f, w in zip(cluster, weights))

        # Use landmarks from highest confidence face
        best_face = max(cluster, key=lambda f: f.det_score)

        # Preserve other attributes from best face
        fused_face = DetectedFace(
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            det_score=fused_confidence,
            aligned_face=best_face.aligned_face,
            embedding=best_face.embedding,
            landmarks=best_face.landmarks,
            temporal_score=best_face.temporal_score,
        )

        return fused_face

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

"""Weak detection memory for temporal recovery of unstable faces."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ecoface_lite.ai_engine.detector import BoundingBox
from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WeakProposal:
    """A weak detection proposal stored in temporal memory."""
    bbox: BoundingBox
    confidence: float
    frame_id: int
    timestamp: float
    detection_count: int = 1
    last_seen_frame: int = 0


@dataclass
class MemoryConfig:
    """Configuration for weak detection memory."""
    max_frames: int = 32
    cluster_iou: float = 0.4
    min_recurrence: int = 3
    promotion_boost: float = 0.15
    max_boost: float = 0.4
    motion_threshold: float = 2.0


class WeakDetectionMemory:
    """Short-term memory for weak detection proposals with temporal consistency."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._config = config or MemoryConfig()
        self._proposals: deque[WeakProposal] = deque(maxlen=self._config.max_frames)
        self._current_frame_id = 0

    def update(
        self,
        faces: list[Any],
        frame_id: int,
    ) -> list[tuple[Any, float]]:
        """Update memory with current frame detections and return promoted faces.

        Args:
            faces: List of detected faces
            frame_id: Current frame index

        Returns:
            List of (face, boost_amount) tuples for promoted faces
        """
        self._current_frame_id = frame_id

        # Cluster current faces with existing proposals
        promoted = self._cluster_and_update(faces)

        # Prune old proposals
        self._prune_old_proposals()

        return promoted

    def _cluster_and_update(
        self,
        faces: list[Any],
    ) -> list[tuple[Any, float]]:
        """Cluster current faces with existing proposals and update counts.

        Args:
            faces: List of detected faces

        Returns:
            List of promoted faces with boost amounts
        """
        promoted: list[tuple[Any, float]] = []

        for face in faces:
            matched = False
            for proposal in self._proposals:
                iou = self._calculate_iou(face.bbox, proposal.bbox)

                if iou >= self._config.cluster_iou:
                    # Update existing proposal
                    proposal.detection_count += 1
                    proposal.last_seen_frame = self._current_frame_id
                    matched = True

                    # Check for promotion
                    if proposal.detection_count >= self._config.min_recurrence:
                        boost = min(
                            self._config.max_boost,
                            proposal.detection_count * self._config.promotion_boost,
                        )
                        promoted.append((face, boost))
                        logger.debug(
                            "Promoted weak face: frame=%s count=%s boost=%.3f",
                            self._current_frame_id,
                            proposal.detection_count,
                            boost,
                        )
                    break

            if not matched:
                # Add new proposal
                new_proposal = WeakProposal(
                    bbox=face.bbox,
                    confidence=face.det_score,
                    frame_id=self._current_frame_id,
                    timestamp=np.datetime64("now").astype(float) / 1e9,
                    detection_count=1,
                    last_seen_frame=self._current_frame_id,
                )
                self._proposals.append(new_proposal)

        return promoted

    def _prune_old_proposals(self) -> None:
        """Remove proposals that haven't been seen recently."""
        current_proposals = []
        frames_since_last = self._current_frame_id

        for proposal in self._proposals:
            frames_since_last = self._current_frame_id - proposal.last_seen_frame

            # Keep if seen recently or has high detection count
            if frames_since_last < 16 or proposal.detection_count >= self._config.min_recurrence:
                current_proposals.append(proposal)

        self._proposals = deque(current_proposals, maxlen=self._config.max_frames)

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

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the current memory state.

        Returns:
            Dictionary with memory statistics
        """
        if not self._proposals:
            return {
                "total_proposals": 0,
                "avg_detection_count": 0.0,
                "max_detection_count": 0,
                "promoted_count": 0,
            }

        detection_counts = [p.detection_count for p in self._proposals]
        promoted_count = sum(1 for p in self._proposals if p.detection_count >= self._config.min_recurrence)

        return {
            "total_proposals": len(self._proposals),
            "avg_detection_count": float(np.mean(detection_counts)),
            "max_detection_count": max(detection_counts),
            "promoted_count": promoted_count,
        }

    def reset(self) -> None:
        """Reset the memory (useful for scene changes)."""
        self._proposals.clear()

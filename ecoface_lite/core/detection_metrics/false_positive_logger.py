"""False positive logger for hard negative dataset collection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


class FalsePositiveCategory(str, Enum):
    """Categories of false positives for dataset organization."""
    POSTER = "poster"
    PAINTING = "painting"
    STATUE = "statue"
    MANNEQUIN = "mannequin"
    TV_FACE = "tv_face"
    PRINTED_FACE = "printed_face"
    CLOTH_PATTERN = "cloth_pattern"
    UNKNOWN = "unknown"


@dataclass
class FalsePositiveRecord:
    """Record of a false positive detection."""
    frame_id: int
    timestamp: float
    bbox: tuple[float, float, float, float]
    confidence: float
    category: str
    rejection_reason: str
    snapshot_path: str | None
    metadata: dict[str, Any]


class FalsePositiveLogger:
    """Logs and stores false positive detections for hard negative dataset."""

    def __init__(
        self,
        base_dir: Path,
        enabled: bool = True,
        max_snapshots: int = 1000,
        sampling_rate: float = 0.10,
        min_confidence: float = 0.60,
    ) -> None:
        self._base_dir = base_dir
        self._enabled = enabled
        self._max_snapshots = max_snapshots
        self._sampling_rate = sampling_rate
        self._min_confidence = min_confidence
        self._records: list[FalsePositiveRecord] = []
        self._category_counts: dict[str, int] = {cat.value: 0 for cat in FalsePositiveCategory}

        if self._enabled:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            for category in FalsePositiveCategory:
                cat_dir = self._base_dir / category.value
                cat_dir.mkdir(parents=True, exist_ok=True)

    def log_false_positive(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        bbox: tuple[float, float, float, float],
        confidence: float,
        rejection_reason: str,
        category: str = FalsePositiveCategory.UNKNOWN.value,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a false positive detection."""
        if not self._enabled:
            return

        # Sampling check
        import random
        if random.random() > self._sampling_rate:
            return

        # Confidence filter
        if confidence < self._min_confidence:
            return

        snapshot_path = self._save_snapshot(frame_bgr, bbox, frame_id, category)

        record = FalsePositiveRecord(
            frame_id=frame_id,
            timestamp=time.time(),
            bbox=bbox,
            confidence=confidence,
            category=category,
            rejection_reason=rejection_reason,
            snapshot_path=snapshot_path,
            metadata=metadata or {},
        )

        self._records.append(record)
        self._category_counts[category] = self._category_counts.get(category, 0) + 1

        # Cleanup if exceeding max
        if len(self._records) > self._max_snapshots:
            self._cleanup_old_snapshots()

    def _save_snapshot(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_id: int,
        category: str,
    ) -> str | None:
        """Save face crop snapshot."""
        try:
            x1, y1, x2, y2 = bbox
            xi1 = max(0, int(x1))
            yi1 = max(0, int(y1))
            xi2 = min(frame_bgr.shape[1], int(x2))
            yi2 = min(frame_bgr.shape[0], int(y2))

            if xi2 <= xi1 or yi2 <= yi1:
                return None

            crop = frame_bgr[yi1:yi2, xi1:xi2]
            if crop.size == 0:
                return None

            ts = int(time.time() * 1000)
            safe_reason = category.replace(" ", "_").replace("/", "_")
            filename = f"fp_{frame_id:06d}_{ts}_{safe_reason}.jpg"
            filepath = self._base_dir / category / filename

            cv2.imwrite(str(filepath), crop)
            return str(filepath)

        except Exception as e:
            logger.warning(f"Failed to save false positive snapshot: {e}")
            return None

    def _cleanup_old_snapshots(self) -> None:
        """Remove oldest snapshots to stay within limit."""
        # Remove oldest records
        excess = len(self._records) - self._max_snapshots
        for i in range(excess):
            old_record = self._records[i]
            if old_record.snapshot_path:
                try:
                    Path(old_record.snapshot_path).unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete old snapshot {old_record.snapshot_path}: {e}")

        self._records = self._records[self._max_snapshots :]

        # Update category counts
        self._category_counts = {cat.value: 0 for cat in FalsePositiveCategory}
        for record in self._records:
            self._category_counts[record.category] = self._category_counts.get(record.category, 0) + 1

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about collected false positives."""
        return {
            "total_records": len(self._records),
            "category_counts": self._category_counts.copy(),
            "category_breakdown": {
                cat: count for cat, count in self._category_counts.items() if count > 0
            },
            "sampling_rate": self._sampling_rate,
            "min_confidence": self._min_confidence,
        }

    def export_metadata(self, export_path: Path) -> None:
        """Export false positive metadata to JSON file."""
        import json

        metadata_list = []
        for record in self._records:
            metadata_list.append({
                "frame_id": record.frame_id,
                "timestamp": record.timestamp,
                "bbox": record.bbox,
                "confidence": record.confidence,
                "category": record.category,
                "rejection_reason": record.rejection_reason,
                "snapshot_path": record.snapshot_path,
                "metadata": record.metadata,
            })

        with open(export_path, "w") as f:
            json.dump({
                "statistics": self.get_statistics(),
                "records": metadata_list,
            }, f, indent=2)

        logger.info(f"Exported false positive metadata to {export_path}")

    def reset(self) -> None:
        """Reset all records and snapshots."""
        for record in self._records:
            if record.snapshot_path:
                try:
                    Path(record.snapshot_path).unlink()
                except Exception:
                    pass

        self._records.clear()
        self._category_counts = {cat.value: 0 for cat in FalsePositiveCategory}

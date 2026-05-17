"""Rolling, quality-weighted embedding memory per track."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


@dataclass
class MemorySample:
    embedding: np.ndarray
    quality: float
    person_id: int | None
    frame_index: int


@dataclass
class IdentityMemoryBank:
    """Stores representative high-quality embeddings for temporal verification."""

    max_samples: int = 12
    min_quality: float = 0.45
    samples: deque[MemorySample] = field(default_factory=deque)

    @classmethod
    def from_settings(cls, settings: Settings) -> IdentityMemoryBank:
        return cls(
            max_samples=settings.tracking_memory_max_samples,
            min_quality=settings.tracking_memory_min_quality,
        )

    def add(
        self,
        embedding: np.ndarray,
        *,
        quality: float,
        person_id: int | None,
        frame_index: int,
    ) -> None:
        if quality < self.min_quality:
            metrics.increment("identity_memory_rejected_low_quality")
            return
        vec = embedding.astype(np.float32).ravel()
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec = vec / norm
        self.samples.append(
            MemorySample(
                embedding=vec,
                quality=quality,
                person_id=person_id,
                frame_index=frame_index,
            )
        )
        while len(self.samples) > self.max_samples:
            self._drop_lowest_quality()
        metrics.observe("identity_memory_size", float(len(self.samples)))

    def best_representative(self) -> np.ndarray | None:
        if not self.samples:
            return None
        best = max(self.samples, key=lambda s: s.quality)
        return best.embedding.copy()

    def centroid(self) -> np.ndarray | None:
        if not self.samples:
            return None
        weights = np.array([s.quality for s in self.samples], dtype=np.float32)
        total = float(weights.sum())
        if total <= 0:
            return self.samples[-1].embedding.copy()
        stacked = np.stack([s.embedding for s in self.samples], axis=0)
        centroid = (stacked * weights[:, None]).sum(axis=0) / total
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-6:
            return self.samples[-1].embedding.copy()
        return (centroid / norm).astype(np.float32)

    def prune_outliers(self, reference: np.ndarray, max_cosine_distance: float) -> int:
        if not self.samples:
            return 0
        ref = reference.astype(np.float32).ravel()
        ref_norm = float(np.linalg.norm(ref))
        if ref_norm < 1e-6:
            return 0
        ref = ref / ref_norm
        removed = 0
        kept: deque[MemorySample] = deque(maxlen=self.max_samples)
        for sample in self.samples:
            sim = float(np.dot(ref, sample.embedding))
            if sim >= (1.0 - max_cosine_distance):
                kept.append(sample)
            else:
                removed += 1
        self.samples = kept
        if removed:
            metrics.increment("identity_memory_outliers_pruned", removed)
        return removed

    def _drop_lowest_quality(self) -> None:
        if not self.samples:
            return
        worst_idx = min(range(len(self.samples)), key=lambda i: self.samples[i].quality)
        del self.samples[worst_idx]

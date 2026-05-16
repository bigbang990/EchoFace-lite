"""Face matching and confidence — cosine similarity on normalized embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MatchResult:
    person_id: int
    confidence: float  # cosine similarity in [0,1] for L2-normalized vectors


class FaceMatcher:
    """Compare a query embedding against stored gallery vectors."""

    def best_match(
        self,
        query: np.ndarray,
        gallery: list[tuple[int, np.ndarray]],
        threshold: float,
    ) -> MatchResult | None:
        top = self.top_match(query, gallery)
        if top is None:
            return None
        logger.info("Top face similarity person_id=%s score=%.4f threshold=%.4f", top.person_id, top.confidence, threshold)
        if top.confidence < threshold:
            return None
        return top

    def top_match(
        self,
        query: np.ndarray,
        gallery: list[tuple[int, np.ndarray]],
    ) -> MatchResult | None:
        candidates = self.top_k(query, gallery, k=1)
        return candidates[0] if candidates else None

    def top_k(
        self,
        query: np.ndarray,
        gallery: list[tuple[int, np.ndarray]],
        *,
        k: int = 3,
    ) -> list[MatchResult]:
        if not gallery:
            return []
        q = query.astype(np.float32).ravel()
        candidates: list[MatchResult] = []
        for person_id, ref in gallery:
            r = ref.astype(np.float32).ravel()
            sim = float(np.dot(q, r))
            candidates.append(MatchResult(person_id=person_id, confidence=sim))
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates[: max(1, k)]

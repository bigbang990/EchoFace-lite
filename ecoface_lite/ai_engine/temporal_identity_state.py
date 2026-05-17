"""Per-track temporal identity accumulation (multi-hypothesis, evidence over time)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ecoface_lite.ai_engine.pose_estimator import PoseBucket
from ecoface_lite.core.metrics import metrics

UNKNOWN_ID = -1


@dataclass
class IdentityHypothesis:
    person_id: int
    confidence: float = 0.0
    evidence_count: int = 0
    last_frame: int = 0


@dataclass
class TemporalIdentityState:
    """Track-centric recognition state — never trust a single frame."""

    top_candidates: dict[int, IdentityHypothesis] = field(default_factory=dict)
    embedding_history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=16))
    confidence_history: deque[float] = field(default_factory=lambda: deque(maxlen=24))
    pose_history: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    blur_history: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    identity_locked: bool = False
    lock_person_id: int | None = None
    lock_until_frame: int = 0
    temporal_consistency: float = 0.0
    last_decay_frame: int = -1

    def observe_match(
        self,
        person_id: int | None,
        raw_confidence: float,
        frame_index: int,
        *,
        quality_weight: float = 1.0,
        decay: float = 0.94,
    ) -> None:
        self._decay_if_needed(frame_index, decay)
        pid = int(person_id) if person_id is not None and person_id >= 0 else UNKNOWN_ID
        weight = max(0.05, min(1.0, quality_weight)) * max(0.0, min(1.0, raw_confidence))
        hyp = self.top_candidates.get(pid)
        if hyp is None:
            hyp = IdentityHypothesis(person_id=pid, confidence=0.0)
            self.top_candidates[pid] = hyp
        hyp.confidence = (decay * hyp.confidence) + ((1.0 - decay) * weight)
        hyp.evidence_count += 1
        hyp.last_frame = frame_index
        self.confidence_history.append(raw_confidence)
        self._recompute_consistency()
        metrics.observe("temporal_identity_confidence", hyp.confidence)

    def record_embedding(self, embedding: np.ndarray, frame_index: int) -> None:
        vec = embedding.astype(np.float32).ravel()
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec = vec / norm
        self.embedding_history.append(vec)
        metrics.observe("track_embedding_history_size", float(len(self.embedding_history)))

    def record_pose_blur(self, pose_bucket: PoseBucket | str, blur_score: float) -> None:
        key = pose_bucket.value if isinstance(pose_bucket, PoseBucket) else str(pose_bucket)
        self.pose_history.append(key)
        self.blur_history.append(blur_score)
        metrics.increment(f"pose_bucket_{key}")

    def leading_identity(self, *, min_confidence: float = 0.0) -> tuple[int | None, float]:
        if not self.top_candidates:
            return None, 0.0
        ranked = sorted(self.top_candidates.values(), key=lambda h: h.confidence, reverse=True)
        best = ranked[0]
        if best.person_id == UNKNOWN_ID:
            if len(ranked) > 1 and ranked[1].confidence >= min_confidence:
                best = ranked[1]
            else:
                return None, best.confidence
        if best.confidence < min_confidence:
            return None, best.confidence
        return best.person_id, best.confidence

    def top_k(self, k: int = 3) -> list[tuple[int, float]]:
        ranked = sorted(self.top_candidates.values(), key=lambda h: h.confidence, reverse=True)
        out: list[tuple[int, float]] = []
        for hyp in ranked[:k]:
            pid = None if hyp.person_id == UNKNOWN_ID else hyp.person_id
            if pid is None:
                out.append((UNKNOWN_ID, hyp.confidence))
            else:
                out.append((pid, hyp.confidence))
        return out

    def agreement_ratio(self, person_id: int) -> float:
        if not self.confidence_history:
            return 0.0
        # Use recent match deque on track when available; fallback to hypothesis evidence
        hyp = self.top_candidates.get(person_id)
        if hyp is None or hyp.evidence_count == 0:
            return 0.0
        total_evidence = sum(h.evidence_count for h in self.top_candidates.values())
        return hyp.evidence_count / max(total_evidence, 1)

    def try_lock(
        self,
        person_id: int,
        frame_index: int,
        *,
        min_agreement_frames: int,
        min_consistency: float,
        lock_duration_frames: int,
    ) -> bool:
        if self.identity_locked and self.lock_person_id == person_id:
            return True
        hyp = self.top_candidates.get(person_id)
        if hyp is None:
            return False
        if hyp.evidence_count < min_agreement_frames:
            return False
        if self.temporal_consistency < min_consistency:
            return False
        self.identity_locked = True
        self.lock_person_id = person_id
        self.lock_until_frame = frame_index + lock_duration_frames
        metrics.increment("identity_temporal_locks")
        return True

    def is_lock_active(self, frame_index: int) -> bool:
        if not self.identity_locked:
            return False
        if frame_index > self.lock_until_frame:
            self.identity_locked = False
            self.lock_person_id = None
            metrics.increment("identity_temporal_lock_expired")
            return False
        return True

    def _decay_if_needed(self, frame_index: int, decay: float) -> None:
        if self.last_decay_frame == frame_index:
            return
        self.last_decay_frame = frame_index
        for hyp in self.top_candidates.values():
            hyp.confidence *= decay
        stale = [pid for pid, h in self.top_candidates.items() if h.confidence < 0.02 and h.evidence_count < 2]
        for pid in stale:
            if pid != UNKNOWN_ID:
                del self.top_candidates[pid]

    def _recompute_consistency(self) -> None:
        if len(self.confidence_history) < 2:
            self.temporal_consistency = 0.0
            return
        arr = np.array(self.confidence_history, dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std())
        self.temporal_consistency = max(0.0, min(1.0, mean * (1.0 - min(1.0, std * 2.0))))
        metrics.observe("temporal_consistency_score", self.temporal_consistency)


def get_temporal_identity(track) -> TemporalIdentityState:
    state = track.metadata.get("temporal_identity")
    if state is None:
        state = TemporalIdentityState()
        track.metadata["temporal_identity"] = state
    return state

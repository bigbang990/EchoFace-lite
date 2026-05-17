"""Multi-stage track-centric identity matching with soft temporal acceptance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecoface_lite.ai_engine.embedding_fusion import EmbeddingFusion
from ecoface_lite.ai_engine.global_identity_memory import GlobalIdentityMemory
from ecoface_lite.ai_engine.identity_memory_bank import IdentityMemoryBank
from ecoface_lite.ai_engine.matcher import FaceMatcher, MatchResult
from ecoface_lite.ai_engine.pose_estimator import PoseBucket
from ecoface_lite.ai_engine.temporal_identity_state import get_temporal_identity
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


@dataclass(frozen=True)
class IdentityMatchDecision:
    person_id: int | None
    confidence: float
    stage: str
    locked: bool = False
    shortlist_size: int = 0
    temporal_confidence: float = 0.0
    soft_match: bool = False


class MultiStageIdentityMatcher:
    """Track-centric matcher: accumulate evidence, delay hard locks."""

    def __init__(
        self,
        settings: Settings,
        matcher: FaceMatcher,
        fusion: EmbeddingFusion | None = None,
        global_memory: GlobalIdentityMemory | None = None,
    ) -> None:
        self._settings = settings
        self._matcher = matcher
        self._fusion = fusion or EmbeddingFusion(settings)
        self._global = global_memory
        self._shortlist_k = settings.tracking_match_shortlist_k
        self._lock_frames = settings.tracking_identity_lock_frames
        self._lock_margin = settings.tracking_identity_lock_margin
        self._temporal_lock_agreement = settings.tracking_temporal_lock_min_agreement

    def match_track(
        self,
        track: TrackedFace,
        query: np.ndarray,
        gallery: list[tuple[int, np.ndarray]],
        threshold: float,
        *,
        quality_weight: float = 1.0,
        frame_index: int = 0,
        pose_bucket: PoseBucket | str | None = None,
        blur_score: float | None = None,
    ) -> IdentityMatchDecision | None:
        temporal = get_temporal_identity(track)
        temporal.record_embedding(query, frame_index)

        memory: IdentityMemoryBank = track.metadata.setdefault(
            "identity_memory",
            IdentityMemoryBank.from_settings(self._settings),
        )

        fused = self._fusion.fuse(
            track,
            query,
            quality_weight=quality_weight,
            pose_bucket=pose_bucket,
            blur_score=blur_score,
        )
        memory.prune_outliers(fused, self._settings.tracking_embedding_outlier_cosine)
        memory.add(fused, quality=quality_weight, person_id=track.identity, frame_index=frame_index)

        query_emb = self._fusion.query_embedding(track, pose_bucket)
        match_query = query_emb if query_emb is not None else fused

        locked = self._locked_identity(track, match_query, gallery, threshold)
        if locked is not None:
            metrics.increment("identity_match_lock_hits")
            return locked

        shortlist = self._matcher.top_k(match_query, gallery, k=self._shortlist_k)
        if not shortlist and self._global is not None:
            profile_boost = self._global_shortlist(match_query, gallery, pose_bucket)
            shortlist = profile_boost
        if not shortlist:
            metrics.increment("identity_match_no_shortlist")
            temporal.observe_match(None, 0.0, frame_index, quality_weight=quality_weight * 0.5)
            return IdentityMatchDecision(
                person_id=None,
                confidence=0.0,
                stage="no_candidate",
                shortlist_size=0,
            )

        verified = self._temporal_verify(shortlist, memory, track, gallery, pose_bucket)
        if verified is None:
            metrics.increment("identity_match_temporal_rejected")
            return None

        temporal.observe_match(verified.person_id, verified.confidence, frame_index, quality_weight=quality_weight)
        leading_pid, hyp_conf = temporal.leading_identity()
        temporal_conf = max(verified.confidence, hyp_conf, temporal.temporal_consistency)

        soft_margin = self._settings.tracking_soft_match_margin
        effective_threshold = max(
            self._settings.tracking_min_soft_threshold,
            threshold - min(soft_margin, track.stable_match_count * 0.012),
        )
        is_soft = verified.confidence < threshold and temporal_conf >= effective_threshold - soft_margin
        if verified.confidence < effective_threshold and not is_soft:
            metrics.increment("identity_match_below_threshold")
            metrics.increment("identity_weak_candidates_retained")
            return IdentityMatchDecision(
                person_id=verified.person_id,
                confidence=verified.confidence,
                stage="weak_candidate",
                shortlist_size=len(shortlist),
                temporal_confidence=temporal_conf,
                soft_match=True,
            )

        if leading_pid is not None and leading_pid == verified.person_id:
            temporal.try_lock(
                leading_pid,
                frame_index,
                min_agreement_frames=self._temporal_lock_agreement,
                min_consistency=self._settings.tracking_temporal_lock_min_consistency,
                lock_duration_frames=self._lock_frames * 3,
            )
        self._apply_lock(track, verified, frame_index)
        if self._global is not None and verified.person_id is not None:
            bucket = pose_bucket.value if isinstance(pose_bucket, PoseBucket) else str(pose_bucket or "frontal")
            self._global.update_person(
                verified.person_id,
                fused,
                quality=quality_weight,
                frame_index=frame_index,
                pose_bucket=bucket,
            )

        metrics.increment("identity_match_accepted")
        if is_soft:
            metrics.increment("identity_soft_match_accepted")
        return IdentityMatchDecision(
            person_id=verified.person_id,
            confidence=verified.confidence,
            stage="soft_verified" if is_soft else "verified",
            locked=bool(track.metadata.get("identity_locked")),
            shortlist_size=len(shortlist),
            temporal_confidence=temporal_conf,
            soft_match=is_soft,
        )

    def _global_shortlist(
        self,
        query: np.ndarray,
        gallery: list[tuple[int, np.ndarray]],
        pose_bucket: PoseBucket | str | None,
    ) -> list[MatchResult]:
        if self._global is None:
            return []
        results: list[MatchResult] = []
        bucket = pose_bucket.value if isinstance(pose_bucket, PoseBucket) else str(pose_bucket or "")
        q = query.astype(np.float32).ravel()
        for person_id, ref in gallery:
            ref_vec = ref.astype(np.float32).ravel()
            prof = self._global.profile_embedding(person_id, bucket or None)
            if prof is not None:
                sim = float(np.dot(q, prof))
                if sim > 0.3:
                    results.append(MatchResult(person_id=person_id, confidence=sim))
                    continue
            sim = float(np.dot(q, ref_vec / max(np.linalg.norm(ref_vec), 1e-6)))
            if sim > 0.35:
                results.append(MatchResult(person_id=person_id, confidence=sim))
        results.sort(key=lambda m: m.confidence, reverse=True)
        return results[: self._shortlist_k]

    def _temporal_verify(
        self,
        shortlist: list[MatchResult],
        memory: IdentityMemoryBank,
        track: TrackedFace,
        gallery: list[tuple[int, np.ndarray]],
        pose_bucket: PoseBucket | str | None,
    ) -> MatchResult | None:
        centroid = memory.centroid()
        gallery_map = {pid: ref.astype(np.float32).ravel() for pid, ref in gallery}
        bucket = pose_bucket.value if isinstance(pose_bucket, PoseBucket) else str(pose_bucket or "")
        pose_vecs: dict = track.metadata.get("pose_embeddings", {})

        best: MatchResult | None = None
        best_score = -1.0
        for candidate in shortlist:
            gallery_sim = candidate.confidence
            ref = gallery_map.get(candidate.person_id)
            mem_sim = gallery_sim
            if centroid is not None and ref is not None:
                ref_norm = ref / max(float(np.linalg.norm(ref)), 1e-6)
                mem_sim = float(np.dot(centroid, ref_norm))
            pose_sim = gallery_sim
            if bucket and bucket in pose_vecs and ref is not None:
                pose_sim = float(np.dot(pose_vecs[bucket], ref / max(float(np.linalg.norm(ref)), 1e-6)))
            vote_bonus = 0.0
            if track.identity == candidate.person_id:
                vote_bonus = min(0.10, track.stable_match_count * 0.012)
            temporal = get_temporal_identity(track)
            hyp = temporal.top_candidates.get(candidate.person_id)
            hyp_bonus = min(0.08, hyp.confidence * 0.15) if hyp is not None else 0.0
            combined = (0.45 * gallery_sim) + (0.25 * mem_sim) + (0.15 * pose_sim) + vote_bonus + hyp_bonus
            if combined > best_score:
                best_score = combined
                best = MatchResult(person_id=candidate.person_id, confidence=combined)
        return best

    def _locked_identity(
        self,
        track: TrackedFace,
        fused: np.ndarray,
        gallery: list[tuple[int, np.ndarray]],
        threshold: float,
    ) -> IdentityMatchDecision | None:
        temporal = get_temporal_identity(track)
        locked_meta = track.metadata.get("identity_locked")
        locked_temporal = temporal.is_lock_active(track.last_seen_frame)
        if not locked_meta and not locked_temporal:
            return None
        if track.identity is None:
            return None
        lock_until = int(track.metadata.get("identity_lock_until_frame", temporal.lock_until_frame))
        if track.last_seen_frame > lock_until and not locked_temporal:
            track.metadata["identity_locked"] = False
            return None
        for person_id, ref in gallery:
            if person_id != track.identity:
                continue
            sim = float(np.dot(fused.ravel(), ref.astype(np.float32).ravel()))
            if sim + self._lock_margin >= threshold - self._settings.tracking_soft_match_margin:
                return IdentityMatchDecision(
                    person_id=person_id,
                    confidence=sim,
                    stage="locked",
                    locked=True,
                )
        track.metadata["identity_locked"] = False
        metrics.increment("identity_lock_breaks")
        return None

    def _apply_lock(self, track: TrackedFace, match: MatchResult, frame_index: int) -> None:
        if track.identity is not None and match.person_id != track.identity:
            return
        if track.stable_match_count + 1 < self._lock_frames:
            return
        track.metadata["identity_locked"] = True
        track.metadata["identity_lock_until_frame"] = frame_index + self._lock_frames * 3
        metrics.increment("identity_locks_applied")

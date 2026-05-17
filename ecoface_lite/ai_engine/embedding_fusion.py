"""Temporal embedding fusion with quality, pose, motion, and blur-aware weighting."""

from __future__ import annotations

import numpy as np

from ecoface_lite.ai_engine.pose_estimator import PoseBucket, pose_bucket_key
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


class EmbeddingFusion:
    """Fuses per-frame embeddings into stable track-level and pose-bucket vectors."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._alpha = settings.tracking_fused_embedding_alpha
        self._outlier_threshold = settings.tracking_embedding_outlier_cosine

    def fuse(
        self,
        track: TrackedFace,
        embedding: np.ndarray,
        *,
        quality_weight: float = 1.0,
        pose_bucket: PoseBucket | str | None = None,
        blur_score: float | None = None,
    ) -> np.ndarray:
        vec = self._normalize(embedding)
        weight = self._fusion_weight(track, quality_weight, blur_score=blur_score)
        if self._is_outlier(track, vec):
            metrics.increment("embedding_fusion_outliers_rejected")
            existing = track.metadata.get("fused_embedding")
            if existing is not None:
                return np.asarray(existing, dtype=np.float32)
            return vec

        effective_alpha = self._alpha * weight
        fused = track.metadata.get("fused_embedding")
        if fused is None:
            track.metadata["fused_embedding"] = vec
            metrics.increment("embedding_fusion_initialized")
        else:
            prev = self._normalize(np.asarray(fused, dtype=np.float32))
            merged = self._normalize(((1.0 - effective_alpha) * prev) + (effective_alpha * vec))
            track.metadata["fused_embedding"] = merged
            metrics.increment("embedding_fusion_updates")
            metrics.observe("embedding_fusion_weight", effective_alpha)
            vec = merged

        if pose_bucket is not None:
            self._update_pose_bucket(track, vec, pose_bucket, effective_alpha)
        return vec

    def query_embedding(
        self,
        track: TrackedFace,
        pose_bucket: PoseBucket | str | None = None,
    ) -> np.ndarray | None:
        if pose_bucket is not None:
            key = pose_bucket_key(pose_bucket) if isinstance(pose_bucket, PoseBucket) else str(pose_bucket)
            buckets: dict = track.metadata.setdefault("pose_embeddings", {})
            if key in buckets:
                return np.asarray(buckets[key], dtype=np.float32)
        fused = track.metadata.get("fused_embedding")
        if fused is not None:
            return np.asarray(fused, dtype=np.float32)
        if track.last_embedding is not None:
            return np.asarray(track.last_embedding, dtype=np.float32)
        return None

    def _fusion_weight(
        self,
        track: TrackedFace,
        quality_weight: float,
        *,
        blur_score: float | None,
    ) -> float:
        motion = float(track.metadata.get("motion_score", 1.0))
        stability = min(1.0, track.visibility_age / max(self._settings.tracking_stable_frames, 1))
        landmark = float(track.metadata.get("landmark_score", 0.5))
        consistency = float(track.metadata.get("embedding_consistency", 1.0))
        base = max(0.05, min(1.0, quality_weight))
        weight = base * (0.35 + 0.25 * motion + 0.20 * stability + 0.10 * landmark + 0.10 * consistency)
        if blur_score is not None:
            min_blur = self._settings.face_quality_min_blur_score
            blur_norm = min(1.0, blur_score / max(min_blur * 2.0, 1.0))
            floor = self._settings.tracking_blur_fusion_suppression
            if blur_score < min_blur * 0.5:
                weight *= floor
            else:
                weight *= max(floor, blur_norm)
        return max(self._settings.tracking_blur_fusion_suppression, min(1.0, weight))

    def _update_pose_bucket(
        self,
        track: TrackedFace,
        vec: np.ndarray,
        pose_bucket: PoseBucket | str,
        alpha: float,
    ) -> None:
        key = pose_bucket_key(pose_bucket) if isinstance(pose_bucket, PoseBucket) else str(pose_bucket)
        track.metadata["pose_bucket"] = key
        buckets: dict = track.metadata.setdefault("pose_embeddings", {})
        prev = buckets.get(key)
        if prev is None:
            buckets[key] = vec.copy()
        else:
            prev_n = self._normalize(np.asarray(prev, dtype=np.float32))
            buckets[key] = self._normalize(((1.0 - alpha) * prev_n) + (alpha * vec))

    def _is_outlier(self, track: TrackedFace, embedding: np.ndarray) -> bool:
        fused = track.metadata.get("fused_embedding")
        if fused is None:
            return False
        prev = self._normalize(np.asarray(fused, dtype=np.float32))
        sim = float(np.dot(prev, embedding))
        return sim < (1.0 - self._outlier_threshold)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        flat = vec.astype(np.float32).ravel()
        norm = float(np.linalg.norm(flat))
        if norm < 1e-6:
            return flat
        return (flat / norm).astype(np.float32)

"""Session-level identity memory for occlusion recovery and long-duration re-id."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ecoface_lite.ai_engine.pose_estimator import PoseBucket
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


@dataclass
class GlobalPersonProfile:
    person_id: int
    fused_embedding: np.ndarray | None = None
    last_seen_frame: int = 0
    confidence_peak: float = 0.0
    observation_count: int = 0
    pose_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    camera_id: str = "default"


@dataclass
class LostTrackSnapshot:
    track_id: str
    person_id: int | None
    fused_embedding: np.ndarray
    last_bbox: tuple[float, float, float, float]
    velocity: tuple[float, float]
    last_frame: int
    identity_confidence: float
    pose_bucket: str = PoseBucket.UNKNOWN.value
    camera_id: str = "default"


class GlobalIdentityMemory:
    """Compare new tracks against recently lost tracks and historical gallery profiles."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ttl = settings.tracking_global_memory_ttl_frames
        self._max_lost = settings.tracking_global_memory_max_lost
        self._profiles: dict[int, GlobalPersonProfile] = {}
        self._lost: deque[LostTrackSnapshot] = deque(maxlen=self._max_lost)

    def update_person(
        self,
        person_id: int,
        embedding: np.ndarray,
        *,
        quality: float,
        frame_index: int,
        pose_bucket: str = PoseBucket.FRONTAL.value,
    ) -> None:
        vec = _normalize(embedding)
        profile = self._profiles.get(person_id)
        if profile is None:
            profile = GlobalPersonProfile(person_id=person_id)
            self._profiles[person_id] = profile
        alpha = self._settings.tracking_fused_embedding_alpha * max(0.1, min(1.0, quality))
        if profile.fused_embedding is None:
            profile.fused_embedding = vec
        else:
            prev = profile.fused_embedding
            profile.fused_embedding = _normalize(((1.0 - alpha) * prev) + (alpha * vec))
        bucket_emb = profile.pose_embeddings.get(pose_bucket)
        if bucket_emb is None:
            profile.pose_embeddings[pose_bucket] = vec.copy()
        else:
            profile.pose_embeddings[pose_bucket] = _normalize(((1.0 - alpha) * bucket_emb) + (alpha * vec))
        profile.last_seen_frame = frame_index
        profile.observation_count += 1
        profile.confidence_peak = max(profile.confidence_peak, quality)
        metrics.observe("global_identity_profile_count", float(len(self._profiles)))

    def archive_lost_track(self, track: TrackedFace, frame_index: int) -> None:
        fused = track.metadata.get("fused_embedding")
        if fused is None and track.last_embedding is not None:
            fused = track.last_embedding
        if fused is None:
            return
        snapshot = LostTrackSnapshot(
            track_id=track.track_id,
            person_id=track.identity,
            fused_embedding=_normalize(np.asarray(fused, dtype=np.float32)),
            last_bbox=track.bbox,
            velocity=tuple(track.metadata.get("velocity", (0.0, 0.0))),
            last_frame=frame_index,
            identity_confidence=track.smoothed_confidence or track.identity_confidence,
            pose_bucket=str(track.metadata.get("pose_bucket", PoseBucket.UNKNOWN.value)),
            camera_id=str(track.metadata.get("camera_id", "default")),
        )
        self._lost.append(snapshot)
        metrics.increment("global_lost_tracks_archived")

    def prune(self, frame_index: int) -> None:
        while self._lost and frame_index - self._lost[0].last_frame > self._ttl:
            self._lost.popleft()
            metrics.increment("global_lost_tracks_expired")

    def match_lost_track(
        self,
        query: np.ndarray,
        frame_index: int,
        *,
        centroid: tuple[float, float] | None = None,
        min_similarity: float | None = None,
    ) -> LostTrackSnapshot | None:
        self.prune(frame_index)
        q = _normalize(query)
        min_sim = min_similarity if min_similarity is not None else self._settings.tracking_reid_min_similarity
        best: LostTrackSnapshot | None = None
        best_score = min_sim
        for snap in self._lost:
            sim = float(np.dot(q, snap.fused_embedding))
            score = sim
            if centroid is not None:
                cx = (snap.last_bbox[0] + snap.last_bbox[2]) / 2.0
                cy = (snap.last_bbox[1] + snap.last_bbox[3]) / 2.0
                dist = ((centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2) ** 0.5
                max_d = self._settings.temporal_max_track_distance
                proximity = max(0.0, 1.0 - (dist / max(max_d, 1.0)))
                frame_gap = frame_index - snap.last_frame
                recency = max(0.0, 1.0 - (frame_gap / max(self._ttl, 1)))
                score = (0.7 * sim) + (0.2 * proximity) + (0.1 * recency)
            if score > best_score:
                best_score = score
                best = snap
        if best is not None:
            metrics.increment("global_reid_success")
        return best

    def profile_embedding(self, person_id: int, pose_bucket: str | None = None) -> np.ndarray | None:
        profile = self._profiles.get(person_id)
        if profile is None:
            return None
        if pose_bucket and pose_bucket in profile.pose_embeddings:
            return profile.pose_embeddings[pose_bucket]
        return profile.fused_embedding


def _normalize(vec: np.ndarray) -> np.ndarray:
    flat = vec.astype(np.float32).ravel()
    norm = float(np.linalg.norm(flat))
    if norm < 1e-6:
        return flat
    return (flat / norm).astype(np.float32)

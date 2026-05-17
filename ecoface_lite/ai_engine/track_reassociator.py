"""Re-associate new tracks with recently lost identities."""

from __future__ import annotations

import numpy as np

from ecoface_lite.ai_engine.global_identity_memory import GlobalIdentityMemory, LostTrackSnapshot
from ecoface_lite.ai_engine.temporal_identity_state import get_temporal_identity
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


class TrackReassociator:
    def __init__(self, settings: Settings, global_memory: GlobalIdentityMemory) -> None:
        self._settings = settings
        self._global = global_memory

    def try_recover_identity(
        self,
        track: TrackedFace,
        query_embedding: np.ndarray,
        frame_index: int,
    ) -> int | None:
        if track.identity is not None and track.stable_match_count >= 2:
            return track.identity
        centroid = track.center_point
        snap = self._global.match_lost_track(
            query_embedding,
            frame_index,
            centroid=centroid,
        )
        if snap is None or snap.person_id is None or snap.person_id < 0:
            return None
        track.identity = snap.person_id
        track.smoothed_confidence = max(track.smoothed_confidence, snap.identity_confidence * 0.9)
        track.recovery_count += 1
        temporal = get_temporal_identity(track)
        temporal.observe_match(snap.person_id, snap.identity_confidence, frame_index, quality_weight=0.85)
        track.metadata["reid_from_track"] = snap.track_id
        metrics.increment("track_reassociation_success")
        metrics.observe("reid_latency_frames", float(frame_index - snap.last_frame))
        return snap.person_id

    def apply_lost_snapshot(self, track: TrackedFace, snap: LostTrackSnapshot, frame_index: int) -> None:
        track.identity = snap.person_id
        track.metadata["fused_embedding"] = snap.fused_embedding.copy()
        track.recovery_count += 1
        get_temporal_identity(track).observe_match(
            snap.person_id,
            snap.identity_confidence,
            frame_index,
            quality_weight=0.8,
        )

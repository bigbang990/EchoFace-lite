"""Temporal identity confidence — grows gradually, decays naturally."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecoface_lite.ai_engine.temporal_identity_state import TemporalIdentityState, get_temporal_identity
from ecoface_lite.ai_engine.track_quality_engine import TrackQualitySnapshot
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics


@dataclass(frozen=True)
class IdentityConfidenceSnapshot:
    temporal_confidence: float
    effective_threshold: float
    soft_accept: bool
    stable_enough: bool
    growth_rate: float


class IdentityConfidenceEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def evaluate(
        self,
        track: TrackedFace,
        raw_match_confidence: float,
        base_threshold: float,
        *,
        quality: TrackQualitySnapshot | None = None,
        person_id: int | None = None,
    ) -> IdentityConfidenceSnapshot:
        temporal = get_temporal_identity(track)
        track_quality = track.track_quality_score
        motion = float(track.metadata.get("motion_score", 1.0))
        persistence = min(1.0, track.visibility_age / max(self._settings.tracking_stable_frames, 1))
        vote_agreement = track.stable_match_count / max(len(track.recent_matches), 1)
        embedding_consistency = float(track.metadata.get("embedding_consistency", 1.0))
        reid_bonus = min(0.06, track.recovery_count * 0.02)

        temporal_conf = (
            (0.30 * temporal.temporal_consistency)
            + (0.25 * (track.smoothed_confidence or raw_match_confidence))
            + (0.15 * track_quality)
            + (0.10 * motion)
            + (0.10 * persistence)
            + (0.05 * vote_agreement)
            + (0.05 * embedding_consistency)
            + reid_bonus
        )
        temporal_conf = max(0.0, min(1.0, temporal_conf))

        margin = self._settings.tracking_soft_match_margin
        persistence_relief = min(margin, track.stable_match_count * 0.012)
        consistency_relief = temporal.temporal_consistency * self._settings.tracking_temporal_threshold_relief
        effective_threshold = max(
            self._settings.tracking_min_soft_threshold,
            base_threshold - persistence_relief - consistency_relief,
        )

        pid = person_id if person_id is not None else track.identity
        hyp_conf = 0.0
        if pid is not None and pid >= 0:
            hyp = temporal.top_candidates.get(int(pid))
            if hyp is not None:
                hyp_conf = hyp.confidence

        combined = max(temporal_conf, hyp_conf, raw_match_confidence)
        soft_accept = combined >= effective_threshold or (
            raw_match_confidence + persistence_relief >= base_threshold - margin
            and track.stable_match_count >= 1
        )
        stable_enough = (
            track.stable_match_count >= self._settings.temporal_min_confirmations
            and combined >= self._settings.temporal_min_average_confidence
        )

        growth = combined - (track.smoothed_confidence or 0.0)
        metrics.observe("identity_temporal_confidence", temporal_conf)
        metrics.observe("identity_effective_threshold", effective_threshold)
        metrics.observe("identity_confidence_growth", growth)

        return IdentityConfidenceSnapshot(
            temporal_confidence=combined,
            effective_threshold=effective_threshold,
            soft_accept=soft_accept,
            stable_enough=stable_enough,
            growth_rate=growth,
        )

    def embedding_suppression_weight(
        self,
        track: TrackedFace,
        *,
        blur_score: float,
        quality_weight: float,
    ) -> float:
        """Reduce fusion influence of blurry/unstable frames without breaking tracking."""
        min_blur = self._settings.face_quality_min_blur_score
        blur_norm = min(1.0, blur_score / max(min_blur * 2.0, 1.0))
        motion = float(track.metadata.get("motion_score", 1.0))
        floor = self._settings.tracking_blur_fusion_suppression
        weight = quality_weight * blur_norm * motion
        if blur_score < min_blur * 0.5:
            weight *= floor
        return max(floor, min(1.0, weight))

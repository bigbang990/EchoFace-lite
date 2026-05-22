"""Stability scoring logic for controlled validation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class StabilityScores:
    """Consolidated stability scores."""
    overall: float
    tracking: float
    identity: float
    detector: float
    grade: str


class StabilityScorer:
    """Calculates stability scores from experiment metrics."""

    def calculate_scores(self, metrics: Dict[str, Any]) -> StabilityScores:
        """Calculate stability scores (0-100) from metrics.
        
        Args:
            metrics: Dictionary containing raw metrics from the experiment.
            
        Returns:
            StabilityScores instance with component and overall scores.
        """
        # 1. Detector Stability (0-100)
        # Components: runtime stability, budget adherence, resolution safety
        det_runtime = metrics.get("detector_runtime_ms", 0.0)
        over_budget = metrics.get("detector_over_budget_count", 0)
        res_overflow = metrics.get("resolution_overflow_count", 0) # Assumed metric name
        
        # Penalize over budget runs (max 40 pts penalty)
        det_score = 100.0
        det_score -= min(40.0, over_budget * 2.0)
        # Penalize high runtime (max 30 pts penalty)
        if det_runtime > 120:
            det_score -= min(30.0, (det_runtime - 120) * 0.5)
        # Penalize resolution safety violations (max 30 pts penalty)
        det_score -= min(30.0, res_overflow * 10.0)
        det_score = max(0.0, det_score)

        # 2. Tracking Stability (0-100)
        # Components: track lifetime, jitter, recovery success
        avg_lifetime = metrics.get("avg_track_duration", 0.0)
        bbox_jitter = metrics.get("avg_bbox_delta_before", 0.0)
        recovery_rate = metrics.get("recovery_success_rate", 0.0)
        
        track_score = 0.0
        # Lifetime: 40 frames = 40 pts, capped at 40
        track_score += min(40.0, avg_lifetime) 
        # Jitter: lower is better (max 30 pts)
        jitter_penalty = min(30.0, bbox_jitter * 0.5)
        track_score += (30.0 - jitter_penalty)
        # Recovery: percentage based (max 30 pts)
        track_score += (recovery_rate * 30.0)
        
        # Normalize to 0-100 if we changed weights
        track_score = max(0.0, min(100.0, track_score))

        # 3. Identity Stability (0-100)
        # Components: switches, lock stability, confidence consistency
        id_switches = metrics.get("identity_switches", 0)
        temporal_conf = metrics.get("identity_temporal_confidence_avg", 0.0)
        
        id_score = 100.0
        # Penalize switches heavily (max 60 pts penalty)
        id_score -= min(60.0, id_switches * 20.0)
        # Reward temporal confidence (max 40 pts)
        id_score = (id_score * 0.6) + (temporal_conf * 40.0)
        id_score = max(0.0, min(100.0, id_score))

        # 4. Overall Stability
        # Weighted average emphasizing identity and tracking
        overall = (det_score * 0.2) + (track_score * 0.4) + (id_score * 0.4)
        
        # Grade Assignment
        if overall >= 90:
            grade = "production-grade"
        elif overall >= 75:
            grade = "operational but risky"
        elif overall >= 50:
            grade = "unstable"
        else:
            grade = "failure state"
            
        return StabilityScores(
            overall=round(overall, 2),
            tracking=round(track_score, 2),
            identity=round(id_score, 2),
            detector=round(det_score, 2),
            grade=grade
        )

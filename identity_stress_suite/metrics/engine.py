from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np

@dataclass
class ContinuityMetrics:
    id_switch_count: int = 0
    continuity_survival_duration: float = 0.0
    reidentification_success_rate: float = 0.0
    track_fragmentation_rate: float = 0.0
    confidence_stability_score: float = 0.0
    bbox_jitter_avg: float = 0.0
    occlusion_recovery_time: float = 0.0
    stable_matches_avg: float = 0.0
    
    # Internal tracking
    total_tracks: int = 0
    successful_reids: int = 0
    total_occlusion_events: int = 0
    total_recovery_time: float = 0.0
    fragmentation_count: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "id_switch_count": float(self.id_switch_count),
            "continuity_survival_duration": self.continuity_survival_duration,
            "reidentification_success_rate": self.reidentification_success_rate,
            "track_fragmentation_rate": self.track_fragmentation_rate,
            "confidence_stability_score": self.confidence_stability_score,
            "bbox_jitter_avg": self.bbox_jitter_avg,
            "occlusion_recovery_time": self.occlusion_recovery_time,
            "stable_matches_avg": self.stable_matches_avg
        }

class ContinuityMetricsEngine:
    """Engine for calculating continuity metrics from track history."""
    
    def __init__(self):
        self.track_history: Dict[str, List[Dict]] = {}
        self.ground_truth_ids: Dict[int, Set[str]] = {} # GT ID -> set of track IDs assigned to it
        
    def record_frame(self, frame_index: int, active_tracks: List[Dict]):
        """Record state of active tracks for a frame."""
        for track in active_tracks:
            tid = track["track_id"]
            if tid not in self.track_history:
                self.track_history[tid] = []
            
            entry = {
                "frame_index": frame_index,
                "bbox": track["bbox"],
                "identity": track.get("identity"),
                "confidence": track.get("confidence", 0.0),
                "state": track.get("state")
            }
            self.track_history[tid].append(entry)

    def calculate(self, ground_truth: Optional[Dict] = None) -> ContinuityMetrics:
        metrics = ContinuityMetrics()
        
        if not self.track_history:
            return metrics

        # 1. BBox Jitter
        jitter_values = []
        for tid, history in self.track_history.items():
            if len(history) < 2: continue
            for i in range(1, len(history)):
                b1 = history[i-1]["bbox"]
                b2 = history[i]["bbox"]
                # Mean absolute error of coordinates
                delta = sum(abs(v1 - v2) for v1, v2 in zip(b1, b2)) / 4.0
                jitter_values.append(delta)
        
        if jitter_values:
            metrics.bbox_jitter_avg = float(np.mean(jitter_values))

        # 2. Confidence Stability (1.0 - Std Dev of confidence)
        conf_stabilities = []
        for tid, history in self.track_history.items():
            confs = [h["confidence"] for h in history if h.get("confidence") is not None]
            if len(confs) > 5:
                conf_stabilities.append(1.0 - min(1.0, float(np.std(confs))))
        
        if conf_stabilities:
            metrics.confidence_stability_score = float(np.mean(conf_stabilities))

        # 3. ID Switches and Continuity
        # Simplified: count how many times a single track changes its 'identity' field
        id_switches = 0
        for tid, history in self.track_history.items():
            last_id = None
            for h in history:
                curr_id = h.get("identity")
                if curr_id is not None:
                    if last_id is not None and curr_id != last_id:
                        id_switches += 1
                    last_id = curr_id
        metrics.id_switch_count = id_switches

        # 4. Survival Duration
        durations = [len(h) for h in self.track_history.values()]
        if durations:
            metrics.continuity_survival_duration = float(np.mean(durations))

        # 5. Occlusion Recovery Time
        # Detect gaps in track history
        recovery_times = []
        for tid, history in self.track_history.items():
            if not history: continue
            last_frame = history[0]["frame_index"]
            for i in range(1, len(history)):
                curr_frame = history[i]["frame_index"]
                if curr_frame - last_frame > 1:
                    # Gap detected
                    recovery_times.append(curr_frame - last_frame)
                last_frame = curr_frame
        
        if recovery_times:
            metrics.occlusion_recovery_time = float(np.mean(recovery_times))
            
        # 6. Stable Matches
        stable_counts = []
        for tid, history in self.track_history.items():
            counts = [h.get("stable_match_count", 0) for h in history]
            if counts:
                stable_counts.append(max(counts))
        
        if stable_counts:
            metrics.stable_matches_avg = float(np.mean(stable_counts))

        return metrics

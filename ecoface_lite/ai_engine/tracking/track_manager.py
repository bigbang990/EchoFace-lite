"""SORT-style multi-face tracker — primary real-time perception layer.

Uses IoU + centroid association with linear bbox prediction on skipped detection
frames. Detections must be confirmed across multiple detector cycles before a track
is admitted (reduces single-frame hallucinations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import time
import numpy as np

from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace
from ecoface_lite.ai_engine.geometry import bbox_iou, compute_face_geometry
from ecoface_lite.ai_engine.motion_analyzer import MotionAnalyzer
from ecoface_lite.ai_engine.track_quality_engine import TrackQualityEngine, TrackQualitySnapshot
from ecoface_lite.ai_engine.tracking.track_state import ACTIVE_RECOGNITION_STATES, TrackLifecycleState
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.config.tracking import TrackingConfig, get_tracking_config
from ecoface_lite.core.logging import get_logger
from ecoface_lite.core.metrics import metrics
from ecoface_lite.core.platform_bootstrap import detect_platform

# Hardware-appropriate survival ceiling — evaluated once at import time so it
# is never re-computed inside the per-frame loop.
_PLATFORM = detect_platform()
_MAX_TRACK_SURVIVAL_MS = _PLATFORM["max_track_survival_ms"]

# Absolute hard-kill ceiling: no track survives longer than this, even if
# starvation prevention would otherwise preserve it indefinitely.
# CPU: 6000 * 3 = 18,000ms   GPU: 3000 * 3 = 9,000ms
_HARD_KILL_MULTIPLIER = 3
_HARD_KILL_MS = _MAX_TRACK_SURVIVAL_MS * _HARD_KILL_MULTIPLIER

if TYPE_CHECKING:
    from ecoface_lite.core.config import Settings

logger = get_logger(__name__)


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(1.0, (x2 - x1) * (y2 - y1))


@dataclass
class _PendingCandidate:
    face: DetectedFace
    hits: int = 1
    last_frame: int = 0
    centroid: tuple[float, float] = (0.0, 0.0)
    first_seen_ts: float = field(default_factory=time.monotonic)
    last_seen_ts: float = field(default_factory=time.monotonic)

    @property
    def duration_ms(self) -> float:
        return (self.last_seen_ts - self.first_seen_ts) * 1000.0


class FaceTrackManager:
    """Assigns persistent track ids and propagates faces between detector cycles."""

    def __init__(self, settings: Settings | None = None, config: TrackingConfig | None = None) -> None:
        from ecoface_lite.core.config import get_settings

        self._settings = settings or get_settings()
        self._cfg = config or get_tracking_config(self._settings)
        self._tracks: dict[str, TrackedFace] = {}
        self._pending: list[_PendingCandidate] = []
        self._removed_buffer: list[TrackedFace] = []
        self._next_id = 1
        self._motion = MotionAnalyzer(self._settings)
        self._quality_engine = TrackQualityEngine(self._settings)
        self._current_pressure_band = 0
        self.lockout_mode = False

    @property
    def active_track_count(self) -> int:
        return sum(
            1
            for t in self._tracks.values()
            if t.is_active and t.state not in {TrackLifecycleState.LOST.value, TrackLifecycleState.REMOVED.value}
        )

    @property
    def confirmed_track_count(self) -> int:
        """Count of tracks in CONFIRMED or STABLE state."""
        return sum(
            1
            for t in self._tracks.values()
            if t.state in {TrackLifecycleState.CONFIRMED.value, TrackLifecycleState.STABLE.value}
        )

    @property
    def has_coarse_tracks(self) -> bool:
        """True if any tracks are in COARSE state."""
        return any(t.state == TrackLifecycleState.COARSE.value for t in self._tracks.values())

    @property
    def candidate_queue_size(self) -> int:
        return len(self._pending)

    def active_tracks(self) -> list[TrackedFace]:
        return [t for t in self._tracks.values() if t.is_active and t.state != TrackLifecycleState.REMOVED.value]

    def consume_removed_tracks(self) -> list[TrackedFace]:
        """Tracks transitioned to REMOVED since last consume (for global re-id memory)."""
        out = self._removed_buffer
        self._removed_buffer = []
        return out

    def visible_tracks(self) -> list[TrackedFace]:
        """Tracks that should produce overlay / pipeline output this frame."""
        return [t for t in self._tracks.values() if t.state in ACTIVE_RECOGNITION_STATES and self._is_admitted(t)]

    def average_motion_stability(self) -> float:
        scores = [float(t.metadata.get("motion_score", 1.0)) for t in self.active_tracks()]
        if len(scores) == 0:
            return 0.0
        return sum(scores) / len(scores)

    def candidate_track(self, face: DetectedFace, frame_index: int) -> TrackedFace | None:
        self._expire_removed()
        centroid = _bbox_center((face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2))
        track = self._best_match(face, centroid, frame_index)
        if track is not None and self._is_admitted(track):
            return track
        return None

    def _calculate_candidate_priority(self, candidate: _PendingCandidate) -> float:
        """Score candidate for ingestion priority (Phase 2)."""
        face = candidate.face
        score = 0.0
        
        # 1. Face size (normalized area)
        area = (face.bbox.x2 - face.bbox.x1) * (face.bbox.y2 - face.bbox.y1)
        score += min(1.0, area / 20000.0) * 0.3
        
        # 2. Temporal consistency (hits)
        score += min(1.0, candidate.hits / 5.0) * 0.3
        
        # 3. Detector confidence
        score += face.det_score * 0.2
        
        # 4. Frontal pose bias (if available)
        # Using a simple heuristic: landmarks symmetry if available
        if face.landmarks is not None:
            score += 0.2
            
        return score

    def _calculate_track_priority(self, track: TrackedFace) -> int:
        """Calculate biometric priority (P0 to P3) for a track (Phase 4)."""
        # P0: Active alert target (mapped from metadata if set by external logic)
        if track.metadata.get("is_alert_target"):
            return 0
            
        # P1: Mature stable target
        if track.state == TrackLifecycleState.STABLE.value and track.visibility_age > 100:
            return 1
            
        # P2: Recently confirmed target
        if track.state in {TrackLifecycleState.CONFIRMED.value, TrackLifecycleState.STABLE.value}:
            # If it's confirmed but very small, it's P3 (background)
            # Area < 2500 (approx 50x50) is considered background
            if track.face_area < 2500:
                return 3
            return 2
            
        # P3: Background candidate / Coarse track
        return 3

    def _check_congestion(self) -> None:
        """Detect queue saturation and track churn storms."""
        # 1. Candidate queue pressure
        pending_count = len(self._pending)
        metrics.observe("candidate_queue_size", pending_count)
        
        # 2. Track churn rate (approximate by removed tracks in buffer)
        churn_rate = len(self._removed_buffer)
        metrics.observe("track_churn_rate", churn_rate)
        
        # 3. Pressure score (0.0 to 1.0)
        # Bounded by 20 candidates and 10 removals
        pressure = min(1.0, (pending_count / 20.0) + (churn_rate / 10.0))
        metrics.observe("state_machine_pressure_score", pressure)
        
        # 4. Pressure Band (Phase 3)
        band = 0
        if pressure > 0.8: band = 3 # CRITICAL
        elif pressure > 0.6: band = 2 # HIGH
        elif pressure > 0.3: band = 1 # ELEVATED
        
        self._current_pressure_band = band
        metrics.observe("tracking_pressure_band", float(band))
        
        # 5. Biometric Budgeting (Phase 4)
        biometric_score = pressure # Simple mapping for now
        if band > 0:
            metrics.observe("biometric_budget_pressure_score", biometric_score)
        
        if pressure > 0.8:
            logger.warning(
                "STATE MACHINE CONGESTION: pressure=%.2f band=%d candidates=%d churn=%d. ",
                pressure, band, pending_count, churn_rate
            )

    def update_from_detections(
        self,
        faces: list[DetectedFace],
        frame_index: int,
        *,
        frame_shape: tuple[int, ...] | None = None,
        frame_bgr: np.ndarray | None = None,
        detector_interval: int = 1,
    ) -> list[tuple[DetectedFace, TrackedFace | None]]:
        """Associate detector outputs to tracks (detection frame).
        
        Returns:
            List of (DetectedFace, TrackedFace | None) pairs in the same order as input faces.
        """
        self._expire_removed()
        self._check_congestion()
        
        matched_ids: set[str] = set()
        results: list[tuple[DetectedFace, TrackedFace | None]] = []

        # ── Phase 2: Priority-based ingestion pre-sort ───────────────────────
        if self._cfg.enable_priority_ingestion and len(faces) > 10:
            # We don't want to reorder faces as results must match input order.
            # Instead, we will process all, but admission logic might use priority.
            pass

        for face in faces:
            bbox = (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2)
            centroid = _bbox_center(bbox)
            track = self._best_match(face, centroid, frame_index, detector_interval=detector_interval)

            if track is None:
                track = self._admit_or_queue_pending(face, frame_index)
                if track is None:
                    # New track pending confirmation or rejected
                    results.append((face, None))
                    continue
                metrics.increment("new_tracks_created")
            elif track.state == TrackLifecycleState.LOST.value or track.state == TrackLifecycleState.COARSE.value:
                old_state = track.state
                track.recovery_count += 1
                metrics.increment("recovered_tracks")
                
                # Telemetry for soft recovery validation
                if track.lost_frames <= self._cfg.soft_recovery_frames:
                    metrics.increment("soft_track_recoveries")
                    metrics.increment("ghost_track_recoveries")
                    metrics.observe_rate("recovery_success_rate", 1.0, 1.0)
                
                if old_state == TrackLifecycleState.COARSE.value:
                    metrics.increment("coarse_tracks_promoted")
                    
                self._transition(track, TrackLifecycleState.CONFIRMED, frame_index, "recovered")
                # Give recovered tracks a protected window synchronized with cadence
                track.recovery_grace_frames = max(15, detector_interval * 2) 
            
            matched_ids.add(track.track_id)
            self._apply_detection(track, face, frame_index, frame_shape, frame_bgr)
            
            if self._is_admitted(track):
                results.append((face, track))
            else:
                results.append((face, None))

        self._decay_pending(frame_index)
        self._prune_low_quality_tracks(frame_index)

        # ── Phase 4: Track Survival Protection ───────────────────────────────
        pressure_band = self._current_pressure_band

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_ids:
                continue
            if track.state == TrackLifecycleState.REMOVED.value:
                continue
            
            # ── Step 3: Recovery Window Stabilization (Ghosting) ─────────────
            track.lost_frames += 1
            if track.lost_frames == 1:
                metrics.increment("ghost_tracks_created")
            
            time_lost = track.time_since_last_seen_ms
            metrics.observe("ghost_survival_duration_ms", time_lost)

            # Phase 4: Mature track survival
            is_mature = track.visibility_age > self._cfg.governance_mature_track_age
            
            # Phase 1: Candidate Immunity
            if track.visibility_age < self._cfg.governance_candidate_immunity_frames:
                track.governance_protected = True
            else:
                track.governance_protected = False

            survival_boost = 1.0
            if (self._cfg.enable_track_survival_protection and 
                (is_mature or track.governance_protected)):
                # Extend recovery buffer for mature or protected tracks during high pressure
                if pressure_band >= 2 or track.governance_protected:
                    survival_boost = 1.5
                    if is_mature:
                        metrics.increment("mature_track_survival_boosts")
                    if track.governance_protected:
                        metrics.increment("candidate_immunity_protections")
                    metrics.observe("protected_track_preservations", 1.0)
                else:
                    metrics.observe("protected_track_preservations", 0.0)

            # ── Adaptive Cadence Synchronization (Phase 2) ──────────────────
            # Scale survival horizon by detector interval to prevent immediate
            # re-degradation when cadence is stretched.
            cadence_multiplier = max(1.0, detector_interval / 4.0)
            
            if time_lost <= self._cfg.recovery_buffer_ms * survival_boost * cadence_multiplier:
                metrics.increment("transient_track_holds")
            
            self._quality_engine.decay_lost_track(track)
            if track.state != TrackLifecycleState.LOST.value:
                self._transition(track, TrackLifecycleState.LOST, frame_index, "unmatched")
            
            # Use time-based expiration for tracks
            survival_horizon = self._cfg.track_expiration_ms * survival_boost * cadence_multiplier
            survival_horizon = min(survival_horizon, _MAX_TRACK_SURVIVAL_MS)
            if track.state == TrackLifecycleState.COARSE.value:
                survival_horizon = self._cfg.coarse_track_survival_ms
                survival_horizon = min(survival_horizon, _MAX_TRACK_SURVIVAL_MS)
                
            # ── Phase 1: Track Survival Floor ─────────────────────────────────
            active_count = self.active_track_count
            
            # Protected recovery window check (prevent immediate re-degradation)
            is_in_recovery_grace = track.recovery_grace_frames > 0
            
            if time_lost > survival_horizon and not is_in_recovery_grace:
                # Phase 3: Downgrade instead of REMOVED if HIGH pressure and not already COARSE
                # Lower requirement for COARSE downgrade to ensure continuity
                can_downgrade_to_coarse = (self._cfg.enable_coarse_tracking and 
                    (pressure_band >= 1 or is_mature)
                    and track.state != TrackLifecycleState.COARSE.value
                    and track.visibility_age >= self._cfg.coarse_track_min_hits)
                
                if can_downgrade_to_coarse:
                    self._transition(track, TrackLifecycleState.COARSE, frame_index, "pressure_downgrade")
                    metrics.increment("coarse_tracks_created")
                    continue

                # Starvation Prevention: Never drop below MIN_SURVIVAL_TRACKS if detections exist
                if active_count < self._cfg.governance_min_survival_tracks or track.governance_protected:
                    if track.governance_protected:
                        metrics.increment("governance_forced_preservations")
                    metrics.increment("protected_track_preservations")
                    metrics.increment("starvation_prevented_events")
                    metrics.observe("governance_starvation_prevention_triggered", 1.0)
                    # Hard kill: starvation counters always fire, but a track that has
                    # been lost longer than _HARD_KILL_MS is removed regardless of
                    # population.  This prevents ghost immortality in single-person scenes.
                    if time_lost <= _HARD_KILL_MS:
                        continue  # Preserve this track — still within hard-kill window
                    metrics.increment("ghost_hard_kill_count")
                
                metrics.observe("governance_starvation_prevention_triggered", 0.0)
                metrics.increment("recovery_window_expired")
                metrics.increment("recovery_timeout_events")
                metrics.increment("ghost_recovery_failures") # Phase 4 addition
                
                self._transition(track, TrackLifecycleState.REMOVED, frame_index, "expired_time")
                metrics.increment("stale_track_replacements")
                if track.state == TrackLifecycleState.COARSE.value:
                    metrics.increment("coarse_tracks_expired")

        metrics.observe("active_track_count", self.active_track_count)
        # Filter out None tracks for metrics calculation
        valid_tracks = [t for _, t in results if t is not None]
        if valid_tracks:
            metrics.observe("avg_track_duration", sum(t.visibility_age for t in valid_tracks) / len(valid_tracks))
            metrics.observe("avg_track_survival_ms", sum(t.lifetime_ms for t in valid_tracks) / len(valid_tracks))
            metrics.observe("avg_track_quality", sum(t.track_quality_score for t in valid_tracks) / len(valid_tracks))
        return results

    def propagate(self, frame_index: int, detector_interval: int = 1) -> list[TrackedFace]:
        """Predict track positions on frames where the detector is skipped."""
        self._expire_removed()
        self._check_congestion()
        propagated: list[TrackedFace] = []
        for track in list(self._tracks.values()):
            if not track.is_active or track.state == TrackLifecycleState.REMOVED.value:
                continue
            if not self._is_admitted(track):
                continue
            velocity = track.metadata.get("velocity", (0.0, 0.0))
            x1, y1, x2, y2 = track.bbox
            
            dx, dy = velocity
            track.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            metrics.observe("recovery_prediction_distance", (dx*dx + dy*dy)**0.5)
            
            track.center_point = _bbox_center(track.bbox)
            track.face_area = _bbox_area(track.bbox)
            track.last_seen_frame = frame_index
            track.visibility_age = frame_index - track.first_seen_frame + 1
            motion = self._motion.update(track.track_id, track.bbox, frame_index)
            track.metadata["velocity"] = motion.velocity
            track.metadata["motion_score"] = motion.motion_stability_score
            
            # Phase 2: Partial continuity accumulation during skipped frames
            # Synchronize accumulation with detector interval (Step 4)
            if track.state in {TrackLifecycleState.NEW.value, TrackLifecycleState.CANDIDATE.value}:
                if motion.motion_stability_score > 0.85:
                    # Increment accumulation slightly based on cadence
                    accumulation = 1.0 + (detector_interval / 8.0)
                    track.confirmation_hits = min(track.confirmation_hits + accumulation, self._cfg.confirm_frames + 2)
            
            self._quality_engine.update(track, None, None, motion, frame_index)
            self._advance_lifecycle(track, frame_index)
            propagated.append(track)
        metrics.observe("active_tracks", self.active_track_count)
        metrics.observe("tracker_reuse_rate", 1.0)
        return propagated

    def should_compute_embedding(
        self,
        track: TrackedFace,
        frame_index: int,
        face: DetectedFace | None = None,
        *,
        quality: TrackQualitySnapshot | None = None,
    ) -> bool:
        if track.state not in {
            TrackLifecycleState.CONFIRMED.value,
            TrackLifecycleState.STABLE.value,
        }:
            metrics.increment("embedding_suppressed_unconfirmed")
            return False

        if quality is not None and not quality.recognition_allowed:
            metrics.increment("embedding_suppressed_low_quality")
            return False

        # Phase 2: Pressure-Aware Embedding Scheduling
        snapshot = metrics.snapshot()
        pressure_band = int(snapshot.recent_values.get("tracking_pressure_band", [0.0])[-1])
        priority = self._calculate_track_priority(track)
        
        # 2.2 Stable Identity Freeze (Phase 2.2)
        if self._cfg.governance_stable_identity_freeze_enabled and pressure_band >= 2:
            if track.state == TrackLifecycleState.STABLE.value and track.identity is not None:
                # If motion is stable, freeze embedding
                motion_score = float(track.metadata.get("motion_score", 1.0))
                if motion_score > 0.8:
                    metrics.increment("stable_identity_freezes")
                    return False

        cooldown_frames = self._cfg.embedding_cooldown_frames
        
        # Phase 2.1: Reduce refresh frequency by 50% under pressure
        if pressure_band == 1: # ELEVATED
            cooldown_frames *= 2
        elif pressure_band >= 2: # HIGH
            # Only refresh for new tracks or if no identity yet
            if track.identity is not None and priority >= 2:
                metrics.increment("embedding_generation_suppression_count")
                return False
            cooldown_frames *= 4

        # 2.3 Time-based cooldown (Phase 2.3)
        time_since_last_ms = (time.monotonic() - track.embedding_timestamp) * 1000.0
        if track.last_embedding is not None and time_since_last_ms < self._cfg.governance_embedding_refresh_cooldown_ms:
            metrics.increment("embedding_refresh_skips")
            return False

        if track.last_embedding is not None and frame_index - track.last_embedding_frame < cooldown_frames:
            if face is None:
                metrics.increment("embedding_skips")
                return False

        if track.last_embedding is None:
            return True
            
        # Priority-aware refresh interval
        refresh_interval = self._cfg.recognition_interval
        if priority >= 2 and pressure_band >= 1:
            refresh_interval *= 2
            
        if frame_index - track.last_embedding_frame >= refresh_interval:
            metrics.increment("embedding_cache_refresh_due")
            return True
        if face is not None:
            overlap = bbox_iou(face.bbox, BoundingBox(*track.bbox))
            if overlap < self._cfg.recognition_cache_min_iou:
                metrics.increment("embedding_cache_invalidations")
                return True
            prev_area = track.metadata.get("last_face_area", track.face_area)
            new_area = _bbox_area((face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2))
            if prev_area > 0 and abs(new_area - prev_area) / prev_area > self._cfg.bbox_area_change_ratio:
                metrics.increment("embedding_cache_invalidations")
                return True
            score = face.temporal_score if face.temporal_score is not None else face.det_score
            if score + self._cfg.confidence_drop_threshold < track.confidence:
                metrics.increment("embedding_cache_invalidations")
                return True
        metrics.increment("embedding_skips")
        return False

    def update_track_quality(
        self,
        track: TrackedFace,
        face: DetectedFace | None,
        frame_bgr: np.ndarray | None,
        frame_index: int,
    ) -> TrackQualitySnapshot:
        motion = self._motion.update(track.track_id, track.bbox, frame_index)
        track.metadata["velocity"] = motion.velocity
        return self._quality_engine.update(track, face, frame_bgr, motion, frame_index)

    def to_detected_face(self, track: TrackedFace) -> DetectedFace:
        x1, y1, x2, y2 = track.bbox
        landmarks = track.metadata.get("landmarks")
        return DetectedFace(
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            det_score=track.confidence,
            landmarks=landmarks,
        )

    def _admit_or_queue_pending(self, face: DetectedFace, frame_index: int) -> TrackedFace | None:
        bbox = face.bbox
        centroid = _bbox_center((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        now = time.monotonic()
        
        # Monitor queue pressure
        metrics.observe("confirmation_queue_pressure", len(self._pending))

        for pending in self._pending:
            iou_val = bbox_iou(bbox, pending.face.bbox)
            if iou_val >= self._settings.temporal_min_track_iou:
                pending.hits += 1
                pending.face = face
                pending.last_frame = frame_index
                pending.last_seen_ts = now
                pending.centroid = _bbox_center((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
                
                # ── Step 4: Adaptive Confirmation Logic (Time-Aware) ──────────
                duration = pending.duration_ms
                metrics.observe("confirmation_duration_ms", duration)

                if self._should_fast_confirm(pending, face):
                    self._pending.remove(pending)
                    track = self._spawn_track(face, frame_index)
                    track.confirmation_hits = pending.hits
                    self._transition(track, TrackLifecycleState.CANDIDATE, frame_index, "fast_confirmed")
                    metrics.increment("fast_confirmations")
                    metrics.observe("fast_confirm_duration_ms", duration)
                    return track

                if duration >= self._cfg.confirm_duration_ms:
                    self._pending.remove(pending)
                    track = self._spawn_track(face, frame_index)
                    track.confirmation_hits = pending.hits
                    self._transition(track, TrackLifecycleState.CANDIDATE, frame_index, "slow_confirmed")
                    metrics.increment("slow_confirmations")
                    metrics.observe("slow_confirm_duration_ms", duration)
                    return track
                
                metrics.increment("track_confirmation_pending")
                return None
        
        # Phase 2: Budget-aware candidate queueing
        if self._cfg.enable_priority_ingestion and len(self._pending) >= self._cfg.governance_max_candidate_queue_size:
            # Drop lowest priority candidate if new one is better
            candidate_scores = [(self._calculate_candidate_priority(p), i) for i, p in enumerate(self._pending)]
            new_candidate_score = self._calculate_candidate_priority(_PendingCandidate(face=face))
            
            lowest_score, lowest_idx = min(candidate_scores)
            if new_candidate_score > lowest_score:
                self._pending.pop(lowest_idx)
                metrics.increment("candidate_queue_drops")
            else:
                metrics.increment("candidate_ingestion_rejections")
                return None

        candidate = _PendingCandidate(
            face=face,
            hits=1,
            last_frame=frame_index,
            centroid=centroid,
            first_seen_ts=now,
            last_seen_ts=now
        )
        
        # Immediate fast-confirm if extremely high quality
        if face.det_score >= 0.95:
             track = self._spawn_track(face, frame_index)
             track.confirmation_hits = candidate.hits
             self._transition(track, TrackLifecycleState.CANDIDATE, frame_index, "instant_fast_confirm")
             metrics.increment("fast_confirmations")
             return track

        self._pending.append(candidate)
        metrics.increment("track_confirmation_pending")
        return None

    def _should_fast_confirm(self, pending: _PendingCandidate, current_face: DetectedFace) -> bool:
        """Heuristic for bypassing confirmation wait-time for high-confidence tracks."""
        # Need at least 2 hits for any confirmed track
        if pending.hits < 2:
            return False
            
        # Check det score stability
        if current_face.det_score < 0.92:
            return False
            
        # Check motion stability (approximate via bbox overlap)
        overlap = bbox_iou(pending.face.bbox, current_face.bbox)
        if overlap < 0.85: # High jitter
            metrics.increment("confirmation_resets") # Re-using as indicator of unstable pending
            return False
            
        return True

    def _decay_pending(self, frame_index: int) -> None:
        """Expire pending candidates with grace window and floor protection (Phase 1 & 2)."""
        now = time.monotonic()
        pressure_band = self._current_pressure_band
        
        ttl_ms = 500.0
        if pressure_band >= 2:
            ttl_ms = 300.0 # Aggressive decay under pressure
            
        initial_count = len(self._pending)
        if initial_count == 0:
            return

        # ── Phase 2: Candidate Grace Window ──────────────────────────────────
        # Candidates within grace window receive immunity from TTL culling
        def is_protected(p: _PendingCandidate) -> bool:
            if self.lockout_mode:
                return True # All protected during lockout
            age_frames = frame_index - p.last_frame # Approximate age since last seen
            # Or better, frame_index - p.first_frame if we added it. 
            # Let's use duration_ms since we have it.
            if p.duration_ms < self._cfg.governance_candidate_grace_frames * 33.3: # ~30fps
                metrics.increment("candidate_grace_preservations")
                return True
            return False

        # Sort by priority so if we must drop, we drop lowest priority
        self._pending.sort(key=lambda p: self._calculate_candidate_priority(p), reverse=True)
        
        retained = []
        for p in self._pending:
            time_since_seen = (now - p.last_seen_ts) * 1000.0
            if time_since_seen < ttl_ms or is_protected(p):
                retained.append(p)
            elif len(retained) < self._cfg.governance_min_survival_candidates:
                # ── Phase 1: Governance Floor Protection ──────────────────────
                retained.append(p)
                metrics.increment("governance_floor_activations")
                metrics.increment("starvation_prevented_events")
                metrics.observe("governance_starvation_prevention_triggered", 1.0)
        
        self._pending = retained
        
        drops = initial_count - len(self._pending)
        if drops > 0:
            metrics.increment("candidate_drop_rate", drops)
            metrics.observe("candidate_grace_window_active", 1.0)
        else:
            metrics.observe("candidate_grace_window_active", 0.0)

    def _spawn_track(self, face: DetectedFace, frame_index: int) -> TrackedFace:
        track_id = f"track_{self._next_id}"
        self._next_id += 1
        bbox = (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2)
        track = TrackedFace(
            track_id=track_id,
            bbox=bbox,
            confidence=face.det_score,
            last_seen_frame=frame_index,
            first_seen_frame=frame_index,
            visibility_age=1,
            lost_frames=0,
            center_point=_bbox_center(bbox),
            face_area=_bbox_area(bbox),
            state=TrackLifecycleState.NEW.value,
            confirmation_hits=1,
        )
        if face.landmarks is not None:
            track.metadata["landmarks"] = face.landmarks
        self._tracks[track_id] = track
        metrics.increment("track_state_new")
        return track

    def _apply_detection(
        self,
        track: TrackedFace,
        face: DetectedFace,
        frame_index: int,
        frame_shape: tuple[int, ...] | None,
        frame_bgr: np.ndarray | None,
    ) -> None:
        prev_center = track.center_point
        bbox = (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2)
        
        # ── Stability Hardening (Step 1): BBox Smoothing ─────────────────────
        track.update_bbox(bbox, self._cfg.bbox_ema_alpha)
        
        score = face.temporal_score if face.temporal_score is not None else face.det_score
        track.confidence = score
        track.center_point = _bbox_center(bbox)
        track.face_area = _bbox_area(bbox)
        track.metadata["last_face_area"] = track.face_area
        track.last_seen_frame = frame_index
        track.last_seen_ts = time.monotonic()
        track.visibility_age = frame_index - track.first_seen_frame + 1
        track.lost_frames = 0
        track.confirmation_hits = max(track.confirmation_hits, self._cfg.confirm_frames)
        if face.landmarks is not None:
            track.metadata["landmarks"] = face.landmarks
        if prev_center != (0.0, 0.0):
            track.metadata["velocity"] = (
                track.center_point[0] - prev_center[0],
                track.center_point[1] - prev_center[1],
            )
        motion = self._motion.update(track.track_id, track.bbox, frame_index)
        track.metadata["velocity"] = motion.velocity
        self._quality_engine.update(track, face, frame_bgr, motion, frame_index)
        self._advance_lifecycle(track, frame_index)
        metrics.observe("face_visibility_ratio", 1.0)

    def _advance_lifecycle(self, track: TrackedFace, frame_index: int) -> None:
        # Scale confirmation requirement slightly for extremely wide intervals
        # but keep it at least confirm_frames for safety.
        # This decouples promotion from raw detector skips (Step 5)
        promotion_threshold = self._cfg.confirm_frames
        
        if track.confirmation_hits < promotion_threshold:
            return
        
        lifetime = track.lifetime_ms
        
        if track.state == TrackLifecycleState.NEW.value:
            self._transition(track, TrackLifecycleState.CANDIDATE, frame_index, "confirmation_hits")
        
        if track.state in {TrackLifecycleState.NEW.value, TrackLifecycleState.CANDIDATE.value}:
            if lifetime >= self._cfg.confirm_duration_ms:
                self._transition(track, TrackLifecycleState.CONFIRMED, frame_index, "time_confirmed")
        
        # ── Step 4 & 5: State Machine & Temporal Audit ─────────────
        if (
            track.state == TrackLifecycleState.CONFIRMED.value
            and lifetime >= self._cfg.stable_duration_ms
            and track.track_quality_score >= self._cfg.min_recognition_quality
        ):
            self._transition(track, TrackLifecycleState.STABLE, frame_index, "stable_track_time")

        # Telemetry: Phase 3 Temporal Consistency Audit
        # drift = abs((actual_lifetime_ms / (frames * target_ms_per_frame)) - 1.0)
        # Simplified: check if visibility_age (frames) correlates with lifetime_ms at high/low FPS
        if track.visibility_age > 10:
            avg_ms_per_frame = lifetime / track.visibility_age
            # Assuming a standard 30 FPS (~33.3ms) or the detector interval
            expected_cadence = self._settings.detector_interval_frames * 33.3
            drift = abs(avg_ms_per_frame - expected_cadence) / max(1.0, expected_cadence)
            metrics.observe("fps_lifecycle_drift", drift)
            metrics.observe("lifecycle_time_consistency_score", max(0.0, 1.0 - drift))

    def _transition(
        self,
        track: TrackedFace,
        new_state: TrackLifecycleState,
        frame_index: int,
        reason: str,
    ) -> None:
        if track.state == new_state.value:
            metrics.increment("duplicate_transition_attempts")
            return
        
        # Phase 4: Invalid transition detection (simplified rules)
        # Example: can't go from REMOVED to anything else
        if track.state == TrackLifecycleState.REMOVED.value:
            metrics.increment("invalid_state_transitions")
            logger.warning("INVALID TRANSITION: track %s is REMOVED but attempted transition to %s", track.track_id, new_state.value)
            return

        prev = track.state
        if new_state == TrackLifecycleState.REMOVED and prev != TrackLifecycleState.REMOVED.value:
            self._removed_buffer.append(track)
        track.state = new_state.value
        metrics.increment(f"track_state_{new_state.value}")
        logger.debug(
            "track_state_transition id=%s %s->%s frame=%s reason=%s",
            track.track_id,
            prev,
            new_state.value,
            frame_index,
            reason,
        )

    def _prune_low_quality_tracks(self, frame_index: int) -> None:
        min_q = self._settings.tracking_min_quality_score
        for track_id, track in list(self._tracks.items()):
            if track.state == TrackLifecycleState.REMOVED.value:
                continue
            if track.visibility_age < self._cfg.confirm_frames:
                continue
            motion_score = float(track.metadata.get("motion_score", 1.0))
            
            # Phase 1 & 2: Governance Immunity from pruning
            if track.governance_protected or self.lockout_mode or track.recovery_grace_frames > 0:
                if track.recovery_grace_frames > 0:
                    track.recovery_grace_frames -= 1
                continue

            if track.track_quality_score < min_q and track.lost_frames == 0 and motion_score < self._cfg.min_motion_stability:
                self._transition(track, TrackLifecycleState.REMOVED, frame_index, "low_quality")
                metrics.increment("low_quality_track_kills")

    def _is_admitted(self, track: TrackedFace) -> bool:
        return track.confirmation_hits >= self._cfg.confirm_frames

    def _best_match(
        self,
        face: DetectedFace,
        centroid: tuple[float, float],
        frame_index: int,
        detector_interval: int = 1,
    ) -> TrackedFace | None:
        best: TrackedFace | None = None
        best_score = 0.0
        max_distance = self._settings.temporal_max_track_distance
        
        # Scale matching tolerance by detector interval (Step 5)
        cadence_relax = max(1.0, detector_interval / 2.0)
        max_distance *= cadence_relax
        
        if self.lockout_mode:
            max_distance *= 1.5 # Relax distance during recovery lockout
        for track in self._tracks.values():
            if not track.is_active or track.state == TrackLifecycleState.REMOVED.value:
                continue
            
            # Phase 3: Allow COARSE tracks to be recovered beyond normal max_lost_frames
            is_lost_too_long = (frame_index - track.last_seen_frame > self._cfg.max_lost_frames)
            
            if is_lost_too_long and track.state != TrackLifecycleState.COARSE.value:
                continue
            
            # Even for COARSE, we check time-based survival horizon
            if track.state == TrackLifecycleState.COARSE.value:
                if track.time_since_last_seen_ms > self._cfg.coarse_track_survival_ms:
                    continue
            dx = centroid[0] - track.center_point[0]
            dy = centroid[1] - track.center_point[1]
            distance = (dx * dx + dy * dy) ** 0.5
            distance_score = max(0.0, 1.0 - (distance / max(max_distance, 1.0)))
            iou = bbox_iou(face.bbox, BoundingBox(*track.bbox))
            score = (0.7 * iou) + (0.3 * distance_score)
            
            min_iou = self._settings.temporal_min_track_iou
            # Phase 3: Relax matching significantly during lockout or for COARSE recovery
            if self.lockout_mode or track.state == TrackLifecycleState.COARSE.value:
                min_iou = 0.01 
                max_distance *= 1.2

            if (
                iou >= min_iou or distance <= max_distance
            ) and score > best_score:
                best = track
                best_score = score
        return best

    def _expire_removed(self) -> None:
        expired = [tid for tid, t in self._tracks.items() if t.state == TrackLifecycleState.REMOVED.value]
        for tid in expired:
            del self._tracks[tid]
            self._motion.remove(tid)


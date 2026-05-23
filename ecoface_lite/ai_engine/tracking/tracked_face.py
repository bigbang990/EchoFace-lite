"""Per-track state carried through detection, tracking, and recognition."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ecoface_lite.ai_engine.tracking.track_state import ACTIVE_RECOGNITION_STATES, TrackLifecycleState


@dataclass
class TrackedFace:
    track_id: str

    bbox: tuple[float, float, float, float]
    confidence: float

    last_seen_frame: int
    first_seen_frame: int

    first_seen_ts: float = field(default_factory=time.monotonic)
    last_seen_ts: float = field(default_factory=time.monotonic)

    visibility_age: int
    lost_frames: int

    last_embedding: np.ndarray | None = None
    embedding_timestamp: float = 0.0
    last_embedding_frame: int = 0

    identity: int | None = None
    identity_confidence: float = 0.0

    smoothed_confidence: float = 0.0

    state: str = TrackLifecycleState.NEW.value

    center_point: tuple[float, float] = (0.0, 0.0)
    face_area: float = 0.0

    recognition_count: int = 0
    stable_match_count: int = 0

    recent_matches: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    recovery_count: int = 0
    identity_switch_count: int = 0

    confirmation_hits: int = 0
    track_quality_score: float = 0.0
    
    # ── Stability Hardening (Phase 3) ────────────────────────────────────────
    smoothed_bbox: tuple[float, float, float, float] | None = None

    metadata: dict = field(default_factory=dict)

    @property
    def numeric_track_id(self) -> int:
        """Stable integer id for APIs that still expect int track ids."""
        raw = self.track_id.replace("track_", "")
        try:
            return int(raw)
        except ValueError:
            return hash(self.track_id) % 1_000_000

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_RECOGNITION_STATES or self.state == TrackLifecycleState.LOST.value

    @property
    def is_stable(self) -> bool:
        return self.state == TrackLifecycleState.STABLE.value

    @property
    def lifetime_ms(self) -> float:
        return (time.monotonic() - self.first_seen_ts) * 1000.0

    @property
    def time_since_last_seen_ms(self) -> float:
        return (time.monotonic() - self.last_seen_ts) * 1000.0

    def touch_embedding(self, embedding: np.ndarray, frame_index: int) -> None:
        self.last_embedding = embedding
        self.last_embedding_frame = frame_index
        self.embedding_timestamp = time.monotonic()

    def record_identity_match(self, person_id: int, confidence: float, alpha: float) -> None:
        from ecoface_lite.ai_engine.temporal_identity_state import get_temporal_identity

        previous_identity = self.identity
        self.recent_matches.append(person_id)
        temporal = get_temporal_identity(self)
        leading, hyp_conf = temporal.leading_identity()
        voted = leading if leading is not None else self._majority_identity()
        if voted is not None:
            if previous_identity is not None and voted != previous_identity:
                self.identity_switch_count += 1
            self.identity = voted
        self.identity_confidence = max(confidence, hyp_conf)
        if self.smoothed_confidence <= 0:
            self.smoothed_confidence = confidence
        else:
            self.smoothed_confidence = (alpha * confidence) + ((1.0 - alpha) * self.smoothed_confidence)
        self.recognition_count += 1
        dominant = self.identity if self.identity is not None else self._majority_identity()
        if dominant is not None:
            self.stable_match_count = sum(1 for pid in self.recent_matches if pid == dominant)
        self.metadata["temporal_consistency"] = temporal.temporal_consistency

    def update_bbox(self, new_bbox: tuple[float, float, float, float], alpha: float) -> None:
        """Apply EMA smoothing to bounding box coordinates."""
        from ecoface_lite.core.metrics import metrics
        
        if self.smoothed_bbox is None:
            self.smoothed_bbox = new_bbox
            self.bbox = new_bbox
            return

        # Telemetry for stability analysis
        old_bbox = self.bbox
        delta = sum(abs(a - b) for a, b in zip(old_bbox, new_bbox)) / 4.0
        metrics.observe("avg_bbox_delta_before", delta)

        # Apply EMA: smoothed = alpha * new + (1 - alpha) * old
        smoothed = tuple(
            alpha * n + (1.0 - alpha) * o 
            for n, o in zip(new_bbox, self.smoothed_bbox)
        )
        self.smoothed_bbox = smoothed
        self.bbox = smoothed # Tracker uses smoothed coords
        
        new_delta = sum(abs(a - b) for a, b in zip(old_bbox, smoothed)) / 4.0
        metrics.observe("avg_bbox_delta_after", new_delta)
        metrics.increment("bbox_smoothing_applied_count")

    def _majority_identity(self) -> int | None:
        if not self.recent_matches:
            return None
        return max(set(self.recent_matches), key=list(self.recent_matches).count)

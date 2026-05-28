"""High-level pipeline orchestration (video frame → optional alert).

Tracking-first architecture:
  detect occasionally → track continuously → recognize intelligently
"""

from __future__ import annotations

import math
import time
import threading
from collections import deque
from dataclasses import dataclass
from queue import Queue, Empty

import numpy as np


@dataclass
class TimingBlock:
    """Unified timing block using time.perf_counter() exclusively."""
    start: float = 0.0
    end: float = 0.0

    def begin(self):
        self.start = time.perf_counter()

    def stop(self):
        self.end = time.perf_counter()

    @property
    def elapsed_ms(self):
        return (self.end - self.start) * 1000.0


@dataclass
class FrameTimingContext:
    """Isolated timing context per frame (no cross-frame contamination)."""
    frame_id: int
    frame_timer: TimingBlock
    preprocess_timer: TimingBlock
    detector_timer: TimingBlock
    tracker_timer: TimingBlock
    validator_timer: TimingBlock

    @property
    def preprocessing_ms(self) -> float:
        return self.preprocess_timer.elapsed_ms

    @property
    def detector_ms(self) -> float:
        return self.detector_timer.elapsed_ms

    @property
    def tracker_ms(self) -> float:
        return self.tracker_timer.elapsed_ms

    @property
    def validator_ms(self) -> float:
        return self.validator_timer.elapsed_ms

    @property
    def frame_total_ms(self) -> float:
        return self.frame_timer.elapsed_ms

    @property
    def subsystem_sum_ms(self) -> float:
        return self.preprocessing_ms + self.detector_ms + self.tracker_ms + self.validator_ms


@dataclass
class DetectionRequestPayload:
    """Payload sent to async detection worker."""
    frame_id: int
    timestamp: float
    prepared_frame_bgr: np.ndarray  # The preprocessed frame from prepare_for_detection
    scale: float


@dataclass
class DetectionResultPayload:
    """Payload returned from async detection worker."""
    frame_id: int
    timestamp: float
    raw_faces: list[DetectedFace]
    scale: float
    detector_inference_ms: float


from ecoface_lite.ai_engine.detection.detectors.multiscale_detector import MultiScaleDetector
from ecoface_lite.ai_engine.detection.detectors.base_detector import BaseDetector, DetectionConfig
from ecoface_lite.ai_engine.detection.fusion.weighted_box_fusion import WeightedBoxFusion, FusionConfig
from ecoface_lite.ai_engine.detection.temporal.weak_detection_memory import WeakDetectionMemory, MemoryConfig
from ecoface_lite.core.detection_metrics.detection_metrics import DetectionMetricsCollector
from ecoface_lite.ai_engine.confidence import ConfidencePolicy
from ecoface_lite.ai_engine.detection_optimizer import DetectionOptimizer
from ecoface_lite.ai_engine.diagnostics import diagnostics
from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace, FaceDetector
from ecoface_lite.ai_engine.embedder import FaceEmbedder
from ecoface_lite.ai_engine.event_validator import EventValidator
from ecoface_lite.ai_engine.face_candidate_validator import FaceCandidateValidator
from ecoface_lite.ai_engine.face_crop_validator import FaceCropValidator
from ecoface_lite.ai_engine.face_quality import FaceQualityAssessor
from ecoface_lite.ai_engine.temporal_detector import TemporalDetectorFilter
from ecoface_lite.ai_engine.geometry import bbox_iou, compute_face_geometry, scale_face_to_original
from ecoface_lite.ai_engine.embedding_fusion import EmbeddingFusion
from ecoface_lite.ai_engine.global_identity_memory import GlobalIdentityMemory
from ecoface_lite.ai_engine.identity_confidence_engine import IdentityConfidenceEngine
from ecoface_lite.ai_engine.identity_matcher import MultiStageIdentityMatcher
from ecoface_lite.ai_engine.matcher import FaceMatcher, MatchResult
from ecoface_lite.ai_engine.pose_estimator import classify_pose_bucket
from ecoface_lite.ai_engine.temporal_identity_state import get_temporal_identity
from ecoface_lite.ai_engine.track_reassociator import TrackReassociator
from ecoface_lite.ai_engine.pipeline_types import FaceDebugTrace, FrameMatch
from ecoface_lite.ai_engine.preprocessing import FramePreprocessor
from ecoface_lite.ai_engine.recognition_session import RecognitionSession
from ecoface_lite.ai_engine.tracking.track_state import ACTIVE_RECOGNITION_STATES, TrackLifecycleState
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.config.tracking import TrackingConfig, get_tracking_config
from ecoface_lite.core.config import Settings
from ecoface_lite.core.runtime_config import EffectiveRuntimeConfig
from ecoface_lite.core.logging import get_logger
from ecoface_lite.core.metrics import metrics
from ecoface_lite.core.validator import FaceValidator, ValidationTier, ValidationResult

logger = get_logger(__name__)


class LegacyDetectorWrapper(BaseDetector):
    """Minimal wrapper for legacy detectors that don't implement BaseDetector."""

    def __init__(self, detector: FaceDetector):
        self._detector = detector

    def detect(
        self,
        frame_bgr: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> list[DetectedFace]:
        # Legacy detector only takes frame_bgr.
        # MultiScaleDetector resizes the frame BEFORE calling detect.
        return self._detector.detect(frame_bgr)

    def get_model_name(self) -> str:
        return getattr(self._detector, "_model_name", "legacy_insightface")

    def get_input_size(self) -> tuple[int, int]:
        return getattr(self._detector, "_det_size", (640, 640))


class RecognitionPipeline:
    def __init__(
        self,
        settings: Settings,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        matcher: FaceMatcher,
        preprocessor: FramePreprocessor | None = None,
        quality_assessor: FaceQualityAssessor | None = None,
        confidence_policy: ConfidencePolicy | None = None,
        recognition_session: RecognitionSession | None = None,
        event_validator: EventValidator | None = None,
        effective_config: EffectiveRuntimeConfig | None = None,
    ) -> None:
        self._settings = settings
        self._effective_config = effective_config
        self._tracking_cfg: TrackingConfig = get_tracking_config(settings)
        
        # Log configuration integrity if available
        if self._effective_config:
            self._effective_config.log_integrity()
        self._detector = detector
        self._embedder = embedder
        self._matcher = matcher
        self._preprocessor = preprocessor or FramePreprocessor(settings)
        self._quality_assessor = quality_assessor or FaceQualityAssessor(settings)
        self._confidence_policy = confidence_policy or ConfidencePolicy(settings)
        self._recognition_session = recognition_session or RecognitionSession(settings)
        self._event_validator = event_validator or EventValidator(settings)
        self._detection_optimizer = DetectionOptimizer(settings)
        self._face_validator = FaceCandidateValidator(settings)
        self._unified_face_validator = FaceValidator(settings)
        self._weak_pass_attempts: dict[int, int] = {}
        self._temporal_detector = TemporalDetectorFilter(settings)
        self._crop_validator = FaceCropValidator(settings)
        self._embedding_fusion = EmbeddingFusion(settings)
        self._global_identity_memory = GlobalIdentityMemory(settings)
        self._identity_matcher = MultiStageIdentityMatcher(
            settings,
            matcher,
            self._embedding_fusion,
            self._global_identity_memory,
        )
        self._identity_confidence = IdentityConfidenceEngine(settings)
        self._track_reassociator = TrackReassociator(settings, self._global_identity_memory)

        # ── Phase 2A.1: Detection Observability Foundation ───────────────────
        self._detection_metrics: DetectionMetricsCollector | None = None
        if self._settings.detection_metrics_enabled:
            try:
                log_dir = self._settings.resolved_detection_metrics_log_dir()
                self._detection_metrics = DetectionMetricsCollector(
                    export_dir=log_dir,
                    export_interval=self._settings.detection_metrics_export_interval
                )
                logger.info("Detection observability initialized at %s", log_dir)
            except Exception as e:
                logger.warning("Failed to initialize detection observability: %s", e)

        # ── Phase 2A.4: Proposal Fusion Engine ───────────────────────────────
        fusion_cfg = FusionConfig(
            iou_threshold=self._settings.fusion_wbf_iou_threshold,
            crowd_iou_threshold=self._settings.fusion_crowd_iou_threshold,
            scale_weight_tiny=self._settings.fusion_scale_weight_tiny,
            scale_weight_small=self._settings.fusion_scale_weight_small,
            scale_weight_baseline=self._settings.fusion_scale_weight_baseline,
        )
        self._proposal_fusion = WeightedBoxFusion(fusion_cfg)

        # ── Phase 2A.5: Temporal Weak Detection Memory ───────────────────────
        memory_cfg = MemoryConfig(
            max_frames=self._settings.weak_memory_max_frames,
            cluster_iou=self._settings.weak_memory_cluster_iou,
            min_recurrence=self._settings.weak_memory_min_recurrence,
            promotion_boost=self._settings.weak_memory_promotion_boost,
        )
        self._weak_detection_memory = WeakDetectionMemory(memory_cfg)

        # ── Phase 2A.2: Multi-Scale Detection Wrapper ────────────────────────
        self._multiscale_detector: MultiScaleDetector | None = None
        if self._settings.enable_multiscale_detection:
            try:
                base = self._detector if isinstance(self._detector, BaseDetector) else LegacyDetectorWrapper(self._detector)
                self._multiscale_detector = MultiScaleDetector(base, self._settings)
                logger.info("Multi-scale detection wrapper initialized")
            except Exception as e:
                logger.warning("Failed to initialize multi-scale detector: %s", e)

        # ── Phase 2D.1: Timing History Buffers ────────────────────────────────
        self._preprocessing_history = deque(maxlen=60)
        self._detector_history = deque(maxlen=60)
        self._tracker_history = deque(maxlen=60)
        self._validator_history = deque(maxlen=60)
        self._frame_history = deque(maxlen=60)
        
        # ── Overload Management State ────────────────────────────────────────
        self._overload_active = False
        self._dynamic_detector_interval = settings.detector_interval_frames
        self._governance_cooldown_frames = 0
        self._detector_cooldown_active = False
        self._current_pressure_band = 0
        self._band_persistence_frames = 0
        self._adaptive_det_confidence = settings.detection_confidence_threshold
        self._adaptive_validator_cutoff = settings.validator_strict_cutoff
        
        # ── Phase 2 & 3 State Flags ──────────────────────────────────────────
        self._governance_lockout_active = False
        self._emergency_rebuild_active = False
        self._recovery_cooldown_frames = 0
        
        # ── Phase 3.0: Async Detection Infrastructure ─────────────────────────
        self._detection_request_queue: Queue[DetectionRequestPayload] = Queue(maxsize=2)
        self._detection_result_queue: Queue[DetectionResultPayload] = Queue(maxsize=2)
        self._shutdown_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._dropped_detection_requests: int = 0
        self._dropped_stale_results: int = 0
        
        # Start worker thread
        self._start_worker_thread()

    def _start_worker_thread(self) -> None:
        """Start the async detection worker thread."""
        self._worker_thread = threading.Thread(target=self._detection_worker, daemon=True, name="DetectionWorker")
        self._worker_thread.start()
        logger.info("Async detection worker thread started")

    def _detection_worker(self) -> None:
        """Background worker thread to perform detector inference."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for a detection request with a small timeout (so we can check shutdown event regularly)
                request = self._detection_request_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                # Perform inference (this is the slow part!)
                detection_timer = TimingBlock()
                detection_timer.begin()
                if self._multiscale_detector and self._settings.enable_multiscale_detection:
                    det_config = DetectionConfig(det_size=request.prepared_frame_bgr.shape[:2][::-1])
                    raw_faces = self._multiscale_detector.detect(request.prepared_frame_bgr, det_config)
                else:
                    raw_faces = self._detector.detect(request.prepared_frame_bgr)
                detection_timer.stop()

                # Create result payload
                result = DetectionResultPayload(
                    frame_id=request.frame_id,
                    timestamp=request.timestamp,
                    raw_faces=raw_faces,
                    scale=request.scale,
                    detector_inference_ms=detection_timer.elapsed_ms
                )

                # Try to put result into result queue (non-blocking, drop if full)
                try:
                    self._detection_result_queue.put_nowait(result)
                except Exception:
                    logger.warning("Detection result queue full, dropped result for frame %d", request.frame_id)
                    self._dropped_stale_results += 1
                    metrics.increment("phase3_dropped_stale_results")

            except Exception as e:
                logger.error("Error in detection worker thread: %s", e, exc_info=True)
            finally:
                # Mark request as done
                self._detection_request_queue.task_done()

    def shutdown(self) -> None:
        """Safely shut down the async detection worker thread."""
        logger.info("Shutting down async detection worker")
        self._shutdown_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)
        logger.info("Async detection worker shut down")

    @property
    def _track_manager(self):
        return self._recognition_session.track_manager

    def enroll_reference_embedding(self, frame_bgr: np.ndarray) -> np.ndarray:
        prepared = self._preprocessor.process(frame_bgr)
        faces = self._detector.detect(prepared.bgr)
        if not faces:
            raise ValueError("No face detected for enrollment")
        best = max(faces, key=lambda f: f.det_score)
        quality = self._quality_assessor.assess(prepared.bgr, best)
        if not quality.accepted:
            raise ValueError(f"Face quality rejected for enrollment: {quality.reason}")
        return self._embedder.embed_face(prepared.bgr, best)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        gallery: list[tuple[int, np.ndarray]],
    ) -> list[FrameMatch]:
        metrics.increment("total_frames_processed")
        
        # ── Phase 3.0: Process Pending Detection Results First ─────────────────
        self._process_pending_detection_results(frame_index, gallery)
        
        # ── Phase 2D.1: Initialize Per-Frame Timing Context ────────────────────
        timing_ctx = FrameTimingContext(
            frame_id=frame_index,
            frame_timer=TimingBlock(),
            preprocess_timer=TimingBlock(),
            detector_timer=TimingBlock(),
            tracker_timer=TimingBlock(),
            validator_timer=TimingBlock()
        )
        
        timing_ctx.frame_timer.begin()
        
        # ── Phase 2: Governance Lockout Mode Activation ──────────────────────
        active_tracks = self._track_manager.active_track_count
        # We need to peek at detections count to decide on lockout
        # This is handled inside _apply_load_governance but we need the flag here
        
        matches = self._process_frame_staged(frame_bgr, frame_index, gallery, timing_ctx)
        
        timing_ctx.frame_timer.stop()
        
        # ── Phase 2D.1: Timing Integrity Validation & Metrics ─────────────────
        self._process_frame_timing(timing_ctx)
        
        # Update metrics and flags
        metrics.observe("governance_lockout_active", 1.0 if self._governance_lockout_active else 0.0)
        metrics.observe("emergency_rebuild_active", 1.0 if self._emergency_rebuild_active else 0.0)
            
        # Track average processing FPS for observability (uses current frame total time ONLY)
        if timing_ctx.frame_total_ms > 0:
            fps = 1000.0 / timing_ctx.frame_total_ms
            metrics.observe("average_processing_fps", fps)
            
            # ── Phase 6: Parity Comparison Readiness ──────────────────────
            # Observe current hardware execution context
            metrics.observe("hardware_backend_type", 1.0 if self._settings.insightface_ctx_id >= 0 else 0.0) # 1 for GPU, 0 for CPU
            
            # ── Step 5 & 6: Integrity & Regression Checks ────────────────
            self._check_telemetry_integrity_phase2d(timing_ctx, fps)
            self._check_regressions(fps)
        
        return matches
        
    def _process_frame_timing(self, timing_ctx: FrameTimingContext):
        """Process per-frame timing: validate, update history, record metrics."""
        # Validate timing integrity
        timing_valid = True
        timing_violation = False
        epsilon_ms = 2.0
        
        # Phase 2D.2: Acquisition Cost Hierarchy Validation
        acquisition_valid = True
        acquisition_subsystem_sum = (
            self._detection_optimizer.roi_crop_ms +
            self._detection_optimizer.roi_upscale_ms +
            self._detection_optimizer.clahe_ms +
            self._detection_optimizer.detector_inference_ms +
            timing_ctx.validator_ms +
            self._detection_optimizer.small_face_rescue_ms +
            self._detection_optimizer.resolution_escalation_ms
        )
        
        if acquisition_subsystem_sum > self._detection_optimizer.acquisition_total_ms + epsilon_ms:
            acquisition_valid = False
            timing_violation = True
            metrics.increment("phase2d2_acquisition_hierarchy_violation")
            logger.warning(
                "PHASE 2D.2 ACQUISITION INTEGRITY VIOLATION: acquisition_subsystem_sum_ms (%.2f) > acquisition_total_ms + ε (%.2f)",
                acquisition_subsystem_sum, self._detection_optimizer.acquisition_total_ms + epsilon_ms
            )
        
        # Check for negative timings
        if (timing_ctx.preprocessing_ms < 0 or
            timing_ctx.detector_ms < 0 or
            timing_ctx.tracker_ms < 0 or
            timing_ctx.validator_ms < 0 or
            timing_ctx.frame_total_ms < 0):
            timing_valid = False
            timing_violation = True
            metrics.increment("timing_negative_value")
            
        # Check for impossible hierarchy: detector_ms > frame_total_ms
        if timing_ctx.detector_ms > timing_ctx.frame_total_ms:
            timing_valid = False
            timing_violation = True
            metrics.increment("timing_detector_exceeds_frame_total")
            logger.warning(
                "TIMING INTEGRITY VIOLATION: detector_ms (%.2f) > frame_total_ms (%.2f)",
                timing_ctx.detector_ms, timing_ctx.frame_total_ms
            )
            
        # Check for subsystem sum exceeding frame total
        subsystem_sum = timing_ctx.subsystem_sum_ms
        if subsystem_sum > timing_ctx.frame_total_ms + epsilon_ms:
            timing_valid = False
            timing_violation = True
            metrics.increment("timing_subsystem_sum_exceeds_frame")
            logger.warning(
                "TIMING INTEGRITY VIOLATION: subsystem_sum_ms (%.2f) > frame_total_ms + ε (%.2f)",
                subsystem_sum, timing_ctx.frame_total_ms + epsilon_ms
            )
            
        # Check for NaN/inf values
        if (math.isnan(timing_ctx.preprocessing_ms) or math.isinf(timing_ctx.preprocessing_ms) or
            math.isnan(timing_ctx.detector_ms) or math.isinf(timing_ctx.detector_ms) or
            math.isnan(timing_ctx.tracker_ms) or math.isinf(timing_ctx.tracker_ms) or
            math.isnan(timing_ctx.validator_ms) or math.isinf(timing_ctx.validator_ms) or
            math.isnan(timing_ctx.frame_total_ms) or math.isinf(timing_ctx.frame_total_ms)):
            timing_valid = False
            timing_violation = True
            metrics.increment("timing_nan_or_inf")
            
        # Update rolling history buffers
        self._preprocessing_history.append(timing_ctx.preprocessing_ms)
        self._detector_history.append(timing_ctx.detector_ms)
        self._tracker_history.append(timing_ctx.tracker_ms)
        self._validator_history.append(timing_ctx.validator_ms)
        self._frame_history.append(timing_ctx.frame_total_ms)
        
        # Calculate rolling averages
        rolling_preprocessing_avg = sum(self._preprocessing_history) / len(self._preprocessing_history)
        rolling_detector_avg = sum(self._detector_history) / len(self._detector_history)
        rolling_tracker_avg = sum(self._tracker_history) / len(self._tracker_history)
        rolling_validator_avg = sum(self._validator_history) / len(self._validator_history)
        rolling_frame_avg = sum(self._frame_history) / len(self._frame_history)
        
        # Record both raw and rolling metrics separately
        # Layer A: Raw per-frame metrics
        metrics.observe("current_frame_preprocessing_ms", timing_ctx.preprocessing_ms)
        metrics.observe("current_frame_detector_ms", timing_ctx.detector_ms)
        metrics.observe("current_frame_tracker_ms", timing_ctx.tracker_ms)
        metrics.observe("current_frame_validator_ms", timing_ctx.validator_ms)
        metrics.observe("current_frame_total_ms", timing_ctx.frame_total_ms)
        
        # Layer B: Rolling aggregates
        metrics.observe("rolling_preprocessing_avg_ms", rolling_preprocessing_avg)
        metrics.observe("rolling_detector_avg_ms", rolling_detector_avg)
        metrics.observe("rolling_tracker_avg_ms", rolling_tracker_avg)
        metrics.observe("rolling_validator_avg_ms", rolling_validator_avg)
        metrics.observe("rolling_frame_avg_ms", rolling_frame_avg)
        
        # Also record integrity metrics
        metrics.observe("timing_valid", 1.0 if timing_valid else 0.0)
        metrics.observe("timing_violation", 1.0 if timing_violation else 0.0)

    def _process_pending_detection_results(self, current_frame_id: int, gallery: list[tuple[int, np.ndarray]]) -> None:
        """Process any pending detection results from the worker thread, with frame age validation."""
        while True:
            try:
                # Non-blocking get to check for results
                result = self._detection_result_queue.get_nowait()
            except Empty:
                break

            # Frame age validation
            frame_age = current_frame_id - result.frame_id
            if frame_age > 30:
                logger.warning("Dropping stale detection result for frame %d (age: %d frames)", result.frame_id, frame_age)
                self._dropped_stale_results += 1
                metrics.increment("phase3_dropped_stale_results")
                continue

            # Record async telemetry
            metrics.observe("phase3_frame_age_at_reconciliation", frame_age)
            metrics.observe("phase3_detector_inference_ms", result.detector_inference_ms)
            self._detection_optimizer.detector_inference_ms = result.detector_inference_ms
            self._detection_optimizer.acquisition_total_ms = result.detector_inference_ms

            # Process the result (this simulates the rest of _stage_detection_path)
            # First, scale the faces back
            raw_faces = self._detection_optimizer.scale_faces(result.raw_faces, result.scale)
            # Apply temporal detector filter
            raw_faces = self._temporal_detector.apply(raw_faces, result.frame_id)

            # Apply proposal fusion (if enabled)
            if self._proposal_fusion and len(raw_faces) > 1:
                # We don't have the actual prepared frame shape here, but let's use a dummy (or skip for now?)
                # Wait, actually we could store more in the payload, but for now let's just skip fusion for async?
                # Or, maybe not - let's just process raw_faces as is
                pass

            # Now update track manager with these detections!
            # Track manager's actual method is update_from_detections!
            # We pass frame_shape as None since we don't have the original frame right now (we only stored prepared frame)
            face_track_results = self._track_manager.update_from_detections(
                raw_faces, 
                result.frame_id,
                detector_interval=self._dynamic_detector_interval
            )

            # Now update optimizer stats!
            self._detection_optimizer._last_faces = raw_faces  # Set last faces so consecutive_weak_detection check works
            self._detection_optimizer.observe_detection_cycle(
                frame_index=result.frame_id, 
                raw_count=len(raw_faces), 
                accepted_count=len(raw_faces), 
                rejected_count=0
            )

    def _check_telemetry_integrity_phase2d(self, timing_ctx: FrameTimingContext, current_fps: float):
        """Phase 2D.1: Check telemetry integrity using current frame metrics only (NOT rolling averages)."""
        # Use ONLY current frame timings (layer A), never rolling averages for integrity validation!
        det_raw_ms = timing_ctx.detector_ms
        cadence = self._settings.detector_interval_frames
        if self._overload_active:
            cadence = self._dynamic_detector_interval
            
        # Per-cycle runtime is the raw measurement from current frame
        metrics.observe("detector_runtime_per_cycle_ms", det_raw_ms)
        
        # Effective frame cost = per-cycle time spread across cadence
        effective_cost = det_raw_ms / max(1, cadence)
        metrics.observe("detector_effective_frame_cost_ms", effective_cost)
        
        # Check for impossible timing (synchronous execution check):
        # total frame time must be >= effective detector cost
        total_ms = timing_ctx.frame_total_ms
        if effective_cost > total_ms * 1.1:  # 10% margin for timing jitter
            msg = f"Timing Inconsistency: detector_cost({effective_cost:.1f}ms) > total_frame({total_ms:.1f}ms)"
            logger.warning("TELEMETRY INTEGRITY WARNING: %s", msg)
            diagnostics.record("telemetry_integrity", msg)
            metrics.increment("telemetry_integrity_warnings")

    def _check_telemetry_integrity(self, current_fps: float) -> None:
        """Audit timing metrics for mathematical coherence."""
        snapshot = metrics.snapshot()
        
        # ── Step 5: Telemetry Integrity Audit ────────────────────────────────
        det_raw_ms = snapshot.recent_values.get("face_detection_duration", [0.0])[-1] * 1000.0
        cadence = self._settings.detector_interval_frames
        if self._overload_active:
            cadence = self._dynamic_detector_interval
            
        # Per-cycle runtime is the raw measurement
        metrics.observe("detector_runtime_per_cycle_ms", det_raw_ms)
        
        # Effective frame cost = per-cycle time spread across cadence
        effective_cost = det_raw_ms / max(1, cadence)
        metrics.observe("detector_effective_frame_cost_ms", effective_cost)
        
        # Check for impossible timing (synchronous execution check)
        # Total frame time must be >= effective detector cost
        total_ms = (1.0 / max(0.1, current_fps)) * 1000.0
        if effective_cost > total_ms * 1.1: # 10% margin for timing jitter
            msg = f"Timing Inconsistency: detector_cost({effective_cost:.1f}ms) > total_frame({total_ms:.1f}ms)"
            logger.warning("TELEMETRY INTEGRITY WARNING: %s", msg)
            diagnostics.record("telemetry_integrity", msg)
            metrics.increment("telemetry_integrity_warnings")

    def _check_regressions(self, current_fps: float) -> None:
        """Log warnings if performance or quality regresses below baselines."""
        snapshot = metrics.snapshot()
        
        # ── Step 4 & 6: Regression Guardrails ────────────────────────────────
        
        # 1. FPS Check
        if current_fps < 15.0:
            msg = f"FPS < 15 (Current: {current_fps:.2f})"
            logger.warning(
                "REGRESSION WARNING: %s. "
                "Probable Root Cause: System resource exhaustion or high-resolution input. "
                "Suggested Action: Reduce input resolution or check CPU/GPU load.",
                msg
            )
            diagnostics.record("regression", msg)
            
        # 2. Detector Runtime Check
        avg_det_runtime = snapshot.averages.get("detector_runtime_per_cycle_ms", 0.0)
        if avg_det_runtime > 120.0:
            msg = f"detector_runtime_ms > 120ms (Avg: {avg_det_runtime:.2f}ms)"
            logger.warning(
                "REGRESSION WARNING: %s. "
                "Probable Root Cause: Detector tensor inflation. "
                "Suggested Action: Lower DETECTOR_MAX_INPUT_PIXELS.",
                msg
            )
            diagnostics.record("regression", msg)

        # 3. Track Lifetime Check
        avg_lifetime = snapshot.averages.get("avg_track_duration", 0.0)
        if avg_lifetime > 0 and avg_lifetime < 10.0:
            msg = f"avg_track_lifetime < 10 (Avg: {avg_lifetime:.2f})"
            logger.warning(
                "REGRESSION WARNING: %s. "
                "Probable Root Cause: Tracking instability or coordinate teleportation. "
                "Suggested Action: Check BBox smoothing parameters (TRACKING_BBOX_EMA_ALPHA).",
                msg
            )
            diagnostics.record("regression", msg)

        # 4. Identity Switch Check
        if snapshot.counters.get("identity_switches", 0) > 0:
            msg = "identity_switch_rate > 0"
            logger.warning(
                "REGRESSION WARNING: %s. "
                "Probable Root Cause: Identity memory instability or low embedding quality. "
                "Suggested Action: Check recognition thresholds or embedding cooldown.",
                msg
            )
            diagnostics.record("regression", msg)
            
        # 5. Recovery Failure Check
        recovered = snapshot.counters.get("recovered_tracks", 0)
        total_detections = snapshot.counters.get("total_faces_detected", 0)
        if total_detections > 50 and recovered == 0:
            msg = "Zero track recoveries despite active detections"
            logger.warning(
                "REGRESSION WARNING: %s. "
                "Probable Root Cause: Ghosting window too small or association logic too strict. "
                "Suggested Action: Increase TRACKING_SOFT_RECOVERY_FRAMES or TEMPORAL_MAX_TRACK_DISTANCE.",
                msg
            )
            diagnostics.record("regression", msg)

        # 6. Confirmation Pending Saturation
        pending = snapshot.counters.get("track_confirmation_pending", 0)
        if pending > 20:
            msg = f"Confirmation Pending Saturation ({pending} tracks)"
            logger.warning(
                "REGRESSION WARNING: %s. "
                "Probable Root Cause: Adaptive confirmation criteria not being met. "
                "Suggested Action: Lower TRACKING_FAST_CONFIRM_MIN_CONSISTENCY or check image quality.",
                msg
            )
            diagnostics.record("regression", msg)

    def _process_frame_staged(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        gallery: list[tuple[int, np.ndarray]],
        timing_ctx: FrameTimingContext,
    ) -> list[FrameMatch]:
        prepared, output_scale = self._stage_preprocess(frame_bgr, timing_ctx)
        
        # Defensive check: if preprocessing failed, we can't continue safely
        if prepared is None or prepared.bgr is None:
            logger.warning("Preprocessing failed for frame %s", frame_index)
            return []

        # ── Adaptive Load Governance (Phase 1 & 5) ───────────────────────────
        if self._settings.enable_adaptive_load_governance:
            self._apply_load_governance(frame_index)

        active = self._track_manager.active_tracks()
        stable_count = sum(1 for t in active if t.is_stable)

        # Use dynamic detector interval if overload is active
        detection_interval = self._settings.detector_interval_frames
        if self._overload_active or self._settings.enable_adaptive_load_governance:
            detection_interval = self._dynamic_detector_interval

        if not self._detector_cooldown_active and self._detection_optimizer.should_detect(
            frame_index,
            active_tracks=len(active),
            stable_tracks=stable_count,
            avg_motion_stability=self._track_manager.average_motion_stability(),
            detector_interval_override=detection_interval
        ):
            metrics.observe("average_detector_interval", float(detection_interval))
            
            # Phase 2D.2: Detector Invocation Cause Attribution
            if self._detection_optimizer.no_face_detected_streak >= 2:
                self._detection_optimizer.detector_invocation_cause = "tracking_starvation"
            elif self._detection_optimizer.zero_face_streaks >=3 or self._detection_optimizer.consecutive_weak_detections >=4:
                self._detection_optimizer.detector_invocation_cause = "resolution_escalation_retry"
            elif frame_index < 10:
                self._detection_optimizer.detector_invocation_cause = "initialization_bootstrap"
            else:
                self._detection_optimizer.detector_invocation_cause = "cadence_refresh"
            metrics.observe("phase2d2_detector_invocation_cause", 1.0 if self._detection_optimizer.detector_invocation_cause == "cadence_refresh" else 2.0 if self._detection_optimizer.detector_invocation_cause == "tracking_starvation" else 3.0)
            
            return self._stage_detection_path(prepared, frame_bgr, frame_index, gallery, output_scale, timing_ctx)
        return self._stage_tracking_path(prepared, frame_index, gallery, output_scale, timing_ctx)

    def _apply_load_governance(self, frame_index: int) -> None:
        """Dynamically adjust detector cadence and budget (Phase 1, 3, 4, 5)."""
        snapshot = metrics.snapshot()
        raw_pressure_band = int(snapshot.recent_values.get("tracking_pressure_band", [0.0])[-1])
        active_tracks = self._track_manager.active_track_count
        confirmed_tracks = self._track_manager.confirmed_track_count
        has_coarse = self._track_manager.has_coarse_tracks
        pending_queue = self._track_manager.candidate_queue_size
        det_runtime = snapshot.recent_values.get("current_frame_detector_ms", [0.0])[-1]
        
        # ── Phase 3: Emergency Recall Recovery ──────────────────────────────
        emergency_recall_active = False
        if (self._settings.enable_emergency_recall_mode and 
            confirmed_tracks == 0 and 
            raw_pressure_band >= 2):
            if pending_queue > 0 or has_coarse:
                emergency_recall_active = True
                if not self._emergency_rebuild_active:
                    metrics.increment("emergency_track_rebuilds")
                self._emergency_rebuild_active = True
                metrics.increment("emergency_recall_recoveries")
                metrics.observe("emergency_recall_mode_active", 1.0)
                logger.warning("GOVERNANCE: Emergency recall mode active (confirmed_tracks=0, candidates=%d, has_coarse=%s)", pending_queue, has_coarse)
        else:
            if self._emergency_rebuild_active and confirmed_tracks >= self._settings.governance_min_survival_tracks:
                if self._recovery_cooldown_frames == 0:
                    self._recovery_cooldown_frames = 15 # Hysteresis cooldown
                
                if self._recovery_cooldown_frames > 0:
                    self._recovery_cooldown_frames -= 1
                    if self._recovery_cooldown_frames == 0:
                        self._emergency_rebuild_active = False
                        metrics.increment("starvation_recovery_successes")
            
            metrics.observe("emergency_recall_mode_active", 0.0)

        # ── Phase 2: Governance Lockout Mode ──────────────────────────────────
        if confirmed_tracks == 0 and (pending_queue > 0 or has_coarse):
            if not self._governance_lockout_active:
                metrics.increment("governance_lockout_activations")
            self._governance_lockout_active = True
            self._track_manager.lockout_mode = True
            self._detection_optimizer.emergency_mode = True
        elif confirmed_tracks >= self._settings.governance_min_survival_tracks:
            self._governance_lockout_active = False
            self._track_manager.lockout_mode = False
            self._detection_optimizer.emergency_mode = False
        
        # 1. Update Cooldown State
        # Bypassed during lockout/emergency rebuild or when coarse tracks need recovery (Step 5)
        if self._governance_cooldown_frames > 0 and not self._governance_lockout_active and not has_coarse:
            self._governance_cooldown_frames -= 1
            self._detector_cooldown_active = True
            metrics.observe("detector_cooldown_active", 1.0)
            return
        else:
            if self._governance_lockout_active or has_coarse:
                self._governance_cooldown_frames = 0
            self._detector_cooldown_active = False
            metrics.observe("detector_cooldown_active", 0.0)

        # 2. Hysteresis Protection (Phase 1.3)
        if raw_pressure_band != self._current_pressure_band:
            self._band_persistence_frames += 1
            if self._band_persistence_frames >= self._settings.governance_pressure_hysteresis_frames:
                old_band = self._current_pressure_band
                self._current_pressure_band = raw_pressure_band
                self._band_persistence_frames = 0
                metrics.increment("pressure_hysteresis_transitions")
                logger.info("GOVERNANCE: Pressure band transitioned %d -> %d", old_band, self._current_pressure_band)
        else:
            self._band_persistence_frames = 0

        # 3. Phase 1 & 4: Adaptive Cadence & Thresholds
        # Hierarchy: Continuity > Compute Efficiency
        band = self._current_pressure_band
        
        # Base settings from band
        if band == 0: # NORMAL
            target_interval = self._settings.governance_low_pressure_interval
            target_conf = self._settings.relaxation_low_confidence
            target_cutoff = self._settings.relaxation_low_cutoff
        elif band == 1: # ELEVATED
            target_interval = self._settings.governance_medium_pressure_interval
            target_conf = self._settings.relaxation_medium_confidence
            target_cutoff = self._settings.relaxation_medium_cutoff
        elif band == 2: # HIGH
            target_interval = self._settings.governance_high_pressure_interval
            target_conf = self._settings.relaxation_high_confidence
            target_cutoff = self._settings.relaxation_high_cutoff
        else: # CRITICAL
            target_interval = self._settings.governance_high_pressure_interval + 4
            target_conf = self._settings.relaxation_high_confidence
            target_cutoff = self._settings.relaxation_high_cutoff
            if self._governance_cooldown_frames == 0:
                self._governance_cooldown_frames = self._settings.governance_critical_cooldown_frames
                logger.warning("LOAD GOVERNANCE: Entering CRITICAL cooldown for %d frames", self._governance_cooldown_frames)

        # ── Phase 3: Emergency Relaxation ────────────────────────────────────
        if emergency_recall_active or self._emergency_rebuild_active or self._governance_lockout_active:
            # Drop thresholds to absolute minimum to recover recall
            # Phase 3 linear reduction step: gradually relax
            current_conf = self._adaptive_det_confidence
            target_min_conf = 0.25
            if current_conf > target_min_conf:
                target_conf = max(target_min_conf, current_conf - 0.05)
            else:
                target_conf = target_min_conf
                
            current_cutoff = self._adaptive_validator_cutoff
            target_min_cutoff = 0.35
            if current_cutoff > target_min_cutoff:
                target_cutoff = max(target_min_cutoff, current_cutoff - 0.05)
            else:
                target_cutoff = target_min_cutoff

            # Reduce interval to get more frequent detections
            target_interval = max(getattr(self._settings, "detector_interval_min_frames", 4), 4)
            
            # During Lockout, disable aggressive degradation
            if self._governance_lockout_active:
                 target_interval = 1 # Force detection on every frame for immediate recovery

        self._adaptive_det_confidence = target_conf
        self._adaptive_validator_cutoff = target_cutoff

        # Apply adaptive thresholds (Phase 1.1)
        metrics.observe("adaptive_detector_confidence", self._adaptive_det_confidence)
        metrics.observe("adaptive_validator_cutoff", self._adaptive_validator_cutoff)

        # 4. Phase 5: Budget Enforcement
        if det_runtime > self._settings.governance_max_detector_runtime_ms:
            target_interval = max(target_interval, 16)
            metrics.increment("governance_budget_violations")
            
        # 5. Queue Pressure secondary check
        if pending_queue > self._settings.governance_max_candidate_queue_size:
            target_interval = max(target_interval, 14)

        # ── Phase 1: Minimum Survival Guarantees ─────────────────────────────
        # If we have very few tracks, or we have COARSE tracks awaiting recovery,
        # don't allow the interval to become too wide.
        if (active_tracks > 0 and active_tracks <= self._settings.governance_min_survival_tracks) or has_coarse:
            # If has_coarse, we want faster recovery if possible
            recovery_limit = 5 if has_coarse else 10
            target_interval = min(target_interval, recovery_limit)
            metrics.increment("protected_track_preservations")

        self._dynamic_detector_interval = target_interval
        metrics.observe("adaptive_detector_interval", float(target_interval))

    def _stage_preprocess(self, frame_bgr: np.ndarray, timing_ctx: FrameTimingContext):
        timing_ctx.preprocess_timer.begin()
        prepared = self._preprocessor.process(frame_bgr)
        timing_ctx.preprocess_timer.stop()
        output_scale = prepared.bgr.shape[1] / max(frame_bgr.shape[1], 1)
        return prepared, output_scale

    def _stage_detection_path(
        self,
        prepared,
        frame_bgr: np.ndarray,
        frame_index: int,
        gallery: list[tuple[int, np.ndarray]],
        output_scale: float,
        timing_ctx: FrameTimingContext,
    ) -> list[FrameMatch]:
        if self._detection_metrics:
            self._detection_metrics.record_frame_start(frame_index)

        matches: list[FrameMatch] = []
        
        # ── Phase 3.0: Dispatch Async Detection Request ─────────────────────────
        detection_frame, scale = self._detection_optimizer.prepare_for_detection(prepared.bgr)
        
        # Create request payload
        request = DetectionRequestPayload(
            frame_id=frame_index,
            timestamp=time.perf_counter(),
            prepared_frame_bgr=detection_frame,
            scale=scale
        )
        
        # Non-blocking put into request queue
        try:
            self._detection_request_queue.put_nowait(request)
            logger.debug("Dispatched detection request for frame %d", frame_index)
        except Exception:
            self._dropped_detection_requests +=1
            metrics.increment("phase3_dropped_detection_requests")
            logger.warning("Detection request queue full, dropped request for frame %d", frame_index)
        
        # ── Phase 3.0: Return Tracker-Only Matches for Current Frame ────────────
        # For now, just propagate the tracker (like _stage_tracking_path)
        self._detection_optimizer.observe_tracking_cycle()
        metrics.increment("frames_skipped")
        diagnostics.record("tracking", "detector_skipped_reused_tracks", frame_index=frame_index)

        timing_ctx.tracker_timer.begin()
        tracks = self._track_manager.propagate(frame_index, detector_interval=self._dynamic_detector_interval)
        timing_ctx.tracker_timer.stop()
        
        for track in tracks:
            if track.state not in ACTIVE_RECOGNITION_STATES:
                continue
            if track.confirmation_hits < self._tracking_cfg.confirm_frames:
                continue
            face = self._track_manager.to_detected_face(track)
            match = self._process_tracked_face(
                face=face,
                track=track,
                prepared=prepared,
                frame_index=frame_index,
                gallery=gallery,
                output_scale=output_scale,
                from_detection=False,
            )
            if match is not None:
                matches.append(match)

        total = max(1, len(tracks))
        emb_hits = metrics.snapshot().counters.get("embedding_cache_hits", 0)
        metrics.observe_rate("embedding_reuse_rate", float(emb_hits), float(total))
        self._finalize_frame(frame_index)
        self._export_tracking_metrics()
        
        return matches

    def _stage_tracking_path(
        self,
        prepared,
        frame_index: int,
        gallery: list[tuple[int, np.ndarray]],
        output_scale: float,
        timing_ctx: FrameTimingContext,
    ) -> list[FrameMatch]:
        self._detection_optimizer.observe_tracking_cycle()
        metrics.increment("frames_skipped")
        diagnostics.record("tracking", "detector_skipped_reused_tracks", frame_index=frame_index)

        matches: list[FrameMatch] = []
        timing_ctx.tracker_timer.begin()
        tracks = self._track_manager.propagate(frame_index, detector_interval=self._dynamic_detector_interval)
        timing_ctx.tracker_timer.stop()
        for track in tracks:
            if track.state not in ACTIVE_RECOGNITION_STATES:
                continue
            if track.confirmation_hits < self._tracking_cfg.confirm_frames:
                continue
            face = self._track_manager.to_detected_face(track)
            match = self._process_tracked_face(
                face=face,
                track=track,
                prepared=prepared,
                frame_index=frame_index,
                gallery=gallery,
                output_scale=output_scale,
                from_detection=False,
            )
            if match is not None:
                matches.append(match)

        total = max(1, len(tracks))
        emb_hits = metrics.snapshot().counters.get("embedding_cache_hits", 0)
        metrics.observe_rate("embedding_reuse_rate", float(emb_hits), float(total))
        self._finalize_frame(frame_index)
        self._export_tracking_metrics()
        return matches

    def _finalize_frame(self, frame_index: int) -> None:
        """Post-processing and metric aggregation (Phase 5)."""
        active_tracks = self._track_manager.active_tracks()
        
        # State Visibility (Phase 5)
        metrics.observe("emergency_recall_mode_active", 1.0 if self._settings.enable_emergency_recall_mode and self._track_manager.active_track_count == 0 else 0.0)
        
        # Aggressive expiration for TRACK_ONLY (garbage) tracks
        for track in active_tracks:
            if track.metadata.get("tier") == ValidationTier.TRACK_ONLY.value:
                # TRACK_ONLY tracks have much shorter lifetime (3 frames)
                if track.lost_frames > 2:
                    track.metadata["no_recovery"] = True
                    track.metadata["no_reassociation"] = True
            
            # Phase 3: Coarse tracking maintenance
            if track.state == TrackLifecycleState.COARSE.value:
                # COARSE tracks bypass biometric validation
                track.metadata["no_recovery"] = True
                track.metadata["no_reassociation"] = True

        for track in self._track_manager.consume_removed_tracks():
            self._global_identity_memory.archive_lost_track(track, frame_index)
        self._global_identity_memory.prune(frame_index)

    def _handle_detector_overload(self, frame_index: int, raw_count: int) -> None:
        """Actively manage pipeline parameters during high load."""
        self._overload_active = True
        # Push detector interval up to save CPU
        self._dynamic_detector_interval = min(16, self._settings.detector_interval_frames + 4)
        metrics.increment("detector_queue_stall_count")
        
        logger.warning(
            "Detector overload frame_index=%s count=%s. Activating load-shedding: interval=%s",
            frame_index, raw_count, self._dynamic_detector_interval
        )
        diagnostics.record("overload", "load_shedding_active", frame_index=frame_index, count=raw_count)

    def _process_tracked_face(
        self,
        *,
        face: DetectedFace,
        track: TrackedFace | None,
        prepared,
        frame_index: int,
        gallery: list[tuple[int, np.ndarray]],
        output_scale: float,
        from_detection: bool,
        validation_tier: ValidationTier = ValidationTier.STRICT_PASS,
    ) -> FrameMatch | None:
        width, height = _face_size(face)
        metrics.observe("avg_detected_face_area", width * height)

        # Stage 1: Validation Tiering & Presence
        if track is not None:
            track.metadata["tier"] = validation_tier.value
            
        if validation_tier == ValidationTier.TRACK_ONLY:
            return self._handle_track_only(face, track, prepared, frame_index, output_scale)
            
        if validation_tier == ValidationTier.WEAK_PASS:
            if not self._check_weak_pass_escalation(face, track, frame_index):
                return self._handle_weak_pass_deferred(face, track, prepared, frame_index, output_scale)
            metrics.increment("validator_weak_pass_escalated")

        # Stage 2: Quality & Geometry Safety (Legacy net)
        quality = self._stage_validate_quality(face, prepared, frame_index, from_detection)
        if not quality.accepted:
            return self._handle_quality_rejection(face, frame_index, prepared, output_scale, quality, width, height)

        # Stage 3: Embedding Policy & Reassociation
        emb = self._stage_embedding_policy(face, track, prepared, frame_index, quality)
        if emb is None:
            return None

        # Stage 4: Recognition & Identity Stabilization
        match_result = self._stage_recognition(face, track, emb, gallery, frame_index, quality, prepared)
        
        # Stage 5: Event Validation & Output
        return self._stage_event_generation(face, track, emb, match_result, prepared, frame_index, output_scale, quality, from_detection)

    def _handle_track_only(self, face, track, prepared, frame_index, output_scale) -> FrameMatch:
        """Handle presence-only tracking for non-human or low-quality objects."""
        if track is not None:
            track.metadata["no_reassociation"] = True
            track.metadata["no_recovery"] = True
            
        metrics.increment("validator_embedding_skips")
        return FrameMatch(
            frame_index=frame_index, person_id=None, confidence=None,
            threshold=self._settings.match_confidence_threshold,
            reason="track_only", track_id=track.numeric_track_id if track else None,
            face=_scale_face_for_output(face, output_scale),
            trace=_trace(face, "yellow", ("DETECTED", "TRACK_ONLY"),
                "track_only", detector_confidence=face.det_score,
                frame_shape=prepared.bgr.shape, output_scale=output_scale,
                validation_tier=ValidationTier.TRACK_ONLY.value),
        )

    def _check_weak_pass_escalation(self, face: DetectedFace, track: TrackedFace | None, frame_index: int) -> bool:
        """Time and stability-based trust accumulation for weak detections."""
        if track is None:
            return False
            
        # Time-based trust: stable tiny faces eventually get embeddings
        weak_promote_age = self._settings.validator_weak_pass_retry_limit * 3
        temporal = get_temporal_identity(track)
        motion_score = float(track.metadata.get("motion_score", 1.0))
        
        # Multi-signal stability
        landmark_score = float(track.metadata.get("landmark_score", 0.0))
        persistence = min(1.0, track.visibility_age / max(self._settings.tracking_stable_frames, 1))
        
        return (
            (
                track.visibility_age >= weak_promote_age 
                and temporal.temporal_consistency >= self._settings.validator_weak_pass_temporal_min 
                and landmark_score > self._settings.validator_weak_pass_landmark_min
            )
            or face.det_score >= (self._settings.validator_min_detector_confidence + self._settings.validator_weak_pass_confidence_boost)
            or (
                temporal.temporal_consistency >= self._settings.validator_weak_pass_soft_temporal 
                and motion_score > self._settings.validator_weak_pass_motion_min 
                and persistence > self._settings.validator_weak_pass_persistence_min
            )
        )

    def _handle_weak_pass_deferred(self, face, track, prepared, frame_index, output_scale) -> FrameMatch:
        """Defer recognition for weak candidates until they earn trust."""
        metrics.increment("validator_embedding_skips")
        return FrameMatch(
            frame_index=frame_index, person_id=None, confidence=None,
            threshold=self._settings.match_confidence_threshold,
            reason="weak_pass_deferred", track_id=track.numeric_track_id if track else None,
            face=_scale_face_for_output(face, output_scale),
            trace=_trace(face, "yellow", ("DETECTED", "WEAK_PASS"),
                "weak_pass_deferred", detector_confidence=face.det_score,
                frame_shape=prepared.bgr.shape, output_scale=output_scale,
                validation_tier=ValidationTier.WEAK_PASS.value),
        )

    def _stage_validate_quality(self, face, prepared, frame_index, from_detection):
        """Stage 2: Apply legacy quality/crop validators as safety net."""
        if from_detection and self._settings.enable_legacy_quality_checks:
            crop_check = self._crop_validator.validate(prepared.bgr, face)
            if not crop_check.accepted:
                return crop_check # Result has accepted=False

        if self._settings.enable_legacy_quality_checks:
            return self._quality_assessor.assess(prepared.bgr, face)
        else:
            from ecoface_lite.ai_engine.face_quality import FaceQualityResult
            return FaceQualityResult(True, blur_score=50.0, quality_score=0.7)

    def _stage_embedding_policy(self, face, track, prepared, frame_index, quality):
        """Stage 3: Decide if embedding is needed; apply reassociation if new track."""
        if track is None:
            track = self._track_manager.candidate_track(face, frame_index)

        # Phase 3: Coarse Tracking Biometric Bypass
        if track is not None and track.state == TrackLifecycleState.COARSE.value:
            metrics.increment("biometric_bypass_events")
            if track.last_embedding is not None:
                return track.last_embedding
            return None

        track_quality = None
        if track is not None:
            track_quality = self._track_manager.update_track_quality(track, face, prepared.bgr, frame_index)

        need_embedding = track is None or self._track_manager.should_compute_embedding(
            track, frame_index, face, quality=track_quality
        )
        
        # Phase 4: Priority-Aware Biometric Budgeting
        if track is not None and need_embedding:
            priority = self._track_manager._calculate_track_priority(track)
            pressure_band = self._current_pressure_band
            
            # P3 (Background) + High Pressure = Degrade Background
            if priority >= 3 and pressure_band >= 2:
                metrics.increment("degraded_background_tracks")
                if track.last_embedding is not None:
                    return track.last_embedding
                return None
                
            # P2 (Recently confirmed) + High Pressure = Partial throttle
            if priority >= 2 and pressure_band >= 2:
                # 75% chance to skip if we already have an identity
                if track.identity is not None and (hash(track.track_id + str(frame_index)) % 100 < 75):
                    metrics.increment("priority_embedding_throttles")
                    return track.last_embedding

            metrics.increment("priority_embedding_allocations")

        # Embedding Cooldown: Avoid thrashing in unstable scenes
        if track is not None and not need_embedding:
            last_emb_frame = track.last_embedding_frame
            min_interval = self._tracking_cfg.embedding_cooldown_frames
            
            # Significant quality improvement override
            prev_quality = track.metadata.get("last_quality_score", 0.0)
            quality_jump = quality.quality_score - prev_quality > self._tracking_cfg.embedding_quality_jump
            
            if (frame_index - last_emb_frame < min_interval) and not quality_jump:
                if track.last_embedding is not None:
                    metrics.increment("embedding_skips_cooldown")
                    return track.last_embedding
        
        # Check cache/reuse
        if track is not None and not need_embedding and track.last_embedding is not None:
            metrics.increment("embedding_cache_hits")
            return track.last_embedding

        # Generate new embedding
        with metrics.timer("embedding_generation_duration"):
            emb = self._embedder.embed_face(prepared.bgr, face)
        metrics.increment("embeddings_generated")
        
        if track is not None:
            track.touch_embedding(emb, frame_index)
            track.metadata["last_quality_score"] = quality.quality_score
            # Reassociation attempt for new tracks
            if track.visibility_age <= 2 and track.recovery_count == 0:
                if not track.metadata.get("no_recovery"):
                    self._track_reassociator.try_recover_identity(track, emb, frame_index)
        
        return emb

    def _stage_recognition(self, face, track, emb, gallery, frame_index, quality, prepared):
        """Stage 4: Multi-stage identity matching and stabilization."""
        # Use diagnostics from prepared frame for adaptive thresholds
        diagnostics_obj = prepared.diagnostics if prepared is not None else None
        policy_threshold = self._confidence_policy.threshold_for(diagnostics_obj, quality)
        
        if track is not None:
            fusion_weight = self._identity_confidence.embedding_suppression_weight(
                track,
                blur_score=quality.blur_score,
                quality_weight=quality.quality_score,
            )
            
            pose_bucket = None
            if face.landmarks is not None:
                pose_bucket = classify_pose_bucket(face.landmarks, face.bbox)
                get_temporal_identity(track).record_pose_blur(pose_bucket, quality.blur_score)

            with metrics.timer("matching_duration"):
                decision_match = self._identity_matcher.match_track(
                    track, emb, gallery, policy_threshold,
                    quality_weight=fusion_weight,
                    frame_index=frame_index,
                    pose_bucket=pose_bucket,
                    blur_score=quality.blur_score,
                )
            
            if decision_match and decision_match.person_id is not None:
                # Candidate Persistence: Accumulate uncertain identity evidence
                track.metadata["candidate_identity"] = decision_match.person_id
                track.metadata["candidate_confidence"] = decision_match.confidence
                track.metadata["candidate_last_frame"] = frame_index
                
                id_conf = self._identity_confidence.evaluate(
                    track, decision_match.confidence, policy_threshold,
                    person_id=decision_match.person_id,
                )
                
                accepted_stages = {"verified", "soft_verified", "locked", "weak_candidate"}
                if decision_match.stage in accepted_stages:
                    if decision_match.confidence >= policy_threshold or decision_match.soft_match or id_conf.soft_accept:
                        conf = max(decision_match.confidence, id_conf.temporal_confidence)
                        return MatchResult(person_id=decision_match.person_id, confidence=conf)
            return None
        else:
            with metrics.timer("matching_duration"):
                return self._matcher.best_match(emb, gallery, policy_threshold)

    def _stage_event_generation(self, face, track, emb, match_result, prepared, frame_index, output_scale, quality, from_detection):
        """Stage 5: Final decision on event emission and overlay state."""
        if match_result is None:
            metrics.increment("failed_matches")
            return self._handle_failed_match(face, track, frame_index, output_scale, prepared, quality, from_detection)

        metrics.increment("successful_matches")
        
        # Identity session observation
        recognition = (
            self._recognition_session.observe_track(track, match_result.person_id, match_result.confidence)
            if track is not None
            else self._recognition_session.observe(face, frame_index, match_result.person_id, match_result.confidence)
        )

        # Event validation
        with metrics.timer("event_validation_duration"):
            event = self._event_validator.evaluate(recognition, frame_index)

        # Determine overlay color
        overlay_state = "green" if event.should_emit else ("yellow" if recognition.confirmations > 0 else "red")
        if event.should_emit:
            metrics.increment("detection_events_validated")
            metrics.increment("green_box_count")
        else:
            metrics.increment("red_box_count")

        return FrameMatch(
            frame_index=frame_index,
            person_id=recognition.person_id if recognition.stable else match_result.person_id,
            confidence=recognition.smoothed_confidence or match_result.confidence,
            threshold=self._settings.match_confidence_threshold,
            stable=recognition.stable,
            should_alert=event.should_emit,
            track_id=recognition.track_id,
            reason=event.reason if not event.should_emit else "accepted",
            face=_scale_face_for_output(face, output_scale),
            trace=_trace(
                face, overlay_state,
                ("DETECTED", "TRACKED", "RECOGNIZED", "VALIDATED") if event.should_emit else ("DETECTED", "TRACKED", "RECOGNIZED"),
                event.reason if not event.should_emit else None,
                detector_confidence=face.det_score,
                blur_score=quality.blur_score,
                frame_shape=prepared.bgr.shape,
                output_scale=output_scale,
            ),
        )

    def _handle_failed_match(self, face, track, frame_index, output_scale, prepared, quality, from_detection) -> FrameMatch:
        metrics.increment("red_box_count")
        return FrameMatch(
            frame_index=frame_index, person_id=None, confidence=None,
            threshold=self._settings.match_confidence_threshold,
            reason="no_match_above_threshold",
            track_id=track.numeric_track_id if track else None,
            face=_scale_face_for_output(face, output_scale),
            trace=_trace(
                face, "red",
                ("TRACKED" if not from_detection else "DETECTED", "FILTERED", "EMBEDDED", "MATCHED_FAILED"),
                "no_match_above_threshold",
                detector_confidence=face.det_score,
                blur_score=quality.blur_score,
                frame_shape=prepared.bgr.shape,
                output_scale=output_scale,
            ),
        )

    def _handle_quality_rejection(self, face, frame_index, prepared, output_scale, quality, width, height) -> FrameMatch:
        # Check if it was a crop rejection or assessment rejection
        reason = getattr(quality, "reason", "quality_rejected")
        label = f"REJECTED: {reason}"
        return self._rejection_match(
            face, frame_index, prepared, output_scale,
            label, "yellow", ("DETECTED", "QUALITY_REJECTED")
        )

    def _quality_rejection(self, face, frame_index, prepared, output_scale, quality, width, height) -> FrameMatch:
        metrics.increment("rejected_faces")
        metrics.increment("yellow_box_count")
        metrics.observe("avg_rejected_face_area", width * height)
        if quality.reason == "blurry_face":
            metrics.increment("rejected_due_to_blur")
        elif quality.reason == "face_too_small":
            metrics.increment("rejected_due_to_size")
        elif quality.reason == "low_detection_confidence":
            metrics.increment("rejected_due_to_low_confidence")
        diagnostics.record(
            "rejection",
            quality.reason or "quality_rejected",
            frame_index=frame_index,
            metadata={"blur_score": quality.blur_score, "det_score": face.det_score},
        )
        return FrameMatch(
            frame_index=frame_index,
            person_id=None,
            confidence=None,
            threshold=self._settings.match_confidence_threshold,
            reason=quality.reason,
            face=_scale_face_for_output(face, output_scale),
            trace=_trace(
                face,
                "yellow",
                ("DETECTED", "FILTERED", "QUALITY_REJECTED"),
                quality.reason,
                detector_confidence=face.det_score,
                blur_score=quality.blur_score,
                frame_shape=prepared.bgr.shape,
                output_scale=output_scale,
            ),
        )

    def _rejection_match(self, face, frame_index, prepared, output_scale, reason, color, stages) -> FrameMatch:
        metrics.increment("rejected_faces")
        metrics.increment("yellow_box_count")
        width, height = _face_size(face)
        metrics.observe("avg_rejected_face_area", width * height)
        diagnostics.record(
            "detector_filter",
            reason,
            frame_index=frame_index,
            metadata={"det_score": face.det_score, "face_width": width, "face_height": height},
        )
        return FrameMatch(
            frame_index=frame_index,
            person_id=None,
            confidence=None,
            threshold=self._settings.match_confidence_threshold,
            reason=reason,
            face=_scale_face_for_output(face, output_scale),
            trace=_trace(
                face,
                color,
                stages,
                reason,
                detector_confidence=face.det_score,
                frame_shape=prepared.bgr.shape,
                output_scale=output_scale,
            ),
        )

    def _match_from_track_identity(
        self,
        face: DetectedFace,
        track: TrackedFace,
        prepared,
        frame_index: int,
        gallery: list[tuple[int, np.ndarray]],
        output_scale: float,
        from_detection: bool,
        quality,
    ) -> FrameMatch:
        """Reuse stabilized track identity without re-embedding or re-matching."""
        metrics.increment("embedding_skips")
        metrics.increment("embedding_cache_hits")
        person_id = int(track.identity) if track.identity is not None else -1
        confidence = track.smoothed_confidence or track.identity_confidence
        recognition = self._recognition_session.stable_from_track(track)
        threshold = self._confidence_policy.threshold_for(prepared.diagnostics, quality)
        unstable = recognition.stable_match_count < self._tracking_cfg.min_stable_matches
        overlay_state = "green" if recognition.stable else ("yellow" if unstable else "red")
        with metrics.timer("event_validation_duration"):
            event = self._event_validator.evaluate(recognition, frame_index)
        if event.should_emit:
            metrics.increment("detection_events_validated")
            metrics.increment("green_box_count")
            overlay_state = "green"
        elif overlay_state != "yellow":
            metrics.increment("red_box_count")
        return FrameMatch(
            frame_index=frame_index,
            person_id=recognition.person_id if recognition.stable else person_id,
            confidence=recognition.smoothed_confidence or confidence,
            threshold=threshold,
            stable=recognition.stable,
            should_alert=event.should_emit,
            track_id=recognition.track_id,
            reason=event.reason if not event.should_emit else "accepted",
            face=_scale_face_for_output(face, output_scale),
            trace=_trace(
                face,
                overlay_state,
                ("TRACKED", "RECOGNIZED", "CACHED") if not from_detection else ("DETECTED", "TRACKED", "CACHED"),
                event.reason if not event.should_emit else None,
                detector_confidence=face.det_score,
                blur_score=quality.blur_score,
                frame_shape=prepared.bgr.shape,
                output_scale=output_scale,
            ),
        )

    def _export_tracking_metrics(self) -> None:
        active = self._track_manager.active_track_count
        metrics.observe("active_tracks", active)
        tracks = self._track_manager.active_tracks()
        if tracks:
            metrics.observe("avg_track_lifetime", sum(t.visibility_age for t in tracks) / len(tracks))
            consistencies = [
                float(t.metadata.get("temporal_consistency", 0.0))
                for t in tracks
                if t.metadata.get("temporal_consistency") is not None
            ]
            if consistencies:
                metrics.observe("avg_temporal_consistency", sum(consistencies) / len(consistencies))
            switches = sum(t.identity_switch_count for t in tracks)
            metrics.observe("identity_switch_rate", switches / max(len(tracks), 1))
        # Phase 2A: validator embedding skip rate
        emb_skips = metrics.snapshot().counters.get("validator_embedding_skips", 0)
        emb_total = metrics.snapshot().counters.get("embeddings_generated", 0) + emb_skips
        metrics.observe_rate("validator_embedding_skip_rate", float(emb_skips), float(max(emb_total, 1)))

    def _validator_rejection_match(self, face, frame_index, prepared, output_scale, result: ValidationResult) -> FrameMatch:
        metrics.increment("rejected_faces")
        metrics.increment("yellow_box_count")
        width, height = _face_size(face)
        metrics.observe("avg_rejected_face_area", width * height)
        reason = result.primary_reason or "validator_rejected"
        diagnostics.record(
            "validator_rejection", reason, frame_index=frame_index,
            metadata={"quality_score": result.quality_score, "det_score": face.det_score},
        )
        return FrameMatch(
            frame_index=frame_index, person_id=None, confidence=None,
            threshold=self._settings.match_confidence_threshold,
            reason=reason, face=_scale_face_for_output(face, output_scale),
            trace=_trace(
                face, "yellow", ("DETECTED", "VALIDATOR_REJECTED"), reason,
                detector_confidence=face.det_score, blur_score=result.blur_score,
                frame_shape=prepared.bgr.shape, output_scale=output_scale,
                validation_tier=result.tier.value, quality_score=result.quality_score,
                fused_confidence=result.fused_confidence,
                validator_reasons=result.rejection_reasons,
            ),
        )

    @staticmethod
    def _face_key(face: DetectedFace) -> tuple[float, float, float, float]:
        return (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2)

    def test_match_frame(
        self,
        frame_bgr: np.ndarray,
        gallery: list[tuple[int, np.ndarray]],
    ) -> MatchResult | None:
        prepared = self._preprocessor.process(frame_bgr)
        faces = self._detector.detect(prepared.bgr)
        if not faces:
            return None
        best_face = max(faces, key=lambda f: f.det_score)
        quality = self._quality_assessor.assess(prepared.bgr, best_face)
        if not quality.accepted:
            return None
        emb = self._embedder.embed_face(prepared.bgr, best_face)
        top = self._matcher.top_match(emb, gallery)
        if top is None:
            return None
        threshold = self._confidence_policy.threshold_for(prepared.diagnostics, quality)
        if top.confidence < threshold:
            return MatchResult(person_id=top.person_id, confidence=top.confidence)
        return top


def _face_size(face) -> tuple[int, int]:
    geometry = compute_face_geometry(face, (10_000, 10_000, 3))
    return geometry.width, geometry.height


def _trace(
    face,
    state: str,
    stages: tuple[str, ...],
    reason: str | None,
    *,
    detector_confidence: float | None,
    blur_score: float | None = None,
    frame_shape: tuple[int, ...] | None = None,
    output_scale: float = 1.0,
    validation_tier: str | None = None,
    quality_score: float | None = None,
    fused_confidence: float | None = None,
    validator_reasons: tuple[str, ...] | None = None,
) -> FaceDebugTrace:
    geometry = compute_face_geometry(face, frame_shape or (10_000, 10_000, 3))
    width = max(1, int(round(geometry.width / max(output_scale, 1e-6))))
    height = max(1, int(round(geometry.height / max(output_scale, 1e-6))))
    return FaceDebugTrace(
        state=state,
        stages=stages,
        face_width=width,
        face_height=height,
        detector_confidence=detector_confidence,
        blur_score=blur_score,
        rejection_reason=reason,
        validation_tier=validation_tier,
        quality_score=quality_score,
        fused_confidence=fused_confidence,
        validator_reasons=validator_reasons,
    )


def _scale_face_for_output(face, output_scale: float):
    if output_scale == 1.0:
        return face
    return scale_face_to_original(face, output_scale)

"""High-level pipeline orchestration (video frame → optional alert).

Tracking-first architecture:
  detect occasionally → track continuously → recognize intelligently
"""

from __future__ import annotations

import numpy as np

from ecoface_lite.ai_engine.confidence import ConfidencePolicy
from ecoface_lite.ai_engine.detection_optimizer import DetectionOptimizer
from ecoface_lite.ai_engine.diagnostics import diagnostics
from ecoface_lite.ai_engine.detector import DetectedFace, FaceDetector
from ecoface_lite.ai_engine.detection.detectors import SCRFDDetector, MultiScaleDetector
from ecoface_lite.ai_engine.detection.fusion import WeightedBoxFusion, DuplicateFilter, ConfidenceNormalizer
from ecoface_lite.ai_engine.detection.temporal import WeakDetectionMemory
from ecoface_lite.ai_engine.embedder import FaceEmbedder
from ecoface_lite.ai_engine.event_validator import EventValidator
from ecoface_lite.ai_engine.face_candidate_validator import FaceCandidateValidator
from ecoface_lite.ai_engine.face_crop_validator import FaceCropValidator
from ecoface_lite.ai_engine.face_quality import FaceQualityAssessor
from ecoface_lite.ai_engine.temporal_detector import TemporalDetectorFilter
from ecoface_lite.ai_engine.geometry import compute_face_geometry, scale_face_to_original
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
from ecoface_lite.ai_engine.tracking.track_state import ACTIVE_RECOGNITION_STATES
from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
from ecoface_lite.config.tracking import TrackingConfig, get_tracking_config
from ecoface_lite.core.config import Settings
from ecoface_lite.core.detection_metrics import DetectionMetricsCollector, RecallMetricsCalculator, FalsePositiveLogger
from ecoface_lite.core.experiment_export import ExperimentExporter, EventTimeline, ExperimentNotesTracker
from ecoface_lite.core.logging import get_logger
from ecoface_lite.core.metrics import metrics
from ecoface_lite.core.validator import FaceValidator, ValidationTier, ValidationResult

logger = get_logger(__name__)


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
    ) -> None:
        self._settings = settings
        self._tracking_cfg: TrackingConfig = get_tracking_config(settings)

        # ── Phase 2A.2: Wrap detector with MultiScaleDetector if enabled ──────────
        if settings.enable_multiscale_detection:
            # Convert InsightFaceDetector to SCRFDDetector wrapper
            if hasattr(detector, '_app'):
                # Extract parameters from existing detector
                model_name = getattr(detector, '_model_name', 'buffalo_l')
                ctx_id = getattr(detector, '_ctx_id', -1)
                det_size = getattr(detector, '_det_size', (640, 640))
                face_app = getattr(detector, '_injected_app', None) or getattr(detector, '_app', None)

                # Create SCRFDDetector wrapper
                scrfd_detector = SCRFDDetector(
                    model_name=model_name,
                    ctx_id=ctx_id,
                    face_app=face_app,
                    default_det_size=det_size,
                )

                # Wrap with MultiScaleDetector
                self._detector = MultiScaleDetector(scrfd_detector, settings)
                logger.info("Multi-scale detection enabled with scales: %s", settings.multiscale_scales)
            else:
                # Already a BaseDetector or compatible
                self._detector = detector
                logger.warning("Detector does not support multi-scale wrapping, using as-is")
        else:
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

        # ── Phase 2A.1: Detection Observability Foundation ─────────────────────
        if settings.detection_metrics_enabled:
            self._detection_metrics = DetectionMetricsCollector(
                export_dir=settings.resolved_detection_metrics_log_dir(),
                export_interval=settings.detection_metrics_export_interval,
            )
            self._recall_metrics = RecallMetricsCalculator(window_size=100)
            self._false_positive_logger = FalsePositiveLogger(
                base_dir=settings.resolved_false_positive_dataset_dir(),
                enabled=settings.false_positive_snapshot_enabled,
                max_snapshots=settings.false_positive_max_snapshots,
                sampling_rate=settings.false_positive_sampling_rate,
                min_confidence=settings.false_positive_min_confidence,
            )
        else:
            self._detection_metrics = None
            self._recall_metrics = None
            self._false_positive_logger = None

        # ── Phase 2A.4: Proposal Fusion Engine ─────────────────────────────────
        if settings.enable_confidence_normalization:
            from ecoface_lite.ai_engine.detection.fusion import FusionConfig, NormalizationConfig

            fusion_config = FusionConfig(
                iou_threshold=settings.fusion_wbf_iou_threshold,
                crowd_iou_threshold=settings.fusion_crowd_iou_threshold,
                scale_weight_tiny=settings.fusion_scale_weight_tiny,
                scale_weight_small=settings.fusion_scale_weight_small,
                scale_weight_baseline=settings.fusion_scale_weight_baseline,
            )

            norm_config = NormalizationConfig(
                tiny_threshold=settings.multiscale_tiny_face_threshold,
                small_threshold=settings.multiscale_small_face_threshold,
            )

            self._wbf_fusion = WeightedBoxFusion(fusion_config)
            self._duplicate_filter = DuplicateFilter()
            self._confidence_normalizer = ConfidenceNormalizer(norm_config)
            logger.info("Proposal fusion engine enabled")
        else:
            self._wbf_fusion = None
            self._duplicate_filter = None
            self._confidence_normalizer = None

        # ── Phase 2A.5: Temporal Weak Detection Recovery ────────────────────────
        if settings.enable_weak_detection_memory:
            from ecoface_lite.ai_engine.detection.temporal import MemoryConfig

            memory_config = MemoryConfig(
                max_frames=settings.weak_memory_max_frames,
                cluster_iou=settings.weak_memory_cluster_iou,
                min_recurrence=settings.weak_memory_min_recurrence,
                promotion_boost=settings.weak_memory_promotion_boost,
                max_boost=settings.weak_memory_max_boost,
                motion_threshold=settings.weak_memory_motion_threshold,
            )

            self._weak_detection_memory = WeakDetectionMemory(memory_config)
            logger.info("Temporal weak detection memory enabled")
        else:
            self._weak_detection_memory = None

        # ── Experiment Export System ───────────────────────────────────────────
        if settings.export_enabled:
            from ecoface_lite.core.experiment_export import ExportConfig

            export_config = ExportConfig(
                export_format=settings.export_format,
                include_screenshots=settings.export_include_screenshots,
                include_false_positives=settings.export_include_false_positives,
                include_graph_data=settings.export_include_graph_data,
                include_event_timeline=settings.export_include_event_timeline,
                compress_images=settings.export_compress_images,
            )

            self._experiment_exporter = ExperimentExporter(settings, export_config)
            self._event_timeline = EventTimeline()
            self._notes_tracker = ExperimentNotesTracker()

            # Link timeline and tracker to exporter
            self._experiment_exporter.set_event_timeline(self._event_timeline)
            self._experiment_exporter.set_notes_tracker(self._notes_tracker)

            logger.info("Experiment export system enabled")
        else:
            self._experiment_exporter = None
            self._event_timeline = None
            self._notes_tracker = None

    @property
    def _track_manager(self):
        return self._recognition_session.track_manager

    def flush_metrics(self) -> None:
        """Flush detection metrics to disk (call before shutdown)."""
        if self._detection_metrics:
            self._detection_metrics.flush()
        if self._false_positive_logger:
            export_path = self._settings.resolved_false_positive_dataset_dir() / "metadata.json"
            self._false_positive_logger.export_metadata(export_path)

    def export_experiment_session(
        self,
        video_name: str,
        video_duration: float = 0.0,
        frame_count: int = 0,
        test_operator: str = "",
        notes: str = "",
    ) -> Path:
        """Export complete experiment session.

        Args:
            video_name: Name of the video file
            video_duration: Duration in seconds
            frame_count: Total number of frames
            test_operator: Name of the test operator
            notes: Experimental notes

        Returns:
            Path to exported file or directory
        """
        if not self._experiment_exporter:
            raise RuntimeError("Experiment export system is not enabled")

        # Set metadata
        self._experiment_exporter.set_metadata(
            video_name=video_name,
            video_duration=video_duration,
            frame_count=frame_count,
            test_operator=test_operator,
            notes=notes,
        )

        # Collect metrics data
        metrics_data = {}
        if self._detection_metrics:
            metrics_data["per_frame_metrics"] = self._detection_metrics.get_all_metrics()

        # Export
        export_dir = self._settings.resolved_export_dir()
        export_dir.mkdir(parents=True, exist_ok=True)

        return self._experiment_exporter.export_session(export_dir, metrics_data)

    def record_experiment_adjustment(
        self,
        adjustment: str,
        old_value: Any,
        new_value: Any,
        reason: str,
    ) -> None:
        """Record an experimental adjustment.

        Args:
            adjustment: Description of the adjustment
            old_value: Previous value
            new_value: New value
            reason: Reason for the adjustment
        """
        if self._notes_tracker:
            self._notes_tracker.record_adjustment(
                adjustment=adjustment,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
            )

    def get_experiment_notes(self) -> str:
        """Get experiment notes summary.

        Returns:
            Formatted summary of all adjustments
        """
        if self._notes_tracker:
            return self._notes_tracker.get_summary()
        return "No experiment notes available."

    def get_event_timeline_statistics(self) -> dict[str, Any]:
        """Get event timeline statistics.

        Returns:
            Dictionary with event statistics
        """
        if self._event_timeline:
            return self._event_timeline.get_statistics()
        return {}

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
        with metrics.timer("total_frame_processing_duration"):
            return self._process_frame_staged(frame_bgr, frame_index, gallery)

    def _process_frame_staged(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        gallery: list[tuple[int, np.ndarray]],
    ) -> list[FrameMatch]:
        prepared, output_scale = self._stage_preprocess(frame_bgr)
        active = self._track_manager.active_tracks()
        stable_count = sum(1 for t in active if t.is_stable)
        if self._detection_optimizer.should_detect(
            frame_index,
            active_tracks=len(active),
            stable_tracks=stable_count,
            avg_motion_stability=self._track_manager.average_motion_stability(),
        ):
            return self._stage_detection_path(prepared, frame_bgr, frame_index, gallery, output_scale)
        return self._stage_tracking_path(prepared, frame_index, gallery, output_scale)

    def _stage_preprocess(self, frame_bgr: np.ndarray):
        with metrics.timer("preprocessing_duration"):
            prepared = self._preprocessor.process(frame_bgr)
        output_scale = prepared.bgr.shape[1] / max(frame_bgr.shape[1], 1)
        return prepared, output_scale

    def _stage_detection_path(
        self,
        prepared,
        frame_bgr: np.ndarray,
        frame_index: int,
        gallery: list[tuple[int, np.ndarray]],
        output_scale: float,
    ) -> list[FrameMatch]:
        matches: list[FrameMatch] = []

        # ── Phase 2A.1: Record frame start for metrics ─────────────────────────
        if self._detection_metrics:
            frame_start_time = self._detection_metrics.record_frame_start(frame_index)

        detection_frame, scale = self._detection_optimizer.prepare_for_detection(prepared.bgr)
        with metrics.timer("face_detection_duration"):
            raw_faces = self._detector.detect(detection_frame)
        detection_latency_ms = metrics.snapshot().recent_values.get("face_detection_duration", [0.0])[-1] * 1000.0
        metrics.observe(
            "detector_runtime_ms",
            detection_latency_ms,
        )
        raw_faces = self._detection_optimizer.scale_faces(raw_faces, scale)
        raw_faces = self._temporal_detector.apply(raw_faces, frame_index)

        # ── Phase 2A.4: Apply proposal fusion if enabled ─────────────────────────
        if self._wbf_fusion and self._confidence_normalizer:
            # Detect if this is a crowd scene
            is_crowd = len(raw_faces) >= 8

            # Normalize confidence across scales
            raw_faces = self._confidence_normalizer.normalize(raw_faces)

            # Apply weighted box fusion
            raw_faces = self._wbf_fusion.fuse(raw_faces, prepared.bgr.shape, is_crowd_scene=is_crowd)

            # Apply duplicate filtering
            raw_faces = self._duplicate_filter.filter(raw_faces, is_crowd_scene=is_crowd)

            # Limit to max proposals
            if len(raw_faces) > self._settings.fusion_max_proposals_per_frame:
                raw_faces = sorted(raw_faces, key=lambda f: f.det_score, reverse=True)
                raw_faces = raw_faces[:self._settings.fusion_max_proposals_per_frame]

            logger.debug(
                "Fusion applied: %d -> %d faces, crowd=%s",
                len(raw_faces),
                len(raw_faces),
                is_crowd,
            )

        # ── Phase 2A.5: Apply temporal weak detection recovery ───────────────────
        if self._weak_detection_memory:
            promoted = self._weak_detection_memory.update(raw_faces, frame_index)

            # Apply confidence boosts to promoted faces
            for face, boost in promoted:
                # Find the face in raw_faces and boost its confidence
                for i, rf in enumerate(raw_faces):
                    if id(rf) == id(face):
                        boosted_confidence = min(1.0, rf.det_score + boost)
                        raw_faces[i] = DetectedFace(
                            bbox=rf.bbox,
                            det_score=boosted_confidence,
                            aligned_face=rf.aligned_face,
                            embedding=rf.embedding,
                            landmarks=rf.landmarks,
                            temporal_score=rf.temporal_score,
                        )
                        logger.debug(
                            "Weak detection promotion applied: boost=%.3f, new_conf=%.3f",
                            boost,
                            boosted_confidence,
                        )

                        # ── Export: Record weak detection promotion event ─────────────
                        if self._event_timeline:
                            face_size = max(rf.bbox.x2 - rf.bbox.x1, rf.bbox.y2 - rf.bbox.y1)
                            self._event_timeline.record_weak_detection_promoted(
                                frame_id=frame_index,
                                track_id=None,
                                face_size=face_size,
                                confidence_before=rf.det_score,
                                confidence_after=boosted_confidence,
                            )
                        break

        # ── Phase 2A: Unified Face Validator pre-filter ─────────────────────
        validator_results: dict[str, ValidationResult] = {}
        face_tier: dict[int, ValidationTier] = {}
        strict_pass_faces: list[DetectedFace] = []
        weak_pass_faces: list[DetectedFace] = []
        track_only_faces: list[DetectedFace] = []

        for idx, face in enumerate(raw_faces):
            face_uuid = f"{frame_index}_{idx}"
            result = self._unified_face_validator.validate(
                face, prepared.bgr, prepared.bgr.shape, frame_index
            )
            validator_results[face_uuid] = result
            face_tier[id(face)] = result.tier

            if result.tier == ValidationTier.REJECT:
                matches.append(self._validator_rejection_match(
                    face, frame_index, prepared, output_scale, result
                ))
                metrics.increment("validator_reject_count")

                # ── Phase 2A.1: Log false positive for rejected faces ─────────────
                if self._false_positive_logger and result.primary_reason:
                    self._false_positive_logger.log_false_positive(
                        frame_bgr=prepared.bgr,
                        frame_id=frame_index,
                        bbox=(face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2),
                        confidence=face.det_score,
                        rejection_reason=result.primary_reason,
                        category="unknown",  # Will be enhanced with motion consistency in 2A.6
                        metadata={"validation_score": result.quality_score},
                    )

                # ── Export: Record validator rejection event ─────────────────────
                if self._event_timeline:
                    face_size = max(face.bbox.x2 - face.bbox.x1, face.bbox.y2 - face.bbox.y1)
                    self._event_timeline.record_validator_rejected(
                        frame_id=frame_index,
                        track_id=None,
                        face_size=face_size,
                        confidence=face.det_score,
                        rejection_reason=result.primary_reason,
                    )
            elif result.tier == ValidationTier.TRACK_ONLY:
                track_only_faces.append(face)
                metrics.increment("validator_track_only_count")
            elif result.tier == ValidationTier.WEAK_PASS:
                weak_pass_faces.append(face)
                metrics.increment("validator_weak_pass_count")
            else:
                strict_pass_faces.append(face)
                metrics.increment("validator_strict_pass_count")

        if validator_results:
            vals = list(validator_results.values())
            metrics.observe_rolling("validator_avg_quality_score",
                sum(r.quality_score for r in vals) / len(vals))
            metrics.observe_rolling("validator_avg_fused_confidence",
                sum(r.fused_confidence for r in vals) / len(vals))

        # ── Legacy validators (secondary safety net, feature-flagged) ───────
        if self._settings.enable_legacy_face_validation:
            legacy_faces = strict_pass_faces + weak_pass_faces
            geometry_accepted: list[DetectedFace] = []
            for face in legacy_faces:
                decision = self._face_validator.validate(
                    face, prepared.bgr.shape, frame_bgr=prepared.bgr, frame_index=frame_index
                )
                if decision.accepted:
                    geometry_accepted.append(face)
                else:
                    label = decision.debug_label or decision.reason or "geometry_rejected"
                    matches.append(
                        self._rejection_match(
                            face, frame_index, prepared, output_scale,
                            label, "yellow", ("DETECTED", "GEOMETRY_REJECTED"),
                        )
                    )
                    metrics.increment("geometry_validation_rejections")

            faces, detector_rejections = self._detection_optimizer.filter_faces(
                geometry_accepted, prepared.bgr.shape
            )
            total_rejected = len(raw_faces) - len(faces)
            self._detection_optimizer.observe_detection_cycle(
                frame_index, len(raw_faces), len(faces), total_rejected
            )

            for rejected_face, reason in detector_rejections:
                debug_label = f"REJECTED: {reason}" if reason and not reason.startswith("REJECTED:") else reason
                matches.append(
                    self._rejection_match(
                        rejected_face, frame_index, prepared, output_scale,
                        debug_label or "detector_filter_rejected", "yellow",
                        ("DETECTED", "FILTERED"),
                    )
                )
        else:
            faces = strict_pass_faces + weak_pass_faces + track_only_faces
            total_rejected = len(raw_faces) - len(faces)

        metrics.increment("total_faces_detected", len(faces))
        metrics.observe("avg_faces_per_frame", len(faces))

        if len(raw_faces) >= self._settings.detector_overload_face_count:
            logger.warning(
                "Detector overload frame_index=%s raw_faces=%s accepted=%s rejected=%s",
                frame_index, len(raw_faces), len(faces), total_rejected,
            )

        tracks = self._track_manager.update_from_detections(
            faces, frame_index,
            frame_shape=prepared.bgr.shape, frame_bgr=prepared.bgr,
        )

        if not faces:
            diagnostics.record("frame", "no_face_detected", frame_index=frame_index)

        for face, track in zip(faces, tracks):
            tier = face_tier.get(id(face), ValidationTier.STRICT_PASS)
            match = self._process_tracked_face(
                face=face, track=track, prepared=prepared,
                frame_index=frame_index, gallery=gallery,
                output_scale=output_scale, from_detection=True,
                validation_tier=tier,
            )
            if match is not None:
                matches.append(match)

        # ── Phase 2A.1: Record detection metrics ─────────────────────────────────
        if self._detection_metrics:
            # Calculate tracker survival time (average visibility_age of active tracks)
            active_tracks = self._track_manager.active_tracks()
            avg_survival_time = 0.0
            if active_tracks:
                avg_survival_time = sum(t.visibility_age for t in active_tracks) / len(active_tracks)

            # Count weak detection promotions (from weak_pass to strict_pass via temporal)
            weak_promotions = 0
            for track in active_tracks:
                if hasattr(track, 'confirmation_hits') and track.confirmation_hits >= 3:
                    weak_promotions += 1

            # Record metrics
            self._detection_metrics.record_detection(
                faces=faces,
                frame_shape=prepared.bgr.shape,
                detection_latency_ms=detection_latency_ms,
                validator_rejections=total_rejected,
                weak_promotions=weak_promotions,
                false_positives=len([m for m in matches if m.reason in ["below_adaptive_threshold", "no_match_above_threshold"]]),
                tracker_survival_time=avg_survival_time,
            )

        self._finalize_frame(frame_index)
        self._export_tracking_metrics()
        return matches

    def _stage_tracking_path(
        self,
        prepared,
        frame_index: int,
        gallery: list[tuple[int, np.ndarray]],
        output_scale: float,
    ) -> list[FrameMatch]:
        self._detection_optimizer.observe_tracking_cycle()
        metrics.increment("frames_skipped")
        diagnostics.record("tracking", "detector_skipped_reused_tracks", frame_index=frame_index)

        matches: list[FrameMatch] = []
        tracks = self._track_manager.propagate(frame_index)
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
        for track in self._track_manager.consume_removed_tracks():
            self._global_identity_memory.archive_lost_track(track, frame_index)
        self._global_identity_memory.prune(frame_index)

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

        # ── TRACK_ONLY: tracking only, no embedding/matching ────────────
        if validation_tier == ValidationTier.TRACK_ONLY:
            metrics.increment("validator_embedding_skips")
            return FrameMatch(
                frame_index=frame_index, person_id=None, confidence=None,
                threshold=self._settings.match_confidence_threshold,
                reason="track_only", track_id=track.numeric_track_id if track else None,
                face=_scale_face_for_output(face, output_scale),
                trace=_trace(face, "yellow", ("DETECTED", "TRACK_ONLY"),
                    "track_only", detector_confidence=face.det_score,
                    frame_shape=prepared.bgr.shape, output_scale=output_scale,
                    validation_tier=validation_tier.value),
            )

        # ── WEAK_PASS: track, defer embedding, enforce retry limit ──────
        if validation_tier == ValidationTier.WEAK_PASS:
            if track is not None:
                tid = track.numeric_track_id
                attempts = self._weak_pass_attempts.get(tid, 0) + 1
                self._weak_pass_attempts[tid] = attempts
                if attempts > self._settings.validator_weak_pass_retry_limit:
                    metrics.increment("validator_embedding_skips")
                    return FrameMatch(
                        frame_index=frame_index, person_id=None, confidence=None,
                        threshold=self._settings.match_confidence_threshold,
                        reason="weak_pass_exhausted", track_id=tid,
                        face=_scale_face_for_output(face, output_scale),
                        trace=_trace(face, "yellow", ("DETECTED", "WEAK_PASS_EXHAUSTED"),
                            "weak_pass_exhausted", detector_confidence=face.det_score,
                            frame_shape=prepared.bgr.shape, output_scale=output_scale,
                            validation_tier=validation_tier.value),
                    )
            metrics.increment("validator_embedding_skips")
            return FrameMatch(
                frame_index=frame_index, person_id=None, confidence=None,
                threshold=self._settings.match_confidence_threshold,
                reason="weak_pass_deferred", track_id=track.numeric_track_id if track else None,
                face=_scale_face_for_output(face, output_scale),
                trace=_trace(face, "yellow", ("DETECTED", "WEAK_PASS"),
                    "weak_pass_deferred", detector_confidence=face.det_score,
                    frame_shape=prepared.bgr.shape, output_scale=output_scale,
                    validation_tier=validation_tier.value),
            )

        # ── STRICT_PASS: full pipeline (unchanged below) ────────────────
        if from_detection and self._settings.enable_legacy_quality_checks:
            crop_check = self._crop_validator.validate(prepared.bgr, face)
            if not crop_check.accepted:
                label = f"REJECTED: {crop_check.reason}" if crop_check.reason else "REJECTED: bad_crop"
                return self._rejection_match(
                    face,
                    frame_index,
                    prepared,
                    output_scale,
                    label,
                    "yellow",
                    ("DETECTED", "CROP_REJECTED"),
                )

        if self._settings.enable_legacy_quality_checks:
            quality = self._quality_assessor.assess(prepared.bgr, face)
            if not quality.accepted:
                return self._quality_rejection(face, frame_index, prepared, output_scale, quality, width, height)
        else:
            from ecoface_lite.ai_engine.face_quality import FaceQualityResult
            quality = FaceQualityResult(True, blur_score=50.0, quality_score=0.7)

        metrics.increment("accepted_faces")
        threshold = self._confidence_policy.threshold_for(prepared.diagnostics, quality)

        if track is None:
            track = self._track_manager.candidate_track(face, frame_index)

        track_quality = None
        if track is not None:
            track_quality = self._track_manager.update_track_quality(track, face, prepared.bgr, frame_index)

        need_embedding = track is None or self._track_manager.should_compute_embedding(
            track, frame_index, face, quality=track_quality
        )
        if (
            track is not None
            and not need_embedding
            and track.identity is not None
            and track.stable_match_count >= self._tracking_cfg.min_stable_matches
        ):
            return self._match_from_track_identity(
                face, track, prepared, frame_index, gallery, output_scale, from_detection, quality
            )

        emb: np.ndarray | None = None
        if track is not None and not need_embedding and track.last_embedding is not None:
            emb = track.last_embedding
            metrics.increment("embedding_cache_hits")
            metrics.increment("reused_embeddings")
        elif need_embedding:
            with metrics.timer("embedding_generation_duration"):
                emb = self._embedder.embed_face(prepared.bgr, face)
            metrics.increment("embeddings_generated")
            if track is not None:
                track.touch_embedding(emb, frame_index)
        else:
            with metrics.timer("embedding_generation_duration"):
                emb = self._embedder.embed_face(prepared.bgr, face)
            metrics.increment("embeddings_generated")

        if emb is None:
            return None

        fusion_weight = (
            self._identity_confidence.embedding_suppression_weight(
                track,
                blur_score=quality.blur_score,
                quality_weight=track_quality.overall_score if track_quality is not None else quality.quality_score,
            )
            if track is not None
            else (track_quality.overall_score if track_quality is not None else quality.quality_score)
        )
        pose_bucket = None
        if face.landmarks is not None:
            pose_bucket = classify_pose_bucket(face.landmarks, face.bbox)
            if track is not None:
                get_temporal_identity(track).record_pose_blur(pose_bucket, quality.blur_score)

        if track is not None and track.visibility_age <= 2 and track.recovery_count == 0:
            self._track_reassociator.try_recover_identity(track, emb, frame_index)

        with metrics.timer("matching_duration"):
            if track is not None:
                decision_match = self._identity_matcher.match_track(
                    track,
                    emb,
                    gallery,
                    threshold,
                    quality_weight=fusion_weight,
                    frame_index=frame_index,
                    pose_bucket=pose_bucket,
                    blur_score=quality.blur_score,
                )
                m = None
                id_conf = None
                if decision_match is not None and decision_match.person_id is not None:
                    id_conf = self._identity_confidence.evaluate(
                        track,
                        decision_match.confidence,
                        threshold,
                        quality=track_quality,
                        person_id=decision_match.person_id,
                    )
                    accepted_stages = {"verified", "soft_verified", "locked", "weak_candidate"}
                    if decision_match.stage in accepted_stages:
                        if (
                            decision_match.confidence >= threshold
                            or decision_match.soft_match
                            or (id_conf is not None and id_conf.soft_accept)
                        ):
                            conf = max(decision_match.confidence, id_conf.temporal_confidence)
                            m = MatchResult(person_id=decision_match.person_id, confidence=conf)
                            if decision_match.stage == "weak_candidate":
                                metrics.increment("identity_recovery_weak_accept")
            else:
                m = self._matcher.best_match(emb, gallery, threshold)

        if m is None:
            metrics.increment("failed_matches")
            metrics.increment("red_box_count")
            diagnostics.record("matching", "no_match_above_threshold", frame_index=frame_index, threshold=threshold)
            return FrameMatch(
                frame_index=frame_index,
                person_id=None,
                confidence=None,
                threshold=threshold,
                reason="no_match_above_threshold",
                track_id=track.numeric_track_id if track else None,
                face=_scale_face_for_output(face, output_scale),
                trace=_trace(
                    face,
                    "red",
                    ("TRACKED" if not from_detection else "DETECTED", "FILTERED", "EMBEDDED", "MATCHED_FAILED"),
                    "no_match_above_threshold",
                    detector_confidence=face.det_score,
                    blur_score=quality.blur_score,
                    frame_shape=prepared.bgr.shape,
                    output_scale=output_scale,
                ),
            )

        metrics.increment("successful_matches")
        metrics.observe("confidence_score", m.confidence)
        temporal_accept = False
        if track is not None:
            id_snap = self._identity_confidence.evaluate(
                track, m.confidence, threshold, quality=track_quality, person_id=m.person_id
            )
            temporal_accept = id_snap.soft_accept
            metrics.observe("identity_stabilization_latency", float(track.visibility_age))
        decision = self._confidence_policy.decide(m.confidence, prepared.diagnostics, quality)
        if not decision.accepted and not temporal_accept:
            metrics.increment("rejected_due_to_low_confidence")
            metrics.increment("red_box_count")
            return FrameMatch(
                frame_index=frame_index,
                person_id=m.person_id,
                confidence=m.confidence,
                threshold=decision.adjusted_threshold,
                reason="below_adaptive_threshold",
                track_id=track.numeric_track_id if track else None,
                face=_scale_face_for_output(face, output_scale),
                trace=_trace(
                    face,
                    "red",
                    ("DETECTED", "FILTERED", "EMBEDDED", "MATCHED", "VALIDATION_REJECTED"),
                    "below_adaptive_threshold",
                    detector_confidence=face.det_score,
                    blur_score=quality.blur_score,
                    frame_shape=prepared.bgr.shape,
                    output_scale=output_scale,
                ),
            )

        if track is None:
            tracks = self._track_manager.update_from_detections(
                [face], frame_index, frame_shape=prepared.bgr.shape
            )
            track = tracks[0] if tracks else None
        if track is not None:
            track.touch_embedding(emb, frame_index)
        recognition = (
            self._recognition_session.observe_track(track, m.person_id, m.confidence)
            if track is not None
            else self._recognition_session.observe(face, frame_index, m.person_id, m.confidence)
        )

        unstable = recognition.stable_match_count < self._tracking_cfg.min_stable_matches or (
            recognition.smoothed_confidence < self._settings.temporal_min_average_confidence
        )
        overlay_state = "green"
        if unstable and recognition.confirmations > 0:
            overlay_state = "yellow"
            reason = "unstable"
        elif not recognition.stable:
            overlay_state = "red"

        with metrics.timer("event_validation_duration"):
            event = self._event_validator.evaluate(recognition, frame_index)

        if event.should_emit:
            metrics.increment("detection_events_validated")
            metrics.increment("green_box_count")
            overlay_state = "green"
        elif overlay_state != "yellow":
            metrics.increment("red_box_count")
            if event.reason == "cooldown":
                metrics.increment("cooldown_suppressions")

        return FrameMatch(
            frame_index=frame_index,
            person_id=recognition.person_id if recognition.stable else m.person_id,
            confidence=recognition.smoothed_confidence or recognition.confidence,
            threshold=decision.adjusted_threshold,
            stable=recognition.stable,
            should_alert=event.should_emit,
            track_id=recognition.track_id,
            reason=event.reason if not event.should_emit else "accepted",
            face=_scale_face_for_output(face, output_scale),
            trace=_trace(
                face,
                overlay_state,
                ("DETECTED", "TRACKED", "RECOGNIZED", "VALIDATED") if event.should_emit else ("DETECTED", "TRACKED", "RECOGNIZED"),
                event.reason if not event.should_emit else None,
                detector_confidence=face.det_score,
                blur_score=quality.blur_score,
                frame_shape=prepared.bgr.shape,
                output_scale=output_scale,
            ),
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

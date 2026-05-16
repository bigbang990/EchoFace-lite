"""High-level pipeline orchestration (video frame → optional alert).

Kept separate from FastAPI so the same pipeline can run from CLI, workers, or tests.
"""

from __future__ import annotations

import numpy as np
from ecoface_lite.ai_engine.confidence import ConfidencePolicy
from ecoface_lite.ai_engine.detection_optimizer import DetectionOptimizer
from ecoface_lite.ai_engine.diagnostics import diagnostics
from ecoface_lite.ai_engine.detector import BoundingBox, FaceDetector
from ecoface_lite.ai_engine.embedder import FaceEmbedder
from ecoface_lite.ai_engine.event_validator import EventValidator
from ecoface_lite.ai_engine.face_quality import FaceQualityAssessor
from ecoface_lite.ai_engine.geometry import bbox_iou, compute_face_geometry, scale_face_to_original
from ecoface_lite.ai_engine.matcher import FaceMatcher, MatchResult
from ecoface_lite.ai_engine.pipeline_types import FaceDebugTrace, FrameMatch
from ecoface_lite.ai_engine.preprocessing import FramePreprocessor
from ecoface_lite.ai_engine.recognition_session import RecognitionSession
from ecoface_lite.core.config import Settings
from ecoface_lite.core.logging import get_logger
from ecoface_lite.core.metrics import metrics

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
        self._detector = detector
        self._embedder = embedder
        self._matcher = matcher
        self._preprocessor = preprocessor or FramePreprocessor(settings)
        self._quality_assessor = quality_assessor or FaceQualityAssessor(settings)
        self._confidence_policy = confidence_policy or ConfidencePolicy(settings)
        self._recognition_session = recognition_session or RecognitionSession(settings)
        self._event_validator = event_validator or EventValidator(settings)
        self._detection_optimizer = DetectionOptimizer(settings)
        self._embedding_cache: dict[int, tuple[np.ndarray, tuple[float, float, float, float], int]] = {}

    def enroll_reference_embedding(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Pick the highest-confidence face and return its embedding (for gallery enrollment)."""
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
        """Run robust recognition and return stable alert candidates."""
        metrics.increment("total_frames_processed")
        with metrics.timer("total_frame_processing_duration"):
            return self._process_frame_observed(frame_bgr, frame_index, gallery)

    def _cached_embedding(self, track_id: int | None, face) -> np.ndarray | None:
        if track_id is None:
            return None
        cached = self._embedding_cache.get(track_id)
        if cached is None:
            return None
        emb, bbox, _frame_index = cached
        previous = BoundingBox(*bbox)
        overlap = bbox_iou(face.bbox, previous)
        if overlap < 0.45:
            metrics.increment("embedding_cache_invalidations")
            return None
        return emb

    def _process_frame_observed(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        gallery: list[tuple[int, np.ndarray]],
    ) -> list[FrameMatch]:
        matches: list[FrameMatch] = []
        with metrics.timer("preprocessing_duration"):
            prepared = self._preprocessor.process(frame_bgr)
        output_scale = prepared.bgr.shape[1] / max(frame_bgr.shape[1], 1)
        if not self._detection_optimizer.should_detect(frame_index):
            self._detection_optimizer.observe_tracking_cycle()
            metrics.increment("frames_skipped")
            diagnostics.record("tracking", "detector_skipped_reused_tracks", frame_index=frame_index)
            return matches
        detection_frame, scale = self._detection_optimizer.prepare_for_detection(prepared.bgr)
        with metrics.timer("face_detection_duration"):
            raw_faces = self._detector.detect(detection_frame)
        raw_faces = self._detection_optimizer.scale_faces(raw_faces, scale)
        faces, detector_rejections = self._detection_optimizer.filter_faces(raw_faces, prepared.bgr.shape)
        self._detection_optimizer.observe_detection_cycle(frame_index, len(raw_faces), len(faces), len(detector_rejections))
        metrics.increment("total_faces_detected", len(faces))
        metrics.observe("avg_faces_per_frame", len(faces))
        for rejected_face, reason in detector_rejections:
            metrics.increment("rejected_faces")
            metrics.increment("yellow_box_count")
            width, height = _face_size(rejected_face)
            metrics.observe("avg_rejected_face_area", width * height)
            diagnostics.record(
                "detector_filter",
                reason,
                frame_index=frame_index,
                metadata={"det_score": rejected_face.det_score, "face_width": width, "face_height": height},
            )
            matches.append(
                FrameMatch(
                    frame_index=frame_index,
                    person_id=None,
                    confidence=None,
                    threshold=self._settings.match_confidence_threshold,
                    reason=reason,
                    face=_scale_face_for_output(rejected_face, output_scale),
                    trace=_trace(
                        rejected_face,
                        "yellow",
                        ("DETECTED", "FILTERED"),
                        reason,
                        detector_confidence=rejected_face.det_score,
                        frame_shape=prepared.bgr.shape,
                        output_scale=output_scale,
                    ),
                )
            )
        if len(raw_faces) >= self._settings.detector_overload_face_count:
            logger.warning(
                "Detector overload frame_index=%s raw_faces=%s accepted=%s rejected=%s",
                frame_index,
                len(raw_faces),
                len(faces),
                len(detector_rejections),
            )
        if not faces:
            diagnostics.record("frame", "no_face_detected", frame_index=frame_index)
        for face in faces:
            width, height = _face_size(face)
            metrics.observe("avg_detected_face_area", width * height)
            quality = self._quality_assessor.assess(prepared.bgr, face)
            metrics.observe("face_quality_score", quality.blur_score)
            if not quality.accepted:
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
                logger.info(
                    "Frame rejected frame_index=%s reason=%s blur_score=%.3f det_score=%.3f",
                    frame_index,
                    quality.reason,
                    quality.blur_score,
                    face.det_score,
                )
                matches.append(
                    FrameMatch(
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
                )
                continue
            metrics.increment("accepted_faces")
            threshold = self._confidence_policy.threshold_for(prepared.diagnostics, quality)
            candidate_track_id = self._recognition_session.candidate_track_id(face, frame_index)
            emb = self._cached_embedding(candidate_track_id, face)
            if emb is None:
                with metrics.timer("embedding_generation_duration"):
                    emb = self._embedder.embed_face(prepared.bgr, face)
                metrics.increment("embeddings_generated")
            else:
                metrics.increment("embedding_cache_hits")
            with metrics.timer("matching_duration"):
                m = self._matcher.best_match(emb, gallery, threshold)
            if m is None:
                metrics.increment("failed_matches")
                metrics.increment("red_box_count")
                diagnostics.record("matching", "no_match_above_threshold", frame_index=frame_index, threshold=threshold)
                matches.append(
                    FrameMatch(
                        frame_index=frame_index,
                        person_id=None,
                        confidence=None,
                        threshold=threshold,
                        reason="no_match_above_threshold",
                        face=_scale_face_for_output(face, output_scale),
                        trace=_trace(
                            face,
                            "red",
                            ("DETECTED", "FILTERED", "EMBEDDED", "MATCHED_FAILED"),
                            "no_match_above_threshold",
                            detector_confidence=face.det_score,
                            blur_score=quality.blur_score,
                            frame_shape=prepared.bgr.shape,
                            output_scale=output_scale,
                        ),
                    )
                )
                continue
            metrics.increment("successful_matches")
            metrics.observe("confidence_score", m.confidence)
            top_candidates = self._matcher.top_k(emb, gallery, k=3)
            diagnostics.record(
                "matching",
                "top_k_candidates",
                frame_index=frame_index,
                person_id=m.person_id,
                confidence=m.confidence,
                threshold=threshold,
                metadata={"candidates": [(c.person_id, c.confidence) for c in top_candidates]},
            )
            diagnostics.record(
                "matching",
                "candidate_match",
                frame_index=frame_index,
                person_id=m.person_id,
                confidence=m.confidence,
                threshold=threshold,
            )
            decision = self._confidence_policy.decide(m.confidence, prepared.diagnostics, quality)
            if not decision.accepted:
                metrics.increment("rejected_due_to_low_confidence")
                metrics.increment("red_box_count")
                diagnostics.record(
                    "matching",
                    "below_adaptive_threshold",
                    frame_index=frame_index,
                    person_id=m.person_id,
                    confidence=m.confidence,
                    threshold=decision.adjusted_threshold,
                )
                logger.info(
                    "Match confidence below threshold frame_index=%s person_id=%s confidence=%.3f threshold=%.3f",
                    frame_index,
                    m.person_id,
                    m.confidence,
                    decision.adjusted_threshold,
                )
                matches.append(
                    FrameMatch(
                        frame_index=frame_index,
                        person_id=m.person_id,
                        confidence=m.confidence,
                        threshold=decision.adjusted_threshold,
                        reason="below_adaptive_threshold",
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
                )
                continue
            recognition = self._recognition_session.observe(face, frame_index, m.person_id, m.confidence)
            self._embedding_cache[recognition.track_id] = (
                emb,
                (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2),
                frame_index,
            )
            with metrics.timer("event_validation_duration"):
                event = self._event_validator.evaluate(recognition, frame_index)
            if event.should_emit:
                metrics.increment("detection_events_validated")
                metrics.increment("green_box_count")
                diagnostics.record(
                    "event",
                    "validated",
                    frame_index=frame_index,
                    person_id=recognition.person_id,
                    confidence=recognition.confidence,
                    threshold=decision.adjusted_threshold,
                )
                logger.info(
                    "Event validated frame_index=%s person_id=%s confidence=%.3f track_id=%s",
                    frame_index,
                    recognition.person_id,
                    recognition.confidence,
                    recognition.track_id,
                )
            else:
                metrics.increment("red_box_count")
                if event.reason == "cooldown":
                    metrics.increment("cooldown_suppressions")
                diagnostics.record(
                    "event",
                    event.reason,
                    frame_index=frame_index,
                    person_id=recognition.person_id,
                    confidence=recognition.confidence,
                    threshold=decision.adjusted_threshold,
                )
            matches.append(
                FrameMatch(
                    frame_index=frame_index,
                    person_id=recognition.person_id if recognition.stable else m.person_id,
                    confidence=recognition.confidence,
                    threshold=decision.adjusted_threshold,
                    stable=recognition.stable,
                    should_alert=event.should_emit,
                    track_id=recognition.track_id,
                    reason=event.reason,
                    face=_scale_face_for_output(face, output_scale),
                    trace=_trace(
                        face,
                        "green" if event.should_emit else "red",
                        ("DETECTED", "FILTERED", "EMBEDDED", "MATCHED", "VALIDATED")
                        if event.should_emit
                        else ("DETECTED", "FILTERED", "EMBEDDED", "MATCHED", "EVENT_NOT_EMITTED"),
                        event.reason,
                        detector_confidence=face.det_score,
                        blur_score=quality.blur_score,
                        frame_shape=prepared.bgr.shape,
                        output_scale=output_scale,
                    ),
                )
            )
        return matches

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
    )


def _scale_face_for_output(face, output_scale: float):
    if output_scale == 1.0:
        return face
    return scale_face_to_original(face, output_scale)

"""High-precision geometric and landmark validation for detector proposals.



The detector is a noisy proposal generator; this module scores candidates and

only hard-rejects clear impossibilities. Uncertain faces pass to temporal stages.

"""



from __future__ import annotations



from dataclasses import dataclass, field



import cv2

import numpy as np



from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace, FaceLandmarks

from ecoface_lite.ai_engine.pose_estimator import (
    classify_pose_bucket, PoseBucket
)

from ecoface_lite.core.config import Settings

from ecoface_lite.core.metrics import metrics





@dataclass
class FaceValidationResult:

    accepted: bool

    reason: str | None = None

    debug_label: str | None = None

    validation_score: float = 0.0

    rejection_reasons: tuple[str, ...] = ()

    metadata: dict = field(default_factory=dict)





def validate_face_candidate(

    face_bbox: BoundingBox,

    landmarks: FaceLandmarks | None,

    frame_shape: tuple[int, ...],

    *,

    det_score: float,

    settings: Settings,

    frame_bgr: np.ndarray | None = None,

) -> FaceValidationResult:

    """Score and validate a detector proposal; soft-fail uncertain candidates."""

    height, width = int(frame_shape[0]), int(frame_shape[1])

    frame_area = max(1, width * height)

    x1, y1, x2, y2 = face_bbox.x1, face_bbox.y1, face_bbox.x2, face_bbox.y2

    box_w = max(1e-6, x2 - x1)

    box_h = max(1e-6, y2 - y1)

    aspect = box_w / box_h

    rejection_reasons: list[str] = []

    metadata: dict = {}



    geometry_score = 1.0

    if aspect < settings.detector_min_aspect_ratio or aspect > settings.detector_max_aspect_ratio:

        metrics.increment("rejected_aspect_ratio")

        return FaceValidationResult(

            False,

            "aspect_ratio",

            "REJECTED: aspect_ratio",

            validation_score=0.0,

            rejection_reasons=("aspect_ratio",),

        )



    face_area = box_w * box_h

    min_area = frame_area * settings.detector_min_face_area_ratio

    if face_area < min_area * 0.5:

        metrics.increment("rejected_tiny_fragment")

        return FaceValidationResult(

            False,

            "tiny_fragment",

            "REJECTED: tiny_fragment",

            validation_score=0.0,

            rejection_reasons=("tiny_fragment",),

        )

    if face_area < min_area:

        geometry_score *= 0.6

        rejection_reasons.append("small_area")



    area_ratio = face_area / frame_area

    required = _required_confidence(area_ratio, settings)

    effective_score = det_score

    if effective_score < required * 0.85:

        metrics.increment("rejected_weak_detector_score")

        return FaceValidationResult(

            False,

            "weak_detector_score",

            "REJECTED: low_confidence",

            validation_score=effective_score,

            rejection_reasons=("weak_detector_score",),

        )

    if effective_score < required:

        geometry_score *= 0.75

        rejection_reasons.append("weak_detector_score")



    edge_touch = _edge_touch_fraction(x1, y1, x2, y2, width, height)

    metadata["edge_touch_ratio"] = edge_touch

    if edge_touch >= settings.detector_edge_touch_ratio and effective_score < settings.detector_edge_high_confidence:

        if edge_touch >= settings.detector_edge_touch_ratio * 1.25:

            metrics.increment("rejected_edge_clipping")

            return FaceValidationResult(

                False,

                "edge_clipping",

                "REJECTED: edge_clipping",

                validation_score=geometry_score * 0.5,

                rejection_reasons=("edge_clipping",),

                metadata=metadata,

            )

        geometry_score *= 0.7

        rejection_reasons.append("edge_clipping")



    landmark_score = 0.0

    pose_yaw = 0.0

    pose_pitch = 0.0

    if landmarks is None:
        metrics.increment("rejected_low_landmarks")
        return FaceValidationResult(
            False,
            "low_landmarks",
            "REJECTED: low_landmarks",
            validation_score=geometry_score * 0.4,
            rejection_reasons=("low_landmarks",),
            metadata=metadata,
        )

    else:

        landmark_result = _validate_landmarks(landmarks, face_bbox, settings)

        if not landmark_result.accepted and landmark_result.reason in {

            "asymmetric_landmarks",

            "jawline_fragment",

        }:

            return landmark_result

        landmark_score = landmark_result.validation_score

        pose_yaw, pose_pitch = _estimate_pose(landmarks, face_bbox)

        pose_bucket = classify_pose_bucket(landmarks, face_bbox)

        metadata["pose_yaw_ratio"] = pose_yaw

        metadata["pose_pitch_ratio"] = pose_pitch

        if pose_yaw > settings.proposal_max_yaw_ratio:

            geometry_score *= 0.65

            rejection_reasons.append("high_yaw")

        if pose_pitch > settings.proposal_max_pitch_ratio:

            geometry_score *= 0.65

            rejection_reasons.append("high_pitch")



    blur_score = 0.0

    brightness_score = 0.0

    if frame_bgr is not None:

        xi1, yi1 = max(0, int(x1)), max(0, int(y1))

        xi2, yi2 = min(width, int(x2)), min(height, int(y2))

        crop = frame_bgr[yi1:yi2, xi1:xi2]

        if crop.size > 0:

            blur_score = _blur_score(crop)

            brightness_score = _brightness_score(crop)

            metadata["blur_score"] = blur_score

            metadata["brightness_score"] = brightness_score



    blur_component = min(1.0, blur_score / max(settings.face_quality_min_blur_score * 2.0, 1.0))

    illum_component = min(1.0, brightness_score / max(settings.face_quality_min_brightness * 2.0, 1.0))

    pose_component = max(0.0, 1.0 - min(1.0, pose_yaw / max(settings.proposal_max_yaw_ratio, 1e-6)))



    validation_score = float(

        max(

            0.0,

            min(

                1.0,

                (settings.proposal_geometry_weight * geometry_score * landmark_score if landmarks is not None else geometry_score * 0.7)

                + (settings.proposal_blur_weight * blur_component)

                + (settings.proposal_illumination_weight * illum_component)

                + (settings.proposal_pose_weight * pose_component),

            ),

        )

    )

    metadata["validation_score"] = validation_score

    metrics.observe("proposal_validation_score", validation_score)



    _profile_buckets = {
        PoseBucket.LEFT_PROFILE,
        PoseBucket.RIGHT_PROFILE,
    }
    effective_cutoff = (
        settings.proposal_min_validation_score
        - settings.validator_profile_cutoff_reduction
        if pose_bucket in _profile_buckets
        else settings.proposal_min_validation_score
    )
    if validation_score < effective_cutoff \
            and len(rejection_reasons) >= 2:

        metrics.increment("rejected_low_validation_score")

        return FaceValidationResult(

            False,

            "low_validation_score",

            "REJECTED: low_validation_score",

            validation_score=validation_score,

            rejection_reasons=tuple(rejection_reasons),

            metadata=metadata,

        )



    return FaceValidationResult(

        True,

        validation_score=validation_score,

        rejection_reasons=tuple(rejection_reasons),

        metadata=metadata,

    )





class FaceCandidateValidator:

    """Stateful wrapper that validates ``DetectedFace`` instances."""



    def __init__(self, settings: Settings) -> None:

        self._settings = settings

        self._proposal_history: list[tuple[tuple[float, float, float, float], int]] = []



    def validate(

        self,

        face: DetectedFace,

        frame_shape: tuple[int, ...],

        frame_bgr: np.ndarray | None = None,

        frame_index: int = 0,

    ) -> FaceValidationResult:

        score = face.temporal_score if face.temporal_score is not None else face.det_score

        result = validate_face_candidate(

            face.bbox,

            face.landmarks,

            frame_shape,

            det_score=score,

            settings=self._settings,

            frame_bgr=frame_bgr,

        )

        if result.accepted:
            self._record_proposal(face, frame_index)
            agreement = self._temporal_agreement(face.bbox)
            metrics.observe("proposal_temporal_agreement", agreement)
            merged_meta = dict(result.metadata)
            merged_meta["temporal_agreement"] = agreement
            return FaceValidationResult(
                accepted=result.accepted,
                reason=result.reason,
                debug_label=result.debug_label,
                validation_score=result.validation_score,
                rejection_reasons=result.rejection_reasons,
                metadata=merged_meta,
            )
        return result



    def validate_detected(self, face: DetectedFace, frame_shape: tuple[int, ...]) -> FaceValidationResult:

        return self.validate(face, frame_shape)



    def _record_proposal(self, face: DetectedFace, frame_index: int) -> None:

        bbox = (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2)

        self._proposal_history.append((bbox, frame_index))

        if len(self._proposal_history) > 32:

            self._proposal_history.pop(0)



    def _temporal_agreement(self, bbox: BoundingBox) -> float:

        if not self._proposal_history:

            return 1.0

        cx = (bbox.x1 + bbox.x2) / 2.0

        cy = (bbox.y1 + bbox.y2) / 2.0

        matches = 0

        for prev_bbox, _ in self._proposal_history[-8:]:

            pcx = (prev_bbox[0] + prev_bbox[2]) / 2.0

            pcy = (prev_bbox[1] + prev_bbox[3]) / 2.0

            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5

            if dist < self._settings.temporal_max_track_distance:

                matches += 1

        return matches / min(8, len(self._proposal_history))





def _required_confidence(area_ratio: float, settings: Settings) -> float:

    if area_ratio < settings.detector_small_face_area_ratio:

        return settings.detector_small_face_threshold

    if area_ratio < settings.detector_medium_face_area_ratio:

        return settings.detector_medium_quality_threshold

    return max(settings.detector_min_score, settings.detector_high_quality_threshold)





def _edge_touch_fraction(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> float:

    touches = 0

    margin = 1.0

    if x1 <= margin:

        touches += 1

    if y1 <= margin:

        touches += 1

    if x2 >= width - margin:

        touches += 1

    if y2 >= height - margin:

        touches += 1

    return touches / 4.0





def _validate_landmarks(landmarks: FaceLandmarks, bbox: BoundingBox, settings: Settings) -> FaceValidationResult:

    pts = landmarks.points

    if pts.shape[0] < 5:

        metrics.increment("rejected_low_landmarks")

        return FaceValidationResult(False, "low_landmarks", "REJECTED: low_landmarks", validation_score=0.0)



    left_eye, right_eye, nose = pts[0], pts[1], pts[2]

    eye_y_delta = abs(left_eye[1] - right_eye[1])

    inter_eye = max(1e-6, float(np.linalg.norm(right_eye - left_eye)))

    symmetry_score = max(0.0, 1.0 - (eye_y_delta / inter_eye) / 0.45)

    if eye_y_delta / inter_eye > 0.55:

        metrics.increment("rejected_asymmetric_landmarks")

        return FaceValidationResult(

            False,

            "asymmetric_landmarks",

            "REJECTED: asymmetric_landmarks",

            validation_score=symmetry_score * 0.5,

        )



    box_w = max(1e-6, bbox.x2 - bbox.x1)

    box_h = max(1e-6, bbox.y2 - bbox.y1)

    in_box = True

    for pt in (left_eye, right_eye, nose):

        pt_x, pt_y = float(pt[0]), float(pt[1])

        if not (bbox.x1 - box_w * 0.15 <= pt_x <= bbox.x2 + box_w * 0.15):

            in_box = False

        if not (bbox.y1 - box_h * 0.15 <= pt_y <= bbox.y2 + box_h * 0.15):

            in_box = False

    if not in_box:

        metrics.increment("rejected_low_landmarks")

        return FaceValidationResult(False, "low_landmarks", "REJECTED: low_landmarks", validation_score=0.4)



    nose_x = float(nose[0])

    center_x = (bbox.x1 + bbox.x2) / 2.0

    asymmetry = abs(nose_x - center_x) / (box_w / 2.0)

    if asymmetry > settings.detector_max_landmark_asymmetry * 1.15:

        metrics.increment("rejected_asymmetric_landmarks")

        return FaceValidationResult(

            False,

            "partial_side_face",

            "REJECTED: asymmetric_landmarks",

            validation_score=max(0.0, 1.0 - asymmetry),

        )



    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0

    if eye_mid_y > bbox.y1 + box_h * 0.72:

        metrics.increment("rejected_jawline_fragment")

        return FaceValidationResult(

            False,

            "jawline_fragment",

            "REJECTED: jawline_fragment",

            validation_score=0.3,

        )



    landmark_score = float(max(0.0, min(1.0, symmetry_score * (1.0 - asymmetry * 0.5))))

    return FaceValidationResult(True, validation_score=landmark_score)





def _estimate_pose(landmarks: FaceLandmarks, bbox: BoundingBox) -> tuple[float, float]:

    pts = landmarks.points

    left_eye, right_eye, nose = pts[0], pts[1], pts[2]

    box_w = max(1e-6, bbox.x2 - bbox.x1)

    box_h = max(1e-6, bbox.y2 - bbox.y1)

    nose_x = float(nose[0])

    center_x = (bbox.x1 + bbox.x2) / 2.0

    yaw = abs(nose_x - center_x) / (box_w / 2.0)

    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0

    pitch = abs(float(nose[1]) - eye_mid_y) / box_h

    return yaw, pitch





def _blur_score(face_bgr: np.ndarray) -> float:

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())





def _brightness_score(face_bgr: np.ndarray) -> float:

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    return float(np.mean(gray))



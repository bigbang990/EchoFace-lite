"""Unified face validation — the airlock between detection and tracking.

Executes AFTER detection, BEFORE tracking/embedding/matching.

Tiers: STRICT_PASS (track+embed+match), WEAK_PASS (track+delayed embed),
TRACK_ONLY (track only), REJECT (discard+snapshot). WEAK_PASS lifecycle:
weak_pass_max_age_frames/weak_pass_retry_limit prevent zombie tracks.
TRACK_ONLY reassociation (Phase 2B): temporal promotion with same track_id.
Snapshot I/O must be buffered/async in production to avoid FPS murder.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace, FaceLandmarks
from ecoface_lite.ai_engine.geometry import clip_bbox_to_frame
class ValidationTier(str, Enum):
    STRICT_PASS = "STRICT_PASS"
    WEAK_PASS = "WEAK_PASS"
    TRACK_ONLY = "TRACK_ONLY"
    REJECT = "REJECT"

    def __ge__(self, other: ValidationTier) -> bool:
        _o = {ValidationTier.STRICT_PASS: 3, ValidationTier.WEAK_PASS: 2, ValidationTier.TRACK_ONLY: 1, ValidationTier.REJECT: 0}
        return _o[self] >= _o[other]

    def __gt__(self, other: ValidationTier) -> bool:
        _o = {ValidationTier.STRICT_PASS: 3, ValidationTier.WEAK_PASS: 2, ValidationTier.TRACK_ONLY: 1, ValidationTier.REJECT: 0}
        return _o[self] > _o[other]
class RejectionReason(str, Enum):
    FACE_CLIPPED = "FACE_CLIPPED"
    INVALID_ASPECT_RATIO = "INVALID_ASPECT_RATIO"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    FACE_TOO_LARGE = "FACE_TOO_LARGE"
    FACE_TOO_BLURRY = "FACE_TOO_BLURRY"
    FACE_TOO_DARK = "FACE_TOO_DARK"
    FACE_TOO_BRIGHT = "FACE_TOO_BRIGHT"
    LANDMARK_MISMATCH = "LANDMARK_MISMATCH"
    LOW_QUALITY_SCORE = "LOW_QUALITY_SCORE"
_REJECTION_PRIORITY: dict[RejectionReason, int] = {
    RejectionReason.FACE_CLIPPED: 0,
    RejectionReason.INVALID_ASPECT_RATIO: 1,
    RejectionReason.FACE_TOO_SMALL: 2,
    RejectionReason.FACE_TOO_LARGE: 3,
    RejectionReason.FACE_TOO_BLURRY: 4,
    RejectionReason.FACE_TOO_DARK: 5,
    RejectionReason.FACE_TOO_BRIGHT: 6,
    RejectionReason.LANDMARK_MISMATCH: 7,
    RejectionReason.LOW_QUALITY_SCORE: 8,
}
def _sort_reasons(reasons: list[RejectionReason]) -> list[RejectionReason]:
    return sorted(reasons, key=lambda r: _REJECTION_PRIORITY.get(r, 99))
@dataclass(frozen=True)
class ValidationResult:
    tier: ValidationTier
    quality_score: float
    fused_confidence: float
    primary_reason: str | None
    rejection_reasons: tuple[str, ...]
    metrics: dict[str, float]
    validated_bbox: tuple[float, float, float, float]
    landmark_score: float
    blur_score: float
    brightness_score: float
    geometry_score: float
    size_score: float
@dataclass
class TrackStateQuality:
    temporal_stability: float = 1.0
    bbox_consistency: float = 1.0
    identity_consistency: float = 1.0
    average_quality: float = 1.0
    motion_score: float = 0.0
    occlusion_ratio: float = 0.0
    visibility_ratio: float = 1.0
@dataclass(frozen=True)
class SceneContext:
    low_light: bool = False
    high_motion: bool = False
    crowd_density: float = 0.0
    indoor: bool = True
# -- geometry, size, blur, brightness ----------------------------------------
def validate_geometry(
    bbox: BoundingBox, frame_shape: tuple[int, ...],
    min_aspect: float, max_aspect: float, edge_margin_ratio: float,
) -> tuple[float, list[RejectionReason], dict[str, float]]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    bw = max(1e-6, bbox.x2 - bbox.x1)
    bh = max(1e-6, bbox.y2 - bbox.y1)
    aspect = bw / bh
    reasons: list[RejectionReason] = []
    metrics: dict[str, float] = {"aspect_ratio": aspect}
    if aspect < min_aspect or aspect > max_aspect:
        reasons.append(RejectionReason.INVALID_ASPECT_RATIO)
        return 0.0, reasons, metrics
    em = edge_margin_ratio * min(w, h)
    touches = sum([bbox.x1 <= em, bbox.y1 <= em, bbox.x2 >= w - em, bbox.y2 >= h - em])
    ef = touches / 4.0
    metrics["edge_touch_fraction"] = ef
    if ef >= 0.5:
        reasons.append(RejectionReason.FACE_CLIPPED)
        return max(0.0, 1.0 - ef), reasons, metrics
    dev = abs(aspect - 0.85) / 0.85
    return max(0.0, 1.0 - dev * 0.5), reasons, metrics


def validate_size(
    bbox: BoundingBox, frame_shape: tuple[int, ...],
    min_area_ratio: float, max_area_ratio: float,
) -> tuple[float, list[RejectionReason], dict[str, float]]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    fa = max(1, w * h)
    bw = max(1e-6, bbox.x2 - bbox.x1)
    bh = max(1e-6, bbox.y2 - bbox.y1)
    face_area = bw * bh
    ar = face_area / fa
    reasons: list[RejectionReason] = []
    metrics = {"face_area": face_area, "frame_area": float(fa), "area_ratio": ar}
    if ar > max_area_ratio:
        reasons.append(RejectionReason.FACE_TOO_LARGE)
        return 0.0, reasons, metrics
    if ar < min_area_ratio * 0.5:
        reasons.append(RejectionReason.FACE_TOO_SMALL)
        return 0.0, reasons, metrics
    if ar < min_area_ratio:
        reasons.append(RejectionReason.FACE_TOO_SMALL)
        return ar / max(min_area_ratio, 1e-6), reasons, metrics
    return min(1.0, ar / (min_area_ratio * 3.0)), reasons, metrics


def compute_blur_score(face_crop_bgr: np.ndarray) -> float:
    if face_crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def validate_blur(blur_raw: float, min_blur_var: float) -> tuple[float, list[RejectionReason]]:
    reasons: list[RejectionReason] = []
    if blur_raw < min_blur_var * 0.35:
        reasons.append(RejectionReason.FACE_TOO_BLURRY)
        return 0.0, reasons
    if blur_raw < min_blur_var:
        reasons.append(RejectionReason.FACE_TOO_BLURRY)
        return blur_raw / max(min_blur_var, 1.0), reasons
    return min(1.0, blur_raw / (min_blur_var * 2.0)), reasons


def compute_brightness_score(face_crop_bgr: np.ndarray) -> float:
    if face_crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def validate_brightness(
    brightness_raw: float, min_brightness: float, max_brightness: float,
) -> tuple[float, list[RejectionReason]]:
    reasons: list[RejectionReason] = []
    if brightness_raw < min_brightness * 0.5:
        reasons.append(RejectionReason.FACE_TOO_DARK)
        return 0.0, reasons
    if brightness_raw < min_brightness:
        reasons.append(RejectionReason.FACE_TOO_DARK)
        return brightness_raw / max(min_brightness, 1.0), reasons
    if brightness_raw > max_brightness:
        reasons.append(RejectionReason.FACE_TOO_BRIGHT)
        over = (brightness_raw - max_brightness) / max(255.0 - max_brightness, 1.0)
        return max(0.0, 1.0 - over), reasons
    dev = abs(brightness_raw - 128.0) / 128.0
    return max(0.0, 1.0 - dev * 0.5), reasons


def validate_detector_confidence(
    det_score: float, min_confidence: float,
) -> tuple[bool, list[RejectionReason]]:
    reasons: list[RejectionReason] = []
    if det_score < min_confidence:
        reasons.append(RejectionReason.LOW_QUALITY_SCORE)
        return False, reasons
    return True, reasons

# -- landmarks (soft-score, never sole hard-reject) --------------------------
def validate_landmark_consistency(
    landmarks: FaceLandmarks | None, bbox: BoundingBox, max_asymmetry: float,
) -> tuple[float, list[RejectionReason], dict[str, float]]:
    reasons: list[RejectionReason] = []
    metrics: dict[str, float] = {}
    if landmarks is None:
        reasons.append(RejectionReason.LANDMARK_MISMATCH)
        return 0.3, reasons, metrics
    pts = landmarks.points
    if pts.shape[0] < 5:
        reasons.append(RejectionReason.LANDMARK_MISMATCH)
        return 0.3, reasons, metrics
    le, re, nose = pts[0], pts[1], pts[2]
    bw = max(1e-6, bbox.x2 - bbox.x1)
    bh = max(1e-6, bbox.y2 - bbox.y1)
    # eye symmetry
    eye_dy = abs(float(le[1]) - float(re[1]))
    inter_eye = max(1e-6, float(np.linalg.norm(re - le)))
    esr = eye_dy / inter_eye
    metrics["eye_symmetry_ratio"] = esr
    sym_score = 0.0 if esr > 0.55 else max(0.0, 1.0 - esr / 0.45)
    if esr > 0.55:
        reasons.append(RejectionReason.LANDMARK_MISMATCH)
    # nose centering
    nx = float(nose[0])
    cx = (bbox.x1 + bbox.x2) / 2.0
    asym = abs(nx - cx) / (bw / 2.0)
    metrics["nose_asymmetry"] = asym
    asym_score = 0.0 if asym > max_asymmetry * 1.15 else max(0.0, 1.0 - asym / max(max_asymmetry, 1e-6))
    if asym > max_asymmetry * 1.15:
        reasons.append(RejectionReason.LANDMARK_MISMATCH)
    # containment
    in_box = True
    for pt in (le, re, nose):
        px, py = float(pt[0]), float(pt[1])
        if not (bbox.x1 - bw * 0.15 <= px <= bbox.x2 + bw * 0.15):
            in_box = False
        if not (bbox.y1 - bh * 0.15 <= py <= bbox.y2 + bh * 0.15):
            in_box = False
    cont_score = 0.4 if not in_box else 1.0
    if not in_box:
        reasons.append(RejectionReason.LANDMARK_MISMATCH)
    # eye vertical position (jawline fragment guard)
    eye_my = (float(le[1]) + float(re[1])) / 2.0
    pos_score = 0.3 if eye_my > bbox.y1 + bh * 0.72 else 1.0
    if eye_my > bbox.y1 + bh * 0.72:
        reasons.append(RejectionReason.LANDMARK_MISMATCH)
    lm_score = float(max(0.0, min(1.0, sym_score * 0.35 + asym_score * 0.30 + cont_score * 0.20 + pos_score * 0.15)))
    return lm_score, reasons, metrics

# -- fused confidence (weighted additive) ------------------------------------
def compute_fused_confidence(
    det_score: float, geometry_score: float, landmark_score: float,
    temporal_consistency: float = 1.0, track_stability: float = 1.0,
    w_det: float = 0.35, w_geo: float = 0.20, w_lm: float = 0.15,
    w_tmp: float = 0.15, w_stb: float = 0.15,
) -> float:
    return float(max(0.0, min(1.0,
        w_det * det_score + w_geo * geometry_score + w_lm * landmark_score
        + w_tmp * temporal_consistency + w_stb * track_stability)))

# -- unified quality score ---------------------------------------------------
def compute_unified_quality(
    blur_quality: float, brightness_quality: float, geometry_score: float,
    landmark_score: float, size_score: float,
    w_blur: float = 0.30, w_bright: float = 0.15, w_geo: float = 0.25,
    w_lm: float = 0.15, w_size: float = 0.15,
) -> float:
    return float(max(0.0, min(1.0,
        w_blur * blur_quality + w_bright * brightness_quality
        + w_geo * geometry_score + w_lm * landmark_score + w_size * size_score)))

# -- tier assignment ---------------------------------------------------------
def _assign_tier(
    quality_score: float, fused_confidence: float,
    rejection_reasons: list[RejectionReason], landmark_score: float,
    quality_cutoff: float, strict_cutoff: float,
) -> ValidationTier:
    hard = {RejectionReason.FACE_CLIPPED, RejectionReason.INVALID_ASPECT_RATIO}
    if any(r in hard for r in rejection_reasons):
        return ValidationTier.REJECT
    if quality_score < quality_cutoff and len(rejection_reasons) >= 2:
        return ValidationTier.REJECT
    if (quality_score < quality_cutoff and len(rejection_reasons) == 1
            and RejectionReason.LANDMARK_MISMATCH in rejection_reasons):
        return ValidationTier.TRACK_ONLY
    if quality_score < quality_cutoff:
        return ValidationTier.REJECT
    if quality_score >= strict_cutoff and len(rejection_reasons) == 0:
        return ValidationTier.STRICT_PASS
    if quality_score >= quality_cutoff:
        return ValidationTier.WEAK_PASS
    return ValidationTier.TRACK_ONLY

# -- snapshot storage --------------------------------------------------------
class RejectedSnapshotStorage(ABC):
    @abstractmethod
    def save_rejected(
        self, face_crop: np.ndarray, reason: str,
        frame_index: int, track_id: int | None = None,
    ) -> None: ...
    @abstractmethod
    def cleanup(self, max_count: int) -> None: ...


class LocalRejectedSnapshotStorage(RejectedSnapshotStorage):
    def __init__(self, base_dir: Path, enabled: bool = True) -> None:
        self._base_dir = base_dir
        self._enabled = enabled
        if self._enabled:
            self._base_dir.mkdir(parents=True, exist_ok=True)

    def save_rejected(
        self, face_crop: np.ndarray, reason: str,
        frame_index: int, track_id: int | None = None,
    ) -> None:
        if not self._enabled or face_crop.size == 0:
            return
        ts = int(time.time() * 1000)
        tid = f"_tid{track_id}" if track_id is not None else ""
        safe = reason.replace(" ", "_").replace("/", "_")
        path = self._base_dir / f"rej_{frame_index:06d}_{ts}{tid}_{safe}.jpg"
        cv2.imwrite(str(path), face_crop)

    def cleanup(self, max_count: int) -> None:
        if not self._enabled:
            return
        files = sorted(self._base_dir.glob("rej_*.jpg"), key=lambda p: p.stat().st_mtime)
        for old in files[:-max_count] if len(files) > max_count else []:
            try:
                old.unlink()
            except OSError:
                pass

# -- FaceValidator -----------------------------------------------------------
class FaceValidator:
    def __init__(self, settings: Any) -> None:
        self._s = settings
        w_sum = (self._s.validator_blur_weight + self._s.validator_brightness_weight
                 + self._s.validator_geometry_weight + self._s.validator_landmark_weight
                 + self._s.validator_size_weight)
        if abs(w_sum - 1.0) >= 0.01:
            raise ValueError(f"Validator quality weights must sum to 1.0, got {w_sum:.4f}")

    def validate(
        self, face: DetectedFace, frame_bgr: np.ndarray,
        frame_shape: tuple[int, ...], frame_index: int = 0,
        scene: SceneContext | None = None,
        track_quality: TrackStateQuality | None = None,
        min_det_confidence: float | None = None,
        strict_cutoff: float | None = None,
    ) -> ValidationResult:
        det_score = face.temporal_score if face.temporal_score is not None else face.det_score
        clipped = clip_bbox_to_frame(face.bbox, frame_shape)
        xi1 = max(0, int(clipped.x1))
        yi1 = max(0, int(clipped.y1))
        xi2 = min(int(frame_shape[1]), int(clipped.x2))
        yi2 = min(int(frame_shape[0]), int(clipped.y2))
        crop = frame_bgr[yi1:yi2, xi1:xi2] if yi2 > yi1 and xi2 > xi1 else np.array([])

        geo_score, geo_reasons, geo_metrics = validate_geometry(
            face.bbox, frame_shape,
            self._s.validator_min_aspect_ratio,
            self._s.validator_max_aspect_ratio,
            self._s.validator_edge_margin_ratio,
        )
        size_score, size_reasons, size_metrics = validate_size(
            face.bbox, frame_shape,
            self._s.validator_min_face_area_ratio,
            self._s.validator_max_face_area_ratio,
        )
        
        det_threshold = min_det_confidence if min_det_confidence is not None else self._s.validator_min_detector_confidence
        det_ok, det_reasons = validate_detector_confidence(
            det_score, det_threshold,
        )
        blur_raw = compute_blur_score(crop)
        blur_quality, blur_reasons = validate_blur(blur_raw, self._s.validator_min_blur_var)
        bright_raw = compute_brightness_score(crop)
        bright_quality, bright_reasons = validate_brightness(
            bright_raw, self._s.validator_min_brightness, self._s.validator_max_brightness,
        )
        lm_score, lm_reasons, lm_metrics = validate_landmark_consistency(
            face.landmarks, face.bbox, self._s.validator_max_landmark_asymmetry,
        )

        all_reasons: list[RejectionReason] = []
        all_reasons.extend(geo_reasons)
        all_reasons.extend(size_reasons)
        all_reasons.extend(blur_reasons)
        all_reasons.extend(bright_reasons)
        all_reasons.extend(lm_reasons)
        if not det_ok:
            all_reasons.extend(det_reasons)

        quality_score = compute_unified_quality(
            blur_quality, bright_quality, geo_score, lm_score, size_score,
            w_blur=self._s.validator_blur_weight,
            w_bright=self._s.validator_brightness_weight,
            w_geo=self._s.validator_geometry_weight,
            w_lm=self._s.validator_landmark_weight,
            w_size=self._s.validator_size_weight,
        )

        tq = track_quality or TrackStateQuality()
        fused = compute_fused_confidence(
            det_score, geo_score, lm_score,
            temporal_consistency=tq.temporal_stability,
            track_stability=tq.bbox_consistency,
        )

        effective_strict_cutoff = strict_cutoff if strict_cutoff is not None else self._s.validator_strict_cutoff
        tier = _assign_tier(
            quality_score, fused, all_reasons, lm_score,
            self._s.validator_quality_cutoff, effective_strict_cutoff,
        )

        sorted_reasons = _sort_reasons(all_reasons)
        primary = sorted_reasons[0].value if sorted_reasons else None

        combined_metrics: dict[str, float] = {}
        combined_metrics.update(geo_metrics)
        combined_metrics.update(size_metrics)
        combined_metrics.update(lm_metrics)
        combined_metrics["blur_raw"] = blur_raw
        combined_metrics["brightness_raw"] = bright_raw
        combined_metrics["blur_quality"] = blur_quality
        combined_metrics["brightness_quality"] = bright_quality
        combined_metrics["detector_confidence"] = det_score

        return ValidationResult(
            tier=tier,
            quality_score=quality_score,
            fused_confidence=fused,
            primary_reason=primary,
            rejection_reasons=tuple(r.value for r in sorted_reasons),
            metrics=combined_metrics,
            validated_bbox=(face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2),
            landmark_score=lm_score,
            blur_score=blur_raw,
            brightness_score=bright_raw,
            geometry_score=geo_score,
            size_score=size_score,
        )

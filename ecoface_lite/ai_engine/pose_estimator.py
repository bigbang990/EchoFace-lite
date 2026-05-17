"""Lightweight pose estimation from 5-point landmarks for pose-aware matching."""

from __future__ import annotations

from enum import Enum

import numpy as np

from ecoface_lite.ai_engine.detector import BoundingBox, FaceLandmarks


class PoseBucket(str, Enum):
    FRONTAL = "frontal"
    LEFT_PROFILE = "left_profile"
    RIGHT_PROFILE = "right_profile"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


def estimate_pose_ratios(landmarks: FaceLandmarks, bbox: BoundingBox) -> tuple[float, float, float]:
    """Return (yaw_ratio, pitch_ratio, roll_proxy) in normalized face coordinates."""
    pts = landmarks.points
    left_eye, right_eye, nose = pts[0], pts[1], pts[2]
    box_w = max(1e-6, bbox.x2 - bbox.x1)
    box_h = max(1e-6, bbox.y2 - bbox.y1)
    nose_x = float(nose[0])
    center_x = (bbox.x1 + bbox.x2) / 2.0
    yaw = abs(nose_x - center_x) / (box_w / 2.0)
    eye_mid_y = (float(left_eye[1]) + float(right_eye[1])) / 2.0
    pitch = abs(float(nose[1]) - eye_mid_y) / box_h
    eye_dx = float(right_eye[0] - left_eye[0])
    eye_dy = float(right_eye[1] - left_eye[1])
    roll_proxy = abs(eye_dy) / max(abs(eye_dx), 1e-6)
    return yaw, pitch, roll_proxy


def classify_pose_bucket(
    landmarks: FaceLandmarks | None,
    bbox: BoundingBox,
    *,
    frontal_yaw_max: float = 0.22,
    profile_yaw_min: float = 0.38,
) -> PoseBucket:
    if landmarks is None:
        return PoseBucket.UNKNOWN
    pts = landmarks.points
    left_eye, right_eye, nose = pts[0], pts[1], pts[2]
    yaw, pitch, _ = estimate_pose_ratios(landmarks, bbox)
    if pitch > 0.55 or yaw > 0.85:
        return PoseBucket.PARTIAL
    nose_x = float(nose[0])
    center_x = (bbox.x1 + bbox.x2) / 2.0
    if yaw <= frontal_yaw_max:
        return PoseBucket.FRONTAL
    if nose_x < center_x:
        return PoseBucket.LEFT_PROFILE
    if nose_x > center_x:
        return PoseBucket.RIGHT_PROFILE
    return PoseBucket.PARTIAL


def pose_bucket_key(bucket: PoseBucket) -> str:
    return bucket.value

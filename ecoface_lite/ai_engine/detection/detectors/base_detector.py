"""Base detector interface for Phase 2A Enterprise Detection Upgrade."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace, FaceLandmarks


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration for detector execution."""
    scale: float = 1.0
    det_size: tuple[int, int] = (640, 640)
    batch_size: int = 1
    enable_enhancement: bool = True


class BaseDetector(ABC):
    """Abstract base class for face detectors."""

    @abstractmethod
    def detect(
        self,
        frame_bgr: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> list[DetectedFace]:
        """Detect faces in a BGR image (OpenCV convention).

        Args:
            frame_bgr: Input frame in BGR format
            config: Optional detection configuration (scale, det_size, etc.)

        Returns:
            List of detected faces with bounding boxes, confidence, landmarks
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the detection model."""

    @abstractmethod
    def get_input_size(self) -> tuple[int, int]:
        """Return the default input size for the detector."""

    def prepare_frame(
        self,
        frame_bgr: np.ndarray,
        scale: float = 1.0,
        target_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Prepare frame for detection by resizing if needed.

        Args:
            frame_bgr: Input frame in BGR format
            scale: Scale factor to apply (1.0 = no scaling)
            target_size: Optional target size (width, height)

        Returns:
            Tuple of (prepared_frame, actual_scale_factor)
        """
        if scale == 1.0 and target_size is None:
            return frame_bgr, 1.0

        h, w = frame_bgr.shape[:2]

        if target_size is not None:
            target_w, target_h = target_size
            scale_factor = target_w / w
            resized = self._resize_frame(frame_bgr, target_w, target_h)
            return resized, scale_factor

        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = self._resize_frame(frame_bgr, new_w, new_h)
            return resized, scale

        return frame_bgr, 1.0

    def _resize_frame(
        self,
        frame_bgr: np.ndarray,
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        """Resize frame to target dimensions using OpenCV."""
        import cv2
        return cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def scale_faces_to_original(
        self,
        faces: list[DetectedFace],
        scale_factor: float,
    ) -> list[DetectedFace]:
        """Scale face bounding boxes back to original frame coordinates.

        Args:
            faces: List of detected faces
            scale_factor: Scale factor applied during detection

        Returns:
            List of faces with scaled bounding boxes
        """
        if scale_factor == 1.0:
            return faces

        scaled_faces = []
        for face in faces:
            scaled_bbox = BoundingBox(
                x1=face.bbox.x1 / scale_factor,
                y1=face.bbox.y1 / scale_factor,
                x2=face.bbox.x2 / scale_factor,
                y2=face.bbox.y2 / scale_factor,
            )
            scaled_face = DetectedFace(
                bbox=scaled_bbox,
                det_score=face.det_score,
                aligned_face=face.aligned_face,
                embedding=face.embedding,
                landmarks=face.landmarks,
                temporal_score=face.temporal_score,
            )
            scaled_faces.append(scaled_face)

        return scaled_faces

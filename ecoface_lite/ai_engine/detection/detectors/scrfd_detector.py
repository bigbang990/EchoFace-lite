"""SCRFD detector implementation using base detector interface."""

from __future__ import annotations

from typing import Any

import numpy as np

from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace, FaceLandmarks
from ecoface_lite.ai_engine.detection.detectors.base_detector import BaseDetector, DetectionConfig
from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


class SCRFDDetector(BaseDetector):
    """SCRFD detector wrapping InsightFace with base detector interface."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
        face_app: Any | None = None,
        default_det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self._model_name = model_name
        self._ctx_id = ctx_id
        self._injected_app = face_app
        self._default_det_size = default_det_size
        self._app: Any = None

    def _ensure_app(self) -> None:
        """Lazy-load InsightFace app."""
        if self._app is not None:
            return
        if self._injected_app is not None:
            self._app = self._injected_app
            return
        from insightface.app import FaceAnalysis

        logger.info("Loading InsightFace model=%s ctx_id=%s", self._model_name, self._ctx_id)
        self._app = FaceAnalysis(name=self._model_name, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=self._ctx_id, det_size=self._default_det_size)

    def detect(
        self,
        frame_bgr: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> list[DetectedFace]:
        """Detect faces using SCRFD model.

        Args:
            frame_bgr: Input frame in BGR format
            config: Optional detection configuration

        Returns:
            List of detected faces
        """
        self._ensure_app()

        if config is None:
            config = DetectionConfig(scale=1.0, det_size=self._default_det_size)

        # Prepare frame with scale if needed
        prepared_frame, scale_factor = self.prepare_frame(
            frame_bgr,
            scale=config.scale,
            target_size=config.det_size if config.det_size != self._default_det_size else None,
        )

        # Run detection
        faces = self._app.get(prepared_frame)

        # Convert to DetectedFace format
        detected_faces = []
        for f in faces:
            bbox = f.bbox.astype(float)
            x1, y1, x2, y2 = bbox.tolist()
            det_score = float(getattr(f, "det_score", 0.0))

            aligned = getattr(f, "normed_face", None)
            emb = getattr(f, "embedding", None)
            emb_arr = np.asarray(emb, dtype=np.float32) if emb is not None else None

            kps = getattr(f, "kps", None)
            landmarks = None
            if kps is not None:
                landmarks = FaceLandmarks(
                    points=np.asarray(kps, dtype=np.float32).reshape(-1, 2)[:5]
                )

            detected_face = DetectedFace(
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                det_score=det_score,
                aligned_face=aligned,
                embedding=emb_arr,
                landmarks=landmarks,
            )
            detected_faces.append(detected_face)

        # Scale back to original coordinates if frame was scaled
        if scale_factor != 1.0:
            detected_faces = self.scale_faces_to_original(detected_faces, scale_factor)

        return detected_faces

    def get_model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def get_input_size(self) -> tuple[int, int]:
        """Return the default input size."""
        return self._default_det_size

    def set_det_size(self, det_size: tuple[int, int]) -> None:
        """Set a new detection size (requires re-preparation).

        Args:
            det_size: New detection size (width, height)
        """
        self._default_det_size = det_size
        if self._app is not None:
            self._app.prepare(ctx_id=self._ctx_id, det_size=det_size)

"""Face detection — wraps InsightFace / OpenCV behind a narrow interface.

Why an interface:
- Swaps implementation (different model, GPU batching) without touching API layer.
- Unit tests can inject a fake detector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class FaceLandmarks:
    """Five-point landmarks from InsightFace: left_eye, right_eye, nose, left_mouth, right_mouth."""

    points: np.ndarray  # shape (5, 2), float32


@dataclass(frozen=True)
class DetectedFace:
    bbox: BoundingBox
    det_score: float
    aligned_face: np.ndarray | None = None  # optional chip for embedding
    embedding: np.ndarray | None = None  # when detector+recognition run together (InsightFace)
    landmarks: FaceLandmarks | None = None
    temporal_score: float | None = None  # blended score after temporal agreement


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        """Detect faces in a BGR image (OpenCV convention)."""


class InsightFaceDetector(FaceDetector):
    """Concrete detector using InsightFace (lazy-loaded or injected shared app)."""

    def __init__(self, model_name: str, ctx_id: int = -1, face_app: Any | None = None, det_size: tuple[int, int] = (320, 320)) -> None:
        self._model_name = model_name
        self._ctx_id = ctx_id
        self._injected_app = face_app
        self._det_size = det_size
        self._app: Any = None

    def _ensure_app(self) -> None:
        if self._app is not None:
            return
        if self._injected_app is not None:
            self._app = self._injected_app
            return
        from insightface.app import FaceAnalysis

        logger.info("Loading InsightFace model=%s ctx_id=%s", self._model_name, self._ctx_id)
        self._app = FaceAnalysis(name=self._model_name, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        self._ensure_app()
        faces = self._app.get(frame_bgr)
        out: list[DetectedFace] = []
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
                landmarks = FaceLandmarks(points=np.asarray(kps, dtype=np.float32).reshape(-1, 2)[:5])
            out.append(
                DetectedFace(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    det_score=det_score,
                    aligned_face=aligned,
                    embedding=emb_arr,
                    landmarks=landmarks,
                )
            )
        return out

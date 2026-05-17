"""Embedding generation — maps detected faces to vectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


class FaceEmbedder(ABC):
    @abstractmethod
    def embed_face(self, frame_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Return L2-normalized embedding vector (float32) if possible."""


class InsightFaceEmbedder(FaceEmbedder):
    """Reuses FaceAnalysis recognition model for embeddings."""

    def __init__(self, model_name: str, ctx_id: int = -1, face_app: Any | None = None) -> None:
        self._model_name = model_name
        self._ctx_id = ctx_id
        self._injected_app = face_app
        self._app: Any = None

    def _ensure_app(self) -> None:
        if self._app is not None:
            return
        if self._injected_app is not None:
            self._app = self._injected_app
            return
        from insightface.app import FaceAnalysis

        logger.info("Loading InsightFace (embed) model=%s", self._model_name)
        self._app = FaceAnalysis(name=self._model_name, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=self._ctx_id, det_size=(320, 320))

    def embed_face(self, frame_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        if face.embedding is not None:
            emb = face.embedding.astype(np.float32).ravel()
        else:
            self._ensure_app()
            faces = self._app.get(frame_bgr)
            if not faces:
                raise ValueError("No faces returned by InsightFace for embedding")
            best = max(faces, key=lambda ff: float(getattr(ff, "det_score", 0.0)))
            emb = np.asarray(best.embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb

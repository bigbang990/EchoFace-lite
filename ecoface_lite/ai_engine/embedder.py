"""Embedding generation — maps detected faces to vectors."""

from __future__ import annotations

import types
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.core.logging import get_logger
from ecoface_lite.core.metrics import metrics
from ecoface_lite.core.platform_bootstrap import detect_platform

logger = get_logger(__name__)

# Cached platform dict — same object as bootstrap.py's PLATFORM (detect_platform is memoised).
_PLATFORM = detect_platform()


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

        providers = _PLATFORM["providers"]
        logger.info(
            "Loading InsightFace (embed) model=%s providers=%s",
            self._model_name,
            providers,
        )
        self._app = FaceAnalysis(name=self._model_name, providers=providers)
        self._app.prepare(ctx_id=self._ctx_id, det_size=(320, 320))

    def embed_face(self, frame_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        if face.embedding is not None:
            emb = face.embedding.astype(np.float32).ravel()
        else:
            self._ensure_app()
            rec_model = self._app.models.get("recognition")

            if rec_model is not None and face.landmarks is not None:
                # Fast path: call the ArcFace recognition model directly on the
                # already-detected face. ArcFaceONNX.get(img, face) uses face.kps
                # for norm_crop alignment and returns the embedding without
                # re-running the detection model across the whole frame.
                fake_face = types.SimpleNamespace(kps=face.landmarks.points)
                raw = rec_model.get(frame_bgr, fake_face)
                emb = np.asarray(raw, dtype=np.float32).ravel()
            else:
                # Slow fallback: full-frame re-detection. Fires only when landmarks
                # are absent or the recognition model wasn't loaded. Should be rare
                # in production — the counter lets us confirm this via /observability/metrics.
                metrics.increment("embedder_full_frame_fallback")
                logger.warning(
                    "embedder_full_frame_fallback fired — landmarks=%s rec_model=%s. "
                    "Running full-frame InsightFace detection (slow path ~10s). "
                    "Check that YOLO keypoints are enabled and face_app is injected.",
                    face.landmarks,
                    type(rec_model).__name__ if rec_model is not None else None,
                )
                faces = self._app.get(frame_bgr)
                if not faces:
                    raise ValueError("No faces returned by InsightFace for embedding")
                best = max(faces, key=lambda ff: float(getattr(ff, "det_score", 0.0)))
                best_score = float(getattr(best, "det_score", 0.0))
                if best.embedding is None or best_score < 0.70:
                    raise ValueError(
                        f"Re-detection produced low-quality candidate "
                        f"(det_score={best_score:.3f} < 0.70) — likely a non-face, skipping."
                    )
                emb = np.asarray(best.embedding, dtype=np.float32).ravel()

        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb

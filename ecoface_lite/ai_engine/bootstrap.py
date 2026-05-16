"""Construct AI stack with shared heavy models (single FaceAnalysis instance)."""

from __future__ import annotations

from typing import Any

from ecoface_lite.ai_engine.detector import FaceDetector, InsightFaceDetector
from ecoface_lite.ai_engine.embedder import FaceEmbedder, InsightFaceEmbedder
from ecoface_lite.ai_engine.matcher import FaceMatcher
from ecoface_lite.ai_engine.pipeline import RecognitionPipeline
from ecoface_lite.core.config import Settings, get_settings
from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


def _create_face_analysis(settings: Settings) -> Any:
    from insightface.app import FaceAnalysis

    logger.info(
        "Initializing InsightFace FaceAnalysis model=%s ctx_id=%s",
        settings.insightface_model_name,
        settings.insightface_ctx_id,
    )
    app = FaceAnalysis(name=settings.insightface_model_name, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=settings.insightface_ctx_id, det_size=(640, 640))
    return app


def build_recognition_pipeline(settings: Settings | None = None) -> RecognitionPipeline:
    settings = settings or get_settings()
    face_app = _create_face_analysis(settings)
    detector: FaceDetector = InsightFaceDetector(
        model_name=settings.insightface_model_name,
        ctx_id=settings.insightface_ctx_id,
        face_app=face_app,
    )
    embedder: FaceEmbedder = InsightFaceEmbedder(
        model_name=settings.insightface_model_name,
        ctx_id=settings.insightface_ctx_id,
        face_app=face_app,
    )
    matcher = FaceMatcher()
    return RecognitionPipeline(settings=settings, detector=detector, embedder=embedder, matcher=matcher)


_pipeline_singleton: RecognitionPipeline | None = None


def get_recognition_pipeline() -> RecognitionPipeline:
    """Process-wide singleton so InsightFace weights load once per worker."""
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = build_recognition_pipeline()
    return _pipeline_singleton

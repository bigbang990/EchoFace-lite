"""Construct AI stack with shared heavy models (single FaceAnalysis instance)."""

from __future__ import annotations

from typing import Any

from ecoface_lite.ai_engine.detector import FaceDetector, InsightFaceDetector
from ecoface_lite.ai_engine.embedder import FaceEmbedder, InsightFaceEmbedder
from ecoface_lite.ai_engine.matcher import FaceMatcher
from ecoface_lite.ai_engine.pipeline import RecognitionPipeline
from ecoface_lite.core.config import Settings, get_settings
from ecoface_lite.core.runtime_config import EffectiveRuntimeConfig
from ecoface_lite.core.runtime_state import get_runtime_state
from ecoface_lite.core.logging import get_logger
from ecoface_lite.core.metrics import metrics

logger = get_logger(__name__)


def _create_face_analysis(settings: Settings) -> Any:
    from insightface.app import FaceAnalysis

    logger.info(
        "Initializing InsightFace FaceAnalysis model=%s ctx_id=%s",
        settings.insightface_model_name,
        settings.insightface_ctx_id,
    )
    app = FaceAnalysis(name=settings.insightface_model_name, providers=["CPUExecutionProvider"])
    det_size = (settings.detector_input_width, settings.detector_input_height)
    app.prepare(ctx_id=settings.insightface_ctx_id, det_size=det_size)
    metrics.observe("detector_input_resolution", det_size[0] * det_size[1])
    metrics.observe("detector_resolution", det_size[0] * det_size[1])
    return app


def build_recognition_pipeline(settings: Settings | None = None) -> RecognitionPipeline:
    settings = settings or get_settings()
    # Perform startup validation
    try:
        from ecoface_lite.ai_engine.tracking.tracked_face import TrackedFace
        from ecoface_lite.ai_engine.tracking.track_manager import FaceTrackManager
        from ecoface_lite.ai_engine.pipeline import RecognitionPipeline
        from ecoface_lite.core.runtime_config import EffectiveRuntimeConfig
        
        logger.info("=== PIPELINE IMPORT VALIDATION PASSED ===")
    except Exception as e:
        logger.error("!!! PIPELINE IMPORT VALIDATION FAILED: %s !!!", e)
        raise RuntimeError(f"Startup validation failed: {e}") from e

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
    
    # Compile effective runtime configuration
    runtime_state = get_runtime_state()
    effective_config = EffectiveRuntimeConfig.compile(
        settings=settings,
        overrides=runtime_state.get_overrides(),
        cpu_protection_state=runtime_state.get_cpu_protection_state(),
        backend_type=runtime_state.get_backend_type(),
        experiment_session_id=runtime_state.get_experiment_session_id()
    )
    
    return RecognitionPipeline(
        settings=settings, 
        detector=detector, 
        embedder=embedder, 
        matcher=matcher,
        effective_config=effective_config
    )


_pipeline_singleton: RecognitionPipeline | None = None


def get_recognition_pipeline() -> RecognitionPipeline:
    """Process-wide singleton so InsightFace weights load once per worker."""
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = build_recognition_pipeline()
    return _pipeline_singleton

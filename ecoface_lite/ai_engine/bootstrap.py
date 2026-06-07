"""Construct AI stack with shared heavy models (single FaceAnalysis instance)."""

from __future__ import annotations

from typing import Any

from ecoface_lite.ai_engine.detector import FaceDetector, InsightFaceDetector
from ecoface_lite.ai_engine.embedder import FaceEmbedder, InsightFaceEmbedder
from ecoface_lite.ai_engine.matcher import FaceMatcher
from ecoface_lite.ai_engine.pipeline import RecognitionPipeline
from ecoface_lite.core.config import Settings, get_settings
from ecoface_lite.core.platform_bootstrap import detect_platform
from ecoface_lite.core.runtime_config import EffectiveRuntimeConfig
from ecoface_lite.core.runtime_state import get_runtime_state
from ecoface_lite.core.logging import get_logger
from ecoface_lite.core.metrics import metrics

logger = get_logger(__name__)

PLATFORM = detect_platform()


def _get_providers(settings: Settings) -> list[str]:
    """Select ONNX Runtime execution providers based on ctx_id and availability.

    ctx_id >= 0  → GPU requested.  Check for CUDAExecutionProvider at runtime.
    ctx_id < 0   → CPU requested.  Return CPU only.

    This replaces the previous hardcoded ["CPUExecutionProvider"] which silently
    prevented GPU inference even when INSIGHTFACE_CTX_ID=0 was set.
    """
    if settings.insightface_ctx_id >= 0:
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                logger.info(
                    "GPU requested (ctx_id=%d) — CUDAExecutionProvider available, using CUDA inference",
                    settings.insightface_ctx_id,
                )
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.warning(
                "GPU requested (ctx_id=%d) but CUDAExecutionProvider is NOT in available providers %s. "
                "Falling back to CPU. To enable GPU: pip install onnxruntime-gpu",
                settings.insightface_ctx_id,
                available,
            )
        except ImportError:
            logger.warning(
                "onnxruntime not importable — cannot check GPU providers, falling back to CPU"
            )
        except Exception as e:
            logger.warning("Provider detection failed (%s) — falling back to CPU", e)
    return ["CPUExecutionProvider"]


def _create_face_analysis(settings: Settings) -> Any:
    from insightface.app import FaceAnalysis

    providers = PLATFORM["providers"]
    logger.info(
        "Initializing InsightFace FaceAnalysis model=%s ctx_id=%s providers=%s",
        settings.insightface_model_name,
        PLATFORM["ctx_id"],
        providers,
    )
    app = FaceAnalysis(name=settings.insightface_model_name, providers=providers)
    det_size = PLATFORM["det_size"]
    app.prepare(ctx_id=PLATFORM["ctx_id"], det_size=det_size)
    metrics.observe("detector_input_resolution", det_size[0] * det_size[1])
    metrics.observe("detector_resolution", det_size[0] * det_size[1])
    is_gpu = "CUDAExecutionProvider" in providers
    logger.info(
        "InsightFace ready — backend=%s det_size=%s",
        "GPU/CUDA" if is_gpu else "CPU",
        det_size,
    )
    return app


def build_recognition_pipeline(settings: Settings | None = None) -> RecognitionPipeline:
    settings = settings or get_settings()
    # Apply hardware-detected threshold overrides.
    # PLATFORM is module-level (evaluated once at import).  These two fields are
    # set by PLATFORM so GPU Colab gets 0.45/0.55 and CPU dev gets 0.35/0.40
    # regardless of what the .env file says — the .env values remain the fallback
    # for every other field.
    settings.detection_confidence_threshold = PLATFORM["conf_threshold"]
    settings.validator_strict_cutoff = PLATFORM["validator_cutoff"]
    logger.info(
        "Platform threshold overrides applied: detection_confidence_threshold=%.2f  "
        "validator_strict_cutoff=%.2f  (backend=%s)",
        settings.detection_confidence_threshold,
        settings.validator_strict_cutoff,
        PLATFORM["backend"],
    )
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

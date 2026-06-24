"""Construct AI stack with shared heavy models (single FaceAnalysis instance).

Threshold bug note
------------------
Prior to this fix, build_recognition_pipeline() contained three lines that
silently overrode .env-configured thresholds with platform defaults on every
startup:

    settings.detection_confidence_threshold = PLATFORM["conf_threshold"]  # BUG
    settings.validator_strict_cutoff        = PLATFORM["validator_cutoff"] # BUG
    settings.insightface_ctx_id             = PLATFORM["ctx_id"]           # BUG

This meant local CPU always ran at 0.35/0.40 regardless of .env, making
every local-vs-Colab comparison invalid.  Those lines have been removed.
Thresholds are now owned exclusively by Settings/.env.

ONNX preservation note
----------------------
The "onnx" detector branch is reserved for future paid GPU infra (A100/H100)
where onnxruntime-gpu is available.  It is NOT implemented yet, but the branch
is kept so switching to it only requires DETECTOR_PROVIDER=onnx in .env — no
code changes.  The embedder (ArcFace/InsightFace) is never affected by detector
selection.
"""

from __future__ import annotations

import os
from pathlib import Path
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


def _resolve_detector_provider() -> str:
    """Return the active detector provider name from the environment.

    Reads DETECTOR_PROVIDER env var, normalised to lowercase.
    Falls back to the value platform_bootstrap detected (itself defaulting
    to "scrfd" when the var is unset).  Isolated here so the selection logic
    is testable without building the full pipeline.
    """
    return os.environ.get(
        "DETECTOR_PROVIDER",
        PLATFORM.get("detector_provider", "scrfd"),
    ).lower().strip()


def build_recognition_pipeline(settings: Settings | None = None) -> RecognitionPipeline:
    settings = settings or get_settings()

    # Log the actual .env-sourced threshold values so startup logs confirm
    # they are being respected (no silent platform overrides any more).
    logger.info(
        "Pipeline thresholds from .env/Settings — "
        "detection_confidence_threshold=%.2f  "
        "validator_strict_cutoff=%.2f  "
        "match_confidence_threshold=%.2f  "
        "insightface_ctx_id=%d  "
        "(backend=%s)",
        settings.detection_confidence_threshold,
        settings.validator_strict_cutoff,
        settings.match_confidence_threshold,
        settings.insightface_ctx_id,
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

    provider = _resolve_detector_provider()
    face_app = None

    if provider == "yolo":
        from ecoface_lite.ai_engine.detection.detectors.yolov8_detector import YOLOv8FaceDetector
        weights = (
            Path(__file__).resolve().parent.parent.parent / "weights" / "yolov8n-face.pt"
        )
        if not weights.is_file():
            raise FileNotFoundError(
                f"YOLOv8 weights not found at {weights}. "
                "Download with:\n"
                "  wget -P weights/ "
                "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
            )
        # Build shared InsightFace app first — YOLO needs it for genderage inference.
        face_app = _create_face_analysis(settings)
        detector: FaceDetector = YOLOv8FaceDetector(
            weights_path=weights,
            det_size=PLATFORM["det_size"],
            face_app=face_app,
        )
        logger.info("Detector: YOLOv8-face (PyTorch) weights=%s", weights)

    elif provider == "onnx":
        # Reserved for future paid GPU infra (A100/H100) with onnxruntime-gpu.
        # Not yet implemented — fall back to SCRFD with a clear warning so the
        # operator knows the chosen provider was not honoured.
        logger.warning(
            "DETECTOR_PROVIDER=onnx is reserved for future GPU infra "
            "and is not yet implemented. Falling back to SCRFD (InsightFace). "
            "Switch to onnxruntime-gpu and implement OnnxFaceDetector before "
            "setting this provider in production."
        )
        face_app = _create_face_analysis(settings)
        detector = InsightFaceDetector(
            model_name=settings.insightface_model_name,
            ctx_id=settings.insightface_ctx_id,
            face_app=face_app,
        )
        logger.info("Detector: SCRFD (InsightFace) [onnx fallback]")

    else:
        # Default: SCRFD via InsightFace buffalo_l
        face_app = _create_face_analysis(settings)
        detector = InsightFaceDetector(
            model_name=settings.insightface_model_name,
            ctx_id=settings.insightface_ctx_id,
            face_app=face_app,
        )
        logger.info("Detector: SCRFD (InsightFace)")

    # Embedder is always ArcFace/InsightFace regardless of detector selection.
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
        experiment_session_id=runtime_state.get_experiment_session_id(),
    )

    return RecognitionPipeline(
        settings=settings,
        detector=detector,
        embedder=embedder,
        matcher=matcher,
        effective_config=effective_config,
    )


_pipeline_singleton: RecognitionPipeline | None = None


def get_recognition_pipeline() -> RecognitionPipeline:
    """Process-wide singleton so InsightFace weights load once per worker."""
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = build_recognition_pipeline()
    return _pipeline_singleton


def get_embedder():
    """
    Return the shared FaceEmbedder instance.
    Stateless — safe to call from any request context.
    Do NOT use for operations that require tracker state.
    """
    pipeline = get_recognition_pipeline()
    return pipeline._embedder


def get_detector():
    """
    Return the shared BaseDetector instance.
    Stateless — safe for per-request detection.
    Do NOT use for operations that require tracker state.
    """
    pipeline = get_recognition_pipeline()
    return pipeline._detector


def get_matcher():
    """
    Return the shared FaceMatcher instance.
    Stateless — safe for per-request matching.
    Do NOT use for operations that require tracker state.
    """
    pipeline = get_recognition_pipeline()
    return pipeline._matcher

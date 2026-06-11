"""Platform detection for EchoFace Lite.

Detects at startup whether CUDA is available and returns a plain dict of
hardware-appropriate parameters for the detector and governance layers.

Public API
----------
    detect_platform() -> dict

The result is cached after the first call so it is safe to call multiple
times — subsequent calls return the same dict without re-running detection.

Keys returned
-------------
    backend               str   "CPU" or "GPU"
    ctx_id                int   -1 (CPU) / 0 (GPU)
    providers             list  ONNX Runtime execution providers in priority order
    det_size              tuple (320, 320) CPU / (640, 640) GPU
    det_interval          int   6 CPU / 3 GPU
    conf_threshold        float 0.35 CPU / 0.45 GPU
    validator_cutoff      float 0.40 CPU / 0.55 GPU
    detector_budget_ms    int   5000 CPU / 150 GPU
    max_track_survival_ms int   6000 CPU / 3000 GPU  — hard ceiling on LOST/COARSE TTL
    interval_ceiling      int   12 CPU / 8 GPU
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PLATFORM_CACHE: dict | None = None


def detect_platform() -> dict:
    """Return hardware-appropriate detector parameters.

    Detection:
    torch.cuda.is_available() — confirms CUDA device is present.

    Returns
    -------
    dict
        Plain dict with the keys documented in the module docstring.
        Same object is returned on every call after the first.
    """
    global _PLATFORM_CACHE
    if _PLATFORM_CACHE is not None:
        return _PLATFORM_CACHE

    # ── Step 1: CUDA device present? ─────────────────────────────────────────
    cuda_available = False
    gpu_name = "unknown GPU"
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            cuda_available = True
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "GPU device 0"
    except ImportError:
        logger.debug("torch not installed — skipping CUDA device check, assuming CPU")

    # ── Build platform dict ───────────────────────────────────────────────────
    if cuda_available:
        _PLATFORM_CACHE = {
            "backend":            "GPU",
            "ctx_id":             0,
            "providers":          ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "det_size":           (640, 640),
            "det_interval":       3,
            "conf_threshold":     0.45,
            "validator_cutoff":   0.55,
            "detector_budget_ms":    150,
            "max_track_survival_ms": 3000,
            "interval_ceiling":      8,
            "detector_provider":     "scrfd",
        }
        logger.info("EchoFace backend=GPU/CUDA — production mode, %s", gpu_name)
    else:
        _PLATFORM_CACHE = {
            "backend":            "CPU",
            "ctx_id":             -1,
            "providers":          ["CPUExecutionProvider"],
            "det_size":           (320, 320),
            "det_interval":       6,
            "conf_threshold":     0.35,
            "validator_cutoff":   0.40,
            "detector_budget_ms":    5000,
            "max_track_survival_ms": 6000,
            "interval_ceiling":      12,
            "detector_provider":     "scrfd",
        }
        logger.info("EchoFace backend=CPU — development mode, not real-time")

    return _PLATFORM_CACHE

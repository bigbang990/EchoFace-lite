from __future__ import annotations

import platform
from typing import Any

from fastapi import APIRouter

from ecoface_lite import __version__

router = APIRouter(tags=["health"])


def _detect_gpu() -> dict[str, Any]:
    result: dict[str, Any] = {"gpu": False, "device": "CPU"}
    try:
        import torch
        if torch.cuda.is_available():
            result["gpu"] = True
            result["device"] = torch.cuda.get_device_name(0) or "CUDA GPU"
    except Exception:
        pass
    return result


@router.get("/health")
async def health() -> dict[str, Any]:
    gpu_info = _detect_gpu()
    return {
        "status": "ok",
        "backend": platform.node() or "ecoface-server",
        "version": __version__,
        "gpu": gpu_info["gpu"],
        "device": gpu_info["device"],
    }

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query

from ecoface_lite.ai_engine.diagnostics import diagnostics
from ecoface_lite.core.config import get_settings
from ecoface_lite.core.metrics import metrics

router = APIRouter(prefix="/observability", tags=["observability"])


@router.get("/metrics")
async def get_metrics() -> dict[str, object]:
    return metrics.export()


@router.post("/metrics/reset")
async def reset_metrics() -> dict[str, str]:
    metrics.reset()
    return {"status": "reset"}


@router.get("/diagnostics")
async def get_diagnostics() -> dict[str, object]:
    return diagnostics.snapshot()


@router.get("/logs/recent")
async def get_recent_logs(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, object]:
    settings = get_settings()
    log_path = settings.resolved_log_dir() / "ecoface_lite.log"
    if not log_path.is_file():
        return {"path": str(log_path), "lines": []}
    lines = _tail_lines(log_path, limit)
    return {"path": str(log_path), "lines": lines}


def _tail_lines(path: Path, limit: int) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f.readlines()[-limit:]]

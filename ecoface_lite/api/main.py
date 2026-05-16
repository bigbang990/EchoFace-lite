"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ecoface_lite import __version__
from ecoface_lite.api.routers import detections, health, live_test, observability, persons, processing
from ecoface_lite.core.config import get_settings
from ecoface_lite.core.logging import get_logger, setup_logging
from ecoface_lite.db.session import init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    data_root = (
        settings.project_root / settings.data_dir
        if not settings.data_dir.is_absolute()
        else settings.data_dir
    )
    data_root.mkdir(parents=True, exist_ok=True)
    settings.resolved_uploads_dir().mkdir(parents=True, exist_ok=True)
    settings.resolved_snapshots_dir().mkdir(parents=True, exist_ok=True)
    settings.resolved_videos_dir().mkdir(parents=True, exist_ok=True)
    settings.resolved_previews_dir().mkdir(parents=True, exist_ok=True)
    settings.resolved_rejected_faces_dir().mkdir(parents=True, exist_ok=True)
    settings.resolved_log_dir().mkdir(parents=True, exist_ok=True)
    await init_db()
    logger.info("Application startup complete (version=%s)", __version__)
    yield
    logger.info("Application shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    settings.resolved_previews_dir().mkdir(parents=True, exist_ok=True)
    settings.resolved_rejected_faces_dir().mkdir(parents=True, exist_ok=True)
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(persons.router, prefix="/api/v1")
    app.include_router(detections.router, prefix="/api/v1")
    app.include_router(processing.router, prefix="/api/v1")
    app.include_router(live_test.router, prefix="/api/v1")
    app.include_router(observability.router, prefix="/api/v1")
    app.mount("/data/previews", StaticFiles(directory=settings.resolved_previews_dir()), name="previews")
    app.mount("/data/debug/rejected_faces", StaticFiles(directory=settings.resolved_rejected_faces_dir()), name="rejected_faces")
    return app


app = create_app()

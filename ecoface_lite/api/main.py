"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ecoface_lite import __version__
from ecoface_lite.api.routers import alerts, cameras, detections, experimental, health, incidents, live_test, observability, persons, processing, sites, zones
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
    # Restore in-memory alert sessions that were open before this restart
    from ecoface_lite.db.session import get_session_factory
    from ecoface_lite.services.alert_session_engine import get_alert_session_engine
    from ecoface_lite.services.health_monitor import start_health_monitor
    _alert_engine = get_alert_session_engine()
    _settings = get_settings()
    _session_factory = get_session_factory()
    async with _session_factory() as _startup_session:
        await _alert_engine.rebuild_from_db(
            _startup_session,
            lookback_minutes=_settings.alert_session_rebuild_minutes,
        )
    # VSL Phase 2: start health monitor as standalone asyncio task
    _health_task = start_health_monitor(_session_factory, _settings)
    logger.info("Application startup complete (version=%s)", __version__)
    yield
    if _health_task is not None:
        _health_task.cancel()
        try:
            await _health_task
        except Exception:
            pass
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(persons.router, prefix="/api/v1")
    app.include_router(detections.router, prefix="/api/v1")
    app.include_router(processing.router, prefix="/api/v1")
    app.include_router(live_test.router, prefix="/api/v1")
    app.include_router(observability.router, prefix="/api/v1")
    app.include_router(experimental.router, prefix="/api/v1")
    app.include_router(cameras.router, prefix="/api/v1")
    app.include_router(sites.router, prefix="/api/v1")
    app.include_router(zones.router, prefix="/api/v1")
    app.include_router(incidents.router, prefix="/api/v1")
    app.include_router(alerts.router, prefix="/api/v1")
    app.mount("/data/previews", StaticFiles(directory=settings.resolved_previews_dir()), name="previews")
    app.mount("/data/snapshots", StaticFiles(directory=settings.resolved_snapshots_dir()), name="snapshots")
    app.mount("/data/debug/rejected_faces", StaticFiles(directory=settings.resolved_rejected_faces_dir()), name="rejected_faces")
    app.mount("/data/uploads", StaticFiles(directory=settings.resolved_uploads_dir()), name="uploads")
    return app


app = create_app()

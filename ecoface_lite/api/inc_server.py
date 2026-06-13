"""Lightweight incident-management API server.

Mounts only INC-related routers (health · incidents · persons).
No ML, no pipeline, no GPU dependencies — runs on any host.

Start:
    uvicorn ecoface_lite.api.inc_server:inc_app --port 8001 --reload

Shares the same SQLite/PostgreSQL database as the core engine.
The engine writes DetectionEvents + Sightings; this server manages
the incident lifecycle and serves them to the frontend and future viewers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ecoface_lite import __version__
from ecoface_lite.api.routers import health, incidents, persons
from ecoface_lite.core.config import get_settings
from ecoface_lite.core.logging import get_logger, setup_logging
from ecoface_lite.db.session import init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    settings.resolved_uploads_dir().mkdir(parents=True, exist_ok=True)
    await init_db()
    logger.info("INC API startup (version=%s)", __version__)
    yield
    logger.info("INC API shutdown")


def create_inc_app() -> FastAPI:
    settings = get_settings()
    settings.resolved_uploads_dir().mkdir(parents=True, exist_ok=True)
    app = FastAPI(
        title="EchoFace INC API",
        description="Case management — incidents · persons · sightings",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # public-facing by design; restrict in production
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(incidents.router, prefix="/api/v1")
    app.include_router(persons.router, prefix="/api/v1")
    app.mount(
        "/data/uploads",
        StaticFiles(directory=settings.resolved_uploads_dir()),
        name="uploads",
    )
    return app


inc_app = create_inc_app()

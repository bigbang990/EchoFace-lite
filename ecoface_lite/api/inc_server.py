"""Incident-management router definitions — NOT currently run as a standalone server.

All three routers (health · incidents · persons) are already mounted by
ecoface_lite.api.main:app on port 8000.  The single-server setup is the
default; both engine and INC routes are served from the same process.

To run standalone in the future (Phase B / public viewer):
    uvicorn ecoface_lite.api.inc_server:inc_app --port 8001 --reload

That configuration requires a second ngrok tunnel when hosted on Colab.
See scripts/colab_start.py for the two-tunnel variant and
AI_CONTEXT/roadmap.md for the Phase B plan.
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

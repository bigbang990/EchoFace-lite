"""Async engine and session factory.

Why SQLAlchemy + async:
- Same ORM layer for SQLite (dev) and PostgreSQL (prod) via `DATABASE_URL`.
- FastAPI endpoints stay non-blocking for I/O bound DB work.

SQLite: `connect_args["timeout"]` increases busy-handler wait (reduces "database is
locked" under concurrent API + background workers). WAL mode improves read/write
concurrency for a single-file DB.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ecoface_lite.core.config import get_settings
from ecoface_lite.core.logging import get_logger
from ecoface_lite.db.base import Base

logger = get_logger(__name__)

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        kwargs: dict = {"echo": settings.debug}
        if settings.database_url.startswith("sqlite"):
            kwargs["connect_args"] = {"timeout": 30.0}
        _engine = create_async_engine(settings.database_url, **kwargs)
        if settings.database_url.startswith("sqlite"):
            from sqlalchemy import event

            @event.listens_for(_engine.sync_engine, "connect")
            def _sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()

        logger.info("Database engine created for URL scheme: %s", settings.database_url.split(":", 1)[0])
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def _sqlite_apply_schema_patches() -> None:
    """Best-effort ALTERs for existing SQLite files (create_all does not add new columns)."""
    settings = get_settings()
    if not settings.database_url.startswith("sqlite"):
        return
    engine = get_engine()
    async with engine.begin() as conn:
        for stmt in (
            "ALTER TABLE persons ADD COLUMN source_image_hash VARCHAR(64)",
            "ALTER TABLE face_embeddings ADD COLUMN ingest_sha256 VARCHAR(64)",
            "ALTER TABLE processing_status ADD COLUMN alerts_created INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE processing_status ADD COLUMN avg_fps FLOAT",
            "ALTER TABLE processing_status ADD COLUMN avg_confidence FLOAT",
            "ALTER TABLE processing_status ADD COLUMN total_faces_detected INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE processing_status ADD COLUMN total_faces_rejected INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE processing_status ADD COLUMN blur_rejections INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE processing_status ADD COLUMN duplicate_suppressions INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE processing_status ADD COLUMN processing_duration_seconds FLOAT",
            "ALTER TABLE processing_status ADD COLUMN camera_id VARCHAR(128)",
            "ALTER TABLE detection_events ADD COLUMN camera_id VARCHAR(128)",
            "ALTER TABLE detection_events ADD COLUMN camera_id_int INTEGER REFERENCES cameras(id)",
            "CREATE TABLE IF NOT EXISTS cameras (id INTEGER PRIMARY KEY AUTOINCREMENT, label VARCHAR(255) NOT NULL, stream_url VARCHAR(1024), location VARCHAR(512), is_active BOOLEAN NOT NULL DEFAULT 1, created_at DATETIME DEFAULT (CURRENT_TIMESTAMP))",
            "CREATE TABLE IF NOT EXISTS incidents (id INTEGER PRIMARY KEY AUTOINCREMENT, title VARCHAR(512) NOT NULL, description TEXT, status VARCHAR(32) NOT NULL DEFAULT 'open', operator_id VARCHAR(128), is_paused BOOLEAN NOT NULL DEFAULT 0, created_at DATETIME DEFAULT (CURRENT_TIMESTAMP), updated_at DATETIME DEFAULT (CURRENT_TIMESTAMP))",
            "CREATE TABLE IF NOT EXISTS sightings (id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE, detection_id INTEGER REFERENCES detection_events(id) ON DELETE SET NULL, camera_id INTEGER REFERENCES cameras(id) ON DELETE SET NULL, notes TEXT, status VARCHAR(32) NOT NULL DEFAULT 'pending', created_at DATETIME DEFAULT (CURRENT_TIMESTAMP))",
            "ALTER TABLE sightings ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT 'pending'",
            "ALTER TABLE incidents ADD COLUMN is_paused BOOLEAN NOT NULL DEFAULT 0",
            "CREATE TABLE IF NOT EXISTS incident_persons (incident_id INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE, person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE, PRIMARY KEY (incident_id, person_id))",
            "ALTER TABLE persons ADD COLUMN extra_photo_paths TEXT",
        ):
            try:
                await conn.execute(text(stmt))
            except Exception:
                pass
        for stmt in (
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_persons_source_image_hash "
            "ON persons(source_image_hash) WHERE source_image_hash IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS ix_face_embeddings_ingest_sha256 ON face_embeddings(ingest_sha256)",
        ):
            try:
                await conn.execute(text(stmt))
            except Exception:
                pass


async def init_db() -> None:
    """Create tables if they do not exist (MVP). Replace with Alembic for production."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _sqlite_apply_schema_patches()

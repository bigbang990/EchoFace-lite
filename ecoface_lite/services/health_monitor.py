"""Health monitor background task (VSL Phase 2).

Polls every registered active camera on a configurable timer.
Writes status + last_seen directly to the DB — zero HTTP overhead.

Hard constraint (see CLAUDE.md VSL hard stops):
  This task runs on its OWN asyncio timer, completely separate from the
  frame acquisition loop. It must NEVER be called from within a frame loop.
  A camera going offline must not block frame processing on any other source.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy import select

from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


async def _poll_all_cameras(session_factory, settings) -> None:
    """One health-check pass across all active cameras."""
    from ecoface_lite.db.models import Camera
    from ecoface_lite.input_sources.source_registry import get_source_registry

    registry = get_source_registry()

    async with session_factory() as session:
        result = await session.execute(select(Camera).where(Camera.is_active == True))  # noqa: E712
        cameras = result.scalars().all()

    if not cameras:
        return

    for camera in cameras:
        new_status = "unknown"
        new_last_seen = None
        error_detail = None
        try:
            source = registry.build_source(camera)
            connected = source.connect()
            health = source.health_check()
            source.disconnect()
            new_status = health.status.value
            if connected:
                new_last_seen = datetime.now(timezone.utc)
        except Exception as exc:
            new_status = "offline"
            error_detail = str(exc)
            logger.warning(
                "Health monitor: camera %d (%s) error — %s",
                camera.id, camera.label, exc,
            )

        try:
            async with session_factory() as session:
                result = await session.execute(select(Camera).where(Camera.id == camera.id))
                cam = result.scalar_one_or_none()
                if cam is not None:
                    cam.status = new_status
                    if new_last_seen is not None:
                        cam.last_seen = new_last_seen
                    await session.commit()
        except Exception as exc:
            logger.error("Health monitor: DB write failed for camera %d — %s", camera.id, exc)

        if error_detail:
            logger.info(
                "Health monitor: camera %d (%s) → %s [%s]",
                camera.id, camera.label, new_status, error_detail,
            )
        else:
            logger.debug(
                "Health monitor: camera %d (%s) → %s",
                camera.id, camera.label, new_status,
            )


async def _health_monitor_loop(session_factory, settings) -> None:
    """Runs forever, polling cameras every health_monitor_interval_seconds."""
    logger.info(
        "Health monitor started — interval=%ds, monitoring active cameras",
        settings.health_monitor_interval_seconds,
    )
    while True:
        try:
            await _poll_all_cameras(session_factory, settings)
        except Exception as exc:
            logger.error("Health monitor poll error (will retry): %s", exc)
        await asyncio.sleep(settings.health_monitor_interval_seconds)


def start_health_monitor(session_factory, settings) -> asyncio.Task | None:
    """Create and schedule the health monitor task. Returns the Task or None if disabled."""
    if not settings.health_monitor_enabled:
        logger.info("Health monitor disabled (HEALTH_MONITOR_ENABLED=false)")
        return None
    task = asyncio.create_task(
        _health_monitor_loop(session_factory, settings),
        name="health_monitor",
    )
    return task

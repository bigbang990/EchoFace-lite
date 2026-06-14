"""SourceRegistry — SQLite-backed registry of registered video sources (VSL Phase 1).

Bridges the Camera DB table and BaseVideoSource instances.
The pipeline receives a BaseVideoSource; it never touches Camera rows directly.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ecoface_lite.core.logging import get_logger
from ecoface_lite.input_sources.android_source import AndroidCameraSource
from ecoface_lite.input_sources.base import BaseVideoSource, SourceType
from ecoface_lite.input_sources.rtsp_source import RTSPSource
from ecoface_lite.input_sources.video_file import VideoFileSource

logger = get_logger(__name__)


class SourceRegistry:
    """Maps Camera DB rows to concrete BaseVideoSource instances.

    Does not own connections — callers are responsible for connect/disconnect.
    """

    def build_source(self, camera: "Camera") -> BaseVideoSource:  # type: ignore[name-defined]
        """Instantiate the appropriate BaseVideoSource for a Camera row."""
        stype = getattr(camera, "source_type", SourceType.FILE.value)
        source_id = str(camera.id)
        name = camera.label
        zone = getattr(camera, "zone", None)
        location = camera.location

        if stype == SourceType.ANDROID.value:
            if not camera.stream_url:
                raise ValueError(f"Camera {camera.id} has source_type 'android' but no stream_url")
            return AndroidCameraSource(
                source_id=source_id,
                name=name,
                stream_url=camera.stream_url,
                zone=zone,
                location=location,
            )

        if stype == SourceType.RTSP.value:
            if not camera.stream_url:
                raise ValueError(f"Camera {camera.id} has source_type 'rtsp' but no stream_url")
            return RTSPSource(
                source_id=source_id,
                name=name,
                stream_url=camera.stream_url,
                zone=zone,
                location=location,
                source_type=SourceType.RTSP,
            )

        # Default: file source (source_type == "file" or legacy rows with no type)
        if not camera.stream_url:
            raise ValueError(f"Camera {camera.id} has source_type 'file' but no stream_url / path")
        return VideoFileSource(
            path=Path(camera.stream_url),
            source_id=source_id,
            name=name,
            zone=zone,
            location=location,
        )

    async def list_cameras(self, session: AsyncSession) -> list:
        """Return all Camera rows."""
        from ecoface_lite.db.models import Camera
        result = await session.execute(select(Camera))
        return list(result.scalars().all())

    async def get_camera(self, session: AsyncSession, camera_id: int):
        """Return a single Camera row or None."""
        from ecoface_lite.db.models import Camera
        result = await session.execute(select(Camera).where(Camera.id == camera_id))
        return result.scalar_one_or_none()

    async def register(
        self,
        session: AsyncSession,
        *,
        name: str,
        source_type: str,
        stream_url: str | None,
        zone: str | None,
        location: str | None,
    ):
        """Create and persist a Camera row. Returns the new Camera."""
        from ecoface_lite.db.models import Camera
        camera = Camera(
            label=name,
            source_type=source_type,
            stream_url=stream_url,
            zone=zone,
            location=location,
            status="unknown",
        )
        session.add(camera)
        await session.commit()
        await session.refresh(camera)
        logger.info("SourceRegistry: registered camera %d (%s) type=%s", camera.id, name, source_type)
        return camera


_registry: SourceRegistry | None = None


def get_source_registry() -> SourceRegistry:
    global _registry
    if _registry is None:
        _registry = SourceRegistry()
    return _registry

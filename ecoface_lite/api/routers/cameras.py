from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from ecoface_lite.api.deps import DbSession
from ecoface_lite.api.schemas import CameraCreate, CameraHealthUpdate, CameraOut
from ecoface_lite.db.models import Camera, Zone
from ecoface_lite.input_sources.source_registry import get_source_registry

router = APIRouter(prefix="/cameras", tags=["cameras"])


class _CameraActiveUpdate(BaseModel):
    is_active: bool


@router.post("", response_model=CameraOut, status_code=201)
async def create_camera(body: CameraCreate, db: DbSession) -> CameraOut:
    if body.zone_id is not None:
        result = await db.execute(select(Zone).where(Zone.id == body.zone_id))
        if result.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Zone not found")
    camera = Camera(
        label=body.label,
        stream_url=body.stream_url,
        location=body.location,
        source_type=body.source_type,
        zone=body.zone,
        zone_id=body.zone_id,
        direction=body.direction,
        overlap_group=body.overlap_group,
        supports_live=body.supports_live,
        supports_historical=body.supports_historical,
        supports_ptz=body.supports_ptz,
        retention_days=body.retention_days,
        trust_level=body.trust_level,
        status="unknown",
    )
    db.add(camera)
    await db.commit()
    await db.refresh(camera)
    return CameraOut.model_validate(camera)


@router.get("", response_model=list[CameraOut])
async def list_cameras(db: DbSession) -> list[CameraOut]:
    result = await db.execute(select(Camera))
    return [CameraOut.model_validate(c) for c in result.scalars().all()]


@router.get("/{camera_id}", response_model=CameraOut)
async def get_camera(camera_id: int, db: DbSession) -> CameraOut:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return CameraOut.model_validate(camera)


@router.patch("/{camera_id}", response_model=CameraOut)
async def update_camera_active(
    camera_id: int,
    body: _CameraActiveUpdate,
    db: DbSession,
) -> CameraOut:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    camera.is_active = body.is_active
    await db.commit()
    await db.refresh(camera)
    return CameraOut.model_validate(camera)


@router.patch("/{camera_id}/health", response_model=CameraOut)
async def update_camera_health(
    camera_id: int,
    body: CameraHealthUpdate,
    db: DbSession,
) -> CameraOut:
    """Update camera health status — called by the health monitor (VSL Phase 2)."""
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    camera.status = body.status
    if body.last_seen is not None:
        camera.last_seen = body.last_seen
    elif body.status == "online":
        camera.last_seen = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(camera)
    return CameraOut.model_validate(camera)


@router.post("/{camera_id}/test-connect", status_code=200)
async def test_connect(camera_id: int, db: DbSession) -> dict:
    """Attempt a live connection to the source and return health status."""
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    registry = get_source_registry()
    try:
        source = registry.build_source(camera)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    connected = source.connect()
    health = source.health_check()
    source.disconnect()

    # Persist the result
    camera.status = health.status.value
    if connected:
        camera.last_seen = datetime.now(timezone.utc)
    await db.commit()

    return {
        "camera_id": camera_id,
        "connected": connected,
        "status": health.status.value,
        "error": health.error,
    }


@router.get("/health-summary", response_model=dict)
async def health_summary(db: DbSession) -> dict:
    """Aggregate camera health counts for the dashboard panel (VSL Phase 2)."""
    result = await db.execute(select(Camera))
    cameras = result.scalars().all()
    counts: dict[str, int] = {"total": 0, "online": 0, "offline": 0, "reconnecting": 0, "unknown": 0}
    for cam in cameras:
        counts["total"] += 1
        status = cam.status or "unknown"
        if status in counts:
            counts[status] += 1
        else:
            counts["unknown"] += 1
    return counts


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: int, db: DbSession) -> None:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    await db.delete(camera)
    await db.commit()

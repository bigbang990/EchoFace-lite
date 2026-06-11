from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from ecoface_lite.api.deps import DbSession
from ecoface_lite.api.schemas import CameraCreate, CameraOut
from ecoface_lite.db.models import Camera

router = APIRouter(prefix="/cameras", tags=["cameras"])


class _CameraActiveUpdate(BaseModel):
    is_active: bool


@router.post("", response_model=CameraOut, status_code=201)
async def create_camera(body: CameraCreate, db: DbSession) -> CameraOut:
    camera = Camera(
        label=body.label,
        stream_url=body.stream_url,
        location=body.location,
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


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: int, db: DbSession) -> None:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    await db.delete(camera)
    await db.commit()

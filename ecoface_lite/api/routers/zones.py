"""Zones router — second level of the location hierarchy (VSL Phase 2)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from ecoface_lite.api.deps import DbSession
from ecoface_lite.api.schemas import ZoneCreate, ZoneOut
from ecoface_lite.db.models import Site, Zone

router = APIRouter(prefix="/zones", tags=["locations"])


@router.post("", response_model=ZoneOut, status_code=201)
async def create_zone(body: ZoneCreate, db: DbSession) -> ZoneOut:
    result = await db.execute(select(Site).where(Site.id == body.site_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Site not found")
    zone = Zone(site_id=body.site_id, name=body.name, description=body.description)
    db.add(zone)
    await db.commit()
    await db.refresh(zone)
    return ZoneOut.model_validate(zone)


@router.get("", response_model=list[ZoneOut])
async def list_zones(db: DbSession) -> list[ZoneOut]:
    result = await db.execute(select(Zone).order_by(Zone.site_id, Zone.name))
    return [ZoneOut.model_validate(z) for z in result.scalars().all()]


@router.get("/{zone_id}", response_model=ZoneOut)
async def get_zone(zone_id: int, db: DbSession) -> ZoneOut:
    result = await db.execute(select(Zone).where(Zone.id == zone_id))
    zone = result.scalar_one_or_none()
    if zone is None:
        raise HTTPException(status_code=404, detail="Zone not found")
    return ZoneOut.model_validate(zone)


@router.delete("/{zone_id}", status_code=204)
async def delete_zone(zone_id: int, db: DbSession) -> None:
    result = await db.execute(select(Zone).where(Zone.id == zone_id))
    zone = result.scalar_one_or_none()
    if zone is None:
        raise HTTPException(status_code=404, detail="Zone not found")
    await db.delete(zone)
    await db.commit()

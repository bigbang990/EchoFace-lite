"""Sites router — top of the location hierarchy (VSL Phase 2)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from ecoface_lite.api.deps import DbSession
from ecoface_lite.api.schemas import SiteCreate, SiteOut, ZoneOut
from ecoface_lite.db.models import Site, Zone

router = APIRouter(prefix="/sites", tags=["locations"])


@router.post("", response_model=SiteOut, status_code=201)
async def create_site(body: SiteCreate, db: DbSession) -> SiteOut:
    site = Site(name=body.name, description=body.description)
    db.add(site)
    await db.commit()
    await db.refresh(site)
    return SiteOut.model_validate(site)


@router.get("", response_model=list[SiteOut])
async def list_sites(db: DbSession) -> list[SiteOut]:
    result = await db.execute(select(Site).order_by(Site.name))
    return [SiteOut.model_validate(s) for s in result.scalars().all()]


@router.get("/{site_id}", response_model=SiteOut)
async def get_site(site_id: int, db: DbSession) -> SiteOut:
    result = await db.execute(select(Site).where(Site.id == site_id))
    site = result.scalar_one_or_none()
    if site is None:
        raise HTTPException(status_code=404, detail="Site not found")
    return SiteOut.model_validate(site)


@router.get("/{site_id}/zones", response_model=list[ZoneOut])
async def list_zones_for_site(site_id: int, db: DbSession) -> list[ZoneOut]:
    result = await db.execute(select(Site).where(Site.id == site_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Site not found")
    result = await db.execute(select(Zone).where(Zone.site_id == site_id).order_by(Zone.name))
    return [ZoneOut.model_validate(z) for z in result.scalars().all()]


@router.delete("/{site_id}", status_code=204)
async def delete_site(site_id: int, db: DbSession) -> None:
    result = await db.execute(select(Site).where(Site.id == site_id))
    site = result.scalar_one_or_none()
    if site is None:
        raise HTTPException(status_code=404, detail="Site not found")
    await db.delete(site)
    await db.commit()

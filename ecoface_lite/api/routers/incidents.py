from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ecoface_lite.api.deps import DbSession
from ecoface_lite.api.schemas import (
    IncidentCreate,
    IncidentOut,
    IncidentPersonOut,
    IncidentStatusUpdate,
    PersonOut,
    SightingCreate,
    SightingOut,
)
from ecoface_lite.db.models import DetectionEvent, Incident, Person, Sighting

router = APIRouter(prefix="/incidents", tags=["incidents"])


class IncidentDetailOut(IncidentOut):
    sightings: list[SightingOut] = []


def _incident_out(inc: Incident) -> IncidentOut:
    """Build IncidentOut with computed ref/counts from loaded relationships."""
    return IncidentOut(
        id=inc.id,
        ref=f"INC-{inc.id:03d}",
        title=inc.title,
        description=inc.description,
        status=inc.status,
        operator_id=inc.operator_id,
        created_at=inc.created_at,
        updated_at=inc.updated_at,
        person_count=len(inc.persons) if inc.persons is not None else 0,
        alert_count=len(inc.sightings) if inc.sightings is not None else 0,
    )


@router.post("", response_model=IncidentOut, status_code=201)
async def create_incident(body: IncidentCreate, db: DbSession) -> IncidentOut:
    incident = Incident(
        title=body.title,
        description=body.description,
        operator_id=body.operator_id,
        status="open",
    )
    db.add(incident)
    await db.commit()
    await db.refresh(incident)
    # eager-load relationships so _incident_out can count them
    result = await db.execute(
        select(Incident)
        .options(selectinload(Incident.persons), selectinload(Incident.sightings))
        .where(Incident.id == incident.id)
    )
    return _incident_out(result.scalar_one())


@router.get("", response_model=list[IncidentOut])
async def list_incidents(
    db: DbSession,
    status: str | None = Query(default=None),
) -> list[IncidentOut]:
    stmt = (
        select(Incident)
        .options(selectinload(Incident.persons), selectinload(Incident.sightings))
    )
    if status is not None:
        stmt = stmt.where(Incident.status == status)
    result = await db.execute(stmt)
    return [_incident_out(i) for i in result.scalars().all()]


@router.get("/{incident_id}", response_model=IncidentDetailOut)
async def get_incident(incident_id: int, db: DbSession) -> IncidentDetailOut:
    result = await db.execute(
        select(Incident)
        .options(selectinload(Incident.persons), selectinload(Incident.sightings))
        .where(Incident.id == incident_id)
    )
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    sighting_result = await db.execute(
        select(Sighting).where(Sighting.incident_id == incident_id)
    )
    sightings_out = [SightingOut.model_validate(s) for s in sighting_result.scalars().all()]
    inc_data = _incident_out(incident).model_dump()
    inc_data["sightings"] = sightings_out
    return IncidentDetailOut(**inc_data)


@router.patch("/{incident_id}/status", response_model=IncidentOut)
async def update_incident_status(
    incident_id: int,
    body: IncidentStatusUpdate,
    db: DbSession,
) -> IncidentOut:
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.status = body.status
    await db.commit()
    result2 = await db.execute(
        select(Incident)
        .options(selectinload(Incident.persons), selectinload(Incident.sightings))
        .where(Incident.id == incident_id)
    )
    return _incident_out(result2.scalar_one())


@router.post("/{incident_id}/sightings", response_model=SightingOut, status_code=201)
async def add_sighting(
    incident_id: int,
    body: SightingCreate,
    db: DbSession,
) -> SightingOut:
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    sighting = Sighting(
        incident_id=incident_id,
        detection_id=body.detection_id,
        camera_id=body.camera_id,
        notes=body.notes,
    )
    db.add(sighting)
    await db.commit()
    await db.refresh(sighting)
    return SightingOut.model_validate(sighting)


@router.get("/{incident_id}/sightings", response_model=list[SightingOut])
async def list_sightings(incident_id: int, db: DbSession) -> list[SightingOut]:
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    sighting_result = await db.execute(
        select(Sighting)
        .options(
            selectinload(Sighting.detection).selectinload(DetectionEvent.person)
        )
        .where(Sighting.incident_id == incident_id)
        .order_by(Sighting.created_at)
    )
    out = []
    for s in sighting_result.scalars().all():
        det = s.detection
        person = det.person if det else None
        out.append(SightingOut(
            id=s.id,
            incident_id=s.incident_id,
            detection_id=s.detection_id,
            camera_id=s.camera_id,
            notes=s.notes,
            created_at=s.created_at,
            person_id=person.id if person else None,
            person_name=person.display_name if person else None,
            confidence=det.confidence if det else None,
            source_name=det.source_label if det else None,
            frame_index=det.frame_index if det else None,
            snapshot_path=det.snapshot_path if det else None,
        ))
    return out


@router.post("/{incident_id}/persons/{person_id}", response_model=IncidentPersonOut, status_code=201)
async def link_person_to_incident(
    incident_id: int,
    person_id: int,
    db: DbSession,
) -> IncidentPersonOut:
    result = await db.execute(
        select(Incident).where(Incident.id == incident_id).options(selectinload(Incident.persons))
    )
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    person_result = await db.execute(select(Person).where(Person.id == person_id))
    person = person_result.scalar_one_or_none()
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    if any(p.id == person_id for p in incident.persons):
        raise HTTPException(status_code=409, detail="Person already linked to this incident")
    incident.persons.append(person)
    await db.commit()
    return IncidentPersonOut(incident_id=incident_id, person_id=person_id, person_name=person.display_name)


@router.delete("/{incident_id}/persons/{person_id}", status_code=204)
async def unlink_person_from_incident(
    incident_id: int,
    person_id: int,
    db: DbSession,
) -> None:
    result = await db.execute(
        select(Incident).where(Incident.id == incident_id).options(selectinload(Incident.persons))
    )
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.persons = [p for p in incident.persons if p.id != person_id]
    await db.commit()


@router.get("/{incident_id}/persons", response_model=list[PersonOut])
async def list_incident_persons(incident_id: int, db: DbSession) -> list[PersonOut]:
    result = await db.execute(
        select(Incident).where(Incident.id == incident_id).options(selectinload(Incident.persons))
    )
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return [PersonOut.model_validate(p) for p in incident.persons]

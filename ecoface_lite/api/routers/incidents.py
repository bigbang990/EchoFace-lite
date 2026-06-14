from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from ecoface_lite.api.deps import DbSession
from ecoface_lite.api.schemas import (
    IncidentCloseRequest,
    IncidentCreate,
    IncidentOut,
    IncidentPersonOut,
    IncidentStatusUpdate,
    IncidentPauseUpdate,
    PersonOut,
    SightingCreate,
    SightingOut,
    SightingStatusUpdate,
)
from ecoface_lite.db.models import Alert, DetectionEvent, Incident, Person, Sighting, incident_persons

router = APIRouter(prefix="/incidents", tags=["incidents"])


class IncidentDetailOut(IncidentOut):
    sightings: list[SightingOut] = []


def _incident_out(inc: Incident, person_count: int = 0, alert_count: int = 0, pending_alert_count: int = 0) -> IncidentOut:
    """Build IncidentOut with explicit counts (avoids selectinload on many-to-many)."""
    return IncidentOut(
        id=inc.id,
        ref=f"INC-{inc.id:03d}",
        title=inc.title,
        description=inc.description,
        status=inc.status,
        operator_id=inc.operator_id,
        is_paused=inc.is_paused,
        created_at=inc.created_at,
        updated_at=inc.updated_at,
        person_count=person_count,
        alert_count=alert_count,
        pending_alert_count=pending_alert_count,
        closing_reason=inc.closing_reason,
        closing_summary=inc.closing_summary,
        closed_by=inc.closed_by,
        closed_at=inc.closed_at,
        evidence_paths=inc.evidence_paths,
    )


async def _count_persons(db: DbSession, incident_ids: list[int]) -> dict[int, int]:
    if not incident_ids:
        return {}
    rows = (await db.execute(
        select(incident_persons.c.incident_id, func.count(incident_persons.c.person_id))
        .where(incident_persons.c.incident_id.in_(incident_ids))
        .group_by(incident_persons.c.incident_id)
    )).all()
    return dict(rows)


async def _count_sightings(db: DbSession, incident_ids: list[int]) -> dict[int, int]:
    if not incident_ids:
        return {}
    rows = (await db.execute(
        select(Sighting.incident_id, func.count(Sighting.id))
        .where(Sighting.incident_id.in_(incident_ids))
        .group_by(Sighting.incident_id)
    )).all()
    return dict(rows)


async def _count_pending_sightings(db: DbSession, incident_ids: list[int]) -> dict[int, int]:
    if not incident_ids:
        return {}
    rows = (await db.execute(
        select(Sighting.incident_id, func.count(Sighting.id))
        .where(Sighting.incident_id.in_(incident_ids))
        .where(Sighting.status == "pending")
        .group_by(Sighting.incident_id)
    )).all()
    return dict(rows)


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
    return _incident_out(incident, person_count=0, alert_count=0, pending_alert_count=0)


@router.get("", response_model=list[IncidentOut])
async def list_incidents(
    db: DbSession,
    status: str | None = Query(default=None),
) -> list[IncidentOut]:
    stmt = select(Incident)
    if status is not None:
        stmt = stmt.where(Incident.status == status)
    result = await db.execute(stmt)
    incidents = result.scalars().all()
    inc_ids = [i.id for i in incidents]
    pc = await _count_persons(db, inc_ids)
    sc = await _count_sightings(db, inc_ids)
    psc = await _count_pending_sightings(db, inc_ids)
    return [_incident_out(i, pc.get(i.id, 0), sc.get(i.id, 0), psc.get(i.id, 0)) for i in incidents]


@router.get("/{incident_id}", response_model=IncidentDetailOut)
async def get_incident(incident_id: int, db: DbSession) -> IncidentDetailOut:
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    pc = await _count_persons(db, [incident_id])
    sc = await _count_sightings(db, [incident_id])
    psc = await _count_pending_sightings(db, [incident_id])
    sightings_out = await _build_sightings_out(db, incident_id)
    inc_data = _incident_out(incident, pc.get(incident_id, 0), sc.get(incident_id, 0), psc.get(incident_id, 0)).model_dump()
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
    await db.refresh(incident)
    pc = await _count_persons(db, [incident_id])
    sc = await _count_sightings(db, [incident_id])
    psc = await _count_pending_sightings(db, [incident_id])
    return _incident_out(incident, pc.get(incident_id, 0), sc.get(incident_id, 0), psc.get(incident_id, 0))


@router.patch("/{incident_id}/pause", response_model=IncidentOut)
async def update_incident_pause(
    incident_id: int,
    body: IncidentPauseUpdate,
    db: DbSession,
) -> IncidentOut:
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.is_paused = body.is_paused
    await db.commit()
    await db.refresh(incident)
    pc = await _count_persons(db, [incident_id])
    sc = await _count_sightings(db, [incident_id])
    psc = await _count_pending_sightings(db, [incident_id])
    return _incident_out(incident, pc.get(incident_id, 0), sc.get(incident_id, 0), psc.get(incident_id, 0))


async def _build_sightings_out(db: DbSession, incident_id: int) -> list[SightingOut]:
    sighting_result = await db.execute(
        select(Sighting)
        .options(selectinload(Sighting.detection).selectinload(DetectionEvent.person))
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
            status=s.status,
            created_at=s.created_at,
            person_id=person.id if person else None,
            person_name=person.display_name if person else None,
            confidence=det.confidence if det else None,
            source_name=det.source_label if det else None,
            frame_index=det.frame_index if det else None,
            snapshot_path=det.snapshot_path if det else None,
        ))
    return out


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
    return await _build_sightings_out(db, incident_id)


@router.patch("/{incident_id}/sightings/{sighting_id}", response_model=SightingOut)
async def update_sighting_status(
    incident_id: int,
    sighting_id: int,
    body: SightingStatusUpdate,
    db: DbSession,
) -> SightingOut:
    result = await db.execute(
        select(Sighting)
        .where(Sighting.id == sighting_id, Sighting.incident_id == incident_id)
    )
    sighting = result.scalar_one_or_none()
    if sighting is None:
        raise HTTPException(status_code=404, detail="Sighting not found")
    sighting.status = body.status
    await db.commit()
    await db.refresh(sighting)
    # Re-fetch with enriched fields
    sightings = await _build_sightings_out(db, incident_id)
    for s in sightings:
        if s.id == sighting_id:
            return s
    return SightingOut.model_validate(sighting)


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


@router.post("/{incident_id}/close", response_model=IncidentOut)
async def close_incident(
    incident_id: int,
    body: IncidentCloseRequest,
    db: DbSession,
) -> IncidentOut:
    """Close a case. Requires a reason and a closing summary.

    Side effects (atomic):
    - incident.status → "closed"
    - All open alerts for this incident are set to status "closed"
    - In-memory alert engine sessions for this incident are evicted
    - Closure metadata (reason, summary, closed_by, closed_at) persisted
    """
    from datetime import datetime, timezone

    from sqlalchemy import update

    from ecoface_lite.services.alert_session_engine import get_alert_session_engine

    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    if incident.status == "closed":
        raise HTTPException(status_code=409, detail="Incident is already closed")

    now = datetime.now(tz=timezone.utc)
    incident.status = "closed"
    incident.is_paused = False
    incident.closing_reason = body.reason
    incident.closing_summary = body.summary
    incident.closed_by = body.closed_by
    incident.closed_at = now
    incident.updated_at = now

    # Close all open alert sessions for this incident
    await db.execute(
        update(Alert)
        .where(Alert.incident_id == incident_id, Alert.status == "open")
        .values(status="closed", updated_at=now)
    )

    await db.commit()

    # Evict from in-memory registry so no new sightings attach to these sessions
    await get_alert_session_engine().evict_incident(incident_id)

    pc = await _count_persons(db, [incident_id])
    sc = await _count_sightings(db, [incident_id])
    psc = await _count_pending_sightings(db, [incident_id])
    return _incident_out(incident, pc.get(incident_id, 0), sc.get(incident_id, 0), psc.get(incident_id, 0))


@router.post("/{incident_id}/evidence", response_model=IncidentOut)
async def upload_incident_evidence(
    incident_id: int,
    db: DbSession,
    files: list[UploadFile] = File(...),
) -> IncidentOut:
    """Upload evidence files to an incident (call before or during closure).

    Files are saved to data/evidence/{incident_id}/ and paths appended to
    incident.evidence_paths (JSON list). Can be called on open or closed incidents.
    """
    import json as _json
    import uuid as _uuid
    from pathlib import Path as _Path

    from ecoface_lite.core.config import get_settings

    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")

    settings = get_settings()
    evidence_dir = settings.data_dir / "evidence" / str(incident_id)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    existing: list[str] = []
    if incident.evidence_paths:
        try:
            existing = _json.loads(incident.evidence_paths)
        except Exception:
            existing = []

    for f in files:
        suffix = _Path(f.filename or "").suffix.lower() or ".bin"
        name = f"{_uuid.uuid4().hex}{suffix}"
        dest = evidence_dir / name
        with dest.open("wb") as fh:
            while chunk := await f.read(1024 * 1024):
                fh.write(chunk)
        existing.append(str(_Path("data/evidence") / str(incident_id) / name))

    incident.evidence_paths = _json.dumps(existing)
    await db.commit()

    pc = await _count_persons(db, [incident_id])
    sc = await _count_sightings(db, [incident_id])
    psc = await _count_pending_sightings(db, [incident_id])
    return _incident_out(incident, pc.get(incident_id, 0), sc.get(incident_id, 0), psc.get(incident_id, 0))

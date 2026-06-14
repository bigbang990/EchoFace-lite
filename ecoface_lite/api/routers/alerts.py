"""Alert Session endpoints — Phase 8.

GET  /incidents/{incident_id}/alerts          list alerts for an incident
GET  /alerts/{alert_id}                       single alert with sightings
PATCH /alerts/{alert_id}/status               operator status update (open/closed/confirmed/rejected)
PATCH /alerts/{alert_id}/level                operator level update (Phase 11 — sighting/candidate/verified/critical)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ecoface_lite.api.deps import DbSession
from ecoface_lite.api.schemas import AlertLevelUpdate, AlertNoteCreate, AlertOut, AlertStatusUpdate, SightingOut
from ecoface_lite.db.models import Alert, Camera, Incident, Person, Sighting

router = APIRouter(tags=["alerts"])


def _alert_out(alert: Alert, *, with_sightings: bool = False, incident_status: str | None = None) -> AlertOut:
    person_name: str | None = None
    if alert.person is not None:
        person_name = alert.person.display_name
    camera_label: str | None = None
    if alert.camera is not None:
        camera_label = alert.camera.label

    sightings: list[SightingOut] = []
    if with_sightings:
        for s in alert.sightings:
            sightings.append(SightingOut(
                id=s.id,
                incident_id=s.incident_id,
                detection_id=s.detection_id,
                camera_id=s.camera_id,
                notes=s.notes,
                status=s.status,
                created_at=s.created_at,
                person_id=s.person_id,
                confidence=s.confidence,
                frame_index=s.frame_index,
                snapshot_path=s.snapshot_path,
            ))

    return AlertOut(
        id=alert.id,
        incident_id=alert.incident_id,
        person_id=alert.person_id,
        camera_id=alert.camera_id,
        zone_id=alert.zone_id,
        status=alert.status,
        level=alert.level,
        source=alert.source,
        first_seen_at=alert.first_seen_at,
        last_seen_at=alert.last_seen_at,
        sighting_count=alert.sighting_count,
        operator_notes=alert.operator_notes,
        created_at=alert.created_at,
        updated_at=alert.updated_at,
        person_name=person_name,
        camera_label=camera_label,
        incident_status=incident_status,
        sightings=sightings,
    )


@router.get("/incidents/{incident_id}/alerts", response_model=list[AlertOut])
async def list_incident_alerts(
    incident_id: int,
    db: DbSession,
    status: str | None = None,
    level: str | None = None,
    source: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[AlertOut]:
    """Return alerts for an incident, most recent first.

    Optional filters: status, level, source.
    """
    stmt = (
        select(Alert)
        .where(Alert.incident_id == incident_id)
        .options(selectinload(Alert.person), selectinload(Alert.camera))
        .order_by(Alert.last_seen_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if status is not None:
        stmt = stmt.where(Alert.status == status)
    if level is not None:
        stmt = stmt.where(Alert.level == level)
    if source is not None:
        stmt = stmt.where(Alert.source == source)

    rows = (await db.execute(stmt)).scalars().all()
    return [_alert_out(a) for a in rows]


@router.get("/alerts/{alert_id}", response_model=AlertOut)
async def get_alert(alert_id: int, db: DbSession) -> AlertOut:
    """Return a single alert including its full sighting history and incident status."""
    alert = (await db.execute(
        select(Alert)
        .where(Alert.id == alert_id)
        .options(
            selectinload(Alert.person),
            selectinload(Alert.camera),
            selectinload(Alert.sightings),
            selectinload(Alert.incident),
        )
    )).scalars().first()

    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")

    inc_status = alert.incident.status if alert.incident else None
    return _alert_out(alert, with_sightings=True, incident_status=inc_status)


@router.patch("/alerts/{alert_id}/status", response_model=AlertOut)
async def update_alert_status(alert_id: int, body: AlertStatusUpdate, db: DbSession) -> AlertOut:
    """Operator status update — closes or confirms an alert session."""
    from datetime import datetime, timezone

    alert = (await db.execute(
        select(Alert)
        .where(Alert.id == alert_id)
        .options(selectinload(Alert.person), selectinload(Alert.camera), selectinload(Alert.incident))
    )).scalars().first()

    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    if alert.incident and alert.incident.status == "closed":
        raise HTTPException(status_code=409, detail="Case is closed — alert status cannot be changed")

    alert.status = body.status
    alert.updated_at = datetime.now(tz=timezone.utc)
    await db.commit()
    inc_status = alert.incident.status if alert.incident else None
    return _alert_out(alert, incident_status=inc_status)


@router.post("/alerts/{alert_id}/notes", response_model=AlertOut)
async def append_alert_note(alert_id: int, body: AlertNoteCreate, db: DbSession) -> AlertOut:
    """Append a timestamped forensic note to the alert. Append-only — no overwrite, no delete."""
    from datetime import datetime, timezone

    alert = (await db.execute(
        select(Alert)
        .where(Alert.id == alert_id)
        .options(selectinload(Alert.person), selectinload(Alert.camera), selectinload(Alert.incident))
    )).scalars().first()

    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    if alert.incident and alert.incident.status == "closed":
        raise HTTPException(status_code=409, detail="Case is closed — notes cannot be appended")

    now_utc = datetime.now(tz=timezone.utc)
    stamp = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    new_line = f"[{stamp}] {body.note.strip()}\n"
    alert.operator_notes = (alert.operator_notes or "") + new_line
    alert.updated_at = now_utc
    await db.commit()
    inc_status = alert.incident.status if alert.incident else None
    return _alert_out(alert, incident_status=inc_status)


@router.patch("/alerts/{alert_id}/level", response_model=AlertOut)
async def update_alert_level(alert_id: int, body: AlertLevelUpdate, db: DbSession) -> AlertOut:
    """Operator level promotion — Phase 11 ladder: sighting → candidate → verified → critical."""
    from datetime import datetime, timezone

    alert = (await db.execute(
        select(Alert)
        .where(Alert.id == alert_id)
        .options(selectinload(Alert.person), selectinload(Alert.camera))
    )).scalars().first()

    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.level = body.level
    alert.updated_at = datetime.now(tz=timezone.utc)
    await db.commit()
    return _alert_out(alert)

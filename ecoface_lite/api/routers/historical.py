"""Historical search router (VSL Phase 4).

POST /incidents/{id}/historical-search
  → queues async job, returns job_id for polling
  → sightings created with source="historical" (never in live alert feed)

GET /incidents/{id}/historical-sightings
  → list sightings where source="historical" for this incident
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from ecoface_lite.api.deps import DbSession
from ecoface_lite.api.schemas import AsyncVideoJobResponse, HistoricalSearchRequest
from ecoface_lite.core.config import get_settings
from ecoface_lite.db.models import Incident, Sighting
from ecoface_lite.services import processing_status_service
from ecoface_lite.services.historical_search import submit_historical_search
from ecoface_lite.services.video_service import safe_video_path

router = APIRouter(tags=["historical-search"])


@router.post("/incidents/{incident_id}/historical-search", response_model=AsyncVideoJobResponse, status_code=202)
async def start_historical_search(
    incident_id: int,
    body: HistoricalSearchRequest,
    db: DbSession,
) -> AsyncVideoJobResponse:
    """Queue a historical footage search for this incident.

    The job runs recognition over the specified video window and creates
    Sighting rows with source='historical'. These never appear in the live
    alert feed — they surface in the case history view only.
    """
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    if incident.status == "closed":
        raise HTTPException(status_code=409, detail="Incident is closed — historical search not allowed")

    settings = get_settings()
    try:
        video_path = safe_video_path(settings, body.video_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found under configured videos directory")

    # Single job_id shared between the status row and the background task
    job_id = uuid.uuid4().hex
    await processing_status_service.create_job(
        db,
        job_id=job_id,
        video_label=f"[historical] {body.video_path} | incident={incident_id}",
    )
    await db.commit()

    # Submit background task — live pipeline is unaffected
    submit_historical_search(
        incident_id=incident_id,
        video_path=video_path,
        start_time=body.start_time,
        end_time=body.end_time,
        video_epoch=body.video_epoch,
        frame_skip=body.frame_skip,
        source_id=f"incident-{incident_id}",
        job_id=job_id,
    )

    return AsyncVideoJobResponse(
        job_id=job_id,
        status="queued",
        status_url=f"/api/v1/videos/processing-status/{job_id}",
    )


@router.get("/incidents/{incident_id}/historical-sightings")
async def list_historical_sightings(incident_id: int, db: DbSession) -> list[dict]:
    """List sightings from historical search jobs for this incident."""
    result = await db.execute(
        select(Sighting)
        .where(Sighting.incident_id == incident_id)
        .where(Sighting.source == "historical")
        .order_by(Sighting.frame_index)
    )
    sightings = result.scalars().all()
    return [
        {
            "id": s.id,
            "person_id": s.person_id,
            "confidence": s.confidence,
            "frame_index": s.frame_index,
            "snapshot_path": s.snapshot_path,
            "status": s.status,
            "source": s.source,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in sightings
    ]

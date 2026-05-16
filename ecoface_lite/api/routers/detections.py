from __future__ import annotations

from fastapi import APIRouter, Query

from ecoface_lite.api.deps import DbSession
from ecoface_lite.api.schemas import DetectionOut
from ecoface_lite.services import detection_service

router = APIRouter(prefix="/detections", tags=["detections"])


@router.get("", response_model=list[DetectionOut])
async def list_detections(db: DbSession, limit: int = Query(default=200, ge=1, le=500)) -> list[DetectionOut]:
    rows = await detection_service.list_recent_detections(db, limit=limit)
    return [
        DetectionOut(
            id=r.id,
            person_id=r.person_id,
            person_name=r.person.display_name if r.person else None,
            confidence=r.confidence,
            threshold_used=r.threshold_used,
            source_type=r.source_type,
            source_label=r.source_label,
            frame_index=r.frame_index,
            snapshot_path=r.snapshot_path,
            created_at=r.created_at,
        )
        for r in rows
    ]

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from ecoface_lite.api.deps import DbSession, RecognitionPipelineDep
from ecoface_lite.api.schemas import LiveTestMatchResponse
from ecoface_lite.core.config import get_settings
from ecoface_lite.services import live_test_service

router = APIRouter(tags=["live-test"])


@router.post("/test-match", response_model=LiveTestMatchResponse)
async def run_test_match(
    db: DbSession,
    pipeline: RecognitionPipelineDep,
    image: UploadFile = File(...),
) -> LiveTestMatchResponse:
    settings = get_settings()
    raw = await image.read()
    if len(raw) > settings.max_image_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        result = await live_test_service.test_match_image(
            db,
            pipeline,
            settings,
            file_bytes=raw,
            persist_event=True,
            source_label=image.filename or "laptop_camera",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return LiveTestMatchResponse(
        matched=result.matched,
        person_id=result.person_id,
        person_name=result.person_name,
        similarity_score=result.similarity_score,
        threshold=result.threshold,
        detail=result.detail,
        snapshot_path=result.snapshot_path,
    )

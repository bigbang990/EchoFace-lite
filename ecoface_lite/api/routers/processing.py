from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from ecoface_lite.api.deps import DbSession, RecognitionPipelineDep
from ecoface_lite.api.schemas import AsyncVideoJobResponse, ProcessingStatusOut, VideoProcessRequest
from ecoface_lite.core.config import get_settings
from ecoface_lite.core.logging import get_logger
from ecoface_lite.services import job_queue, video_service

logger = get_logger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("/process")
async def process_video(
    db: DbSession,
    pipeline: RecognitionPipelineDep,
    body: VideoProcessRequest,
) -> dict[str, int | str]:
    settings = get_settings()
    try:
        return await video_service.process_prerecorded_video(
            db,
            pipeline,
            settings,
            video_relative_path=body.video_relative_path,
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/process/async", response_model=AsyncVideoJobResponse)
async def process_video_async(
    db: DbSession,
    body: VideoProcessRequest,
) -> AsyncVideoJobResponse:
    """Queue video processing and poll `GET /videos/processing-status/{job_id}`."""
    job_id = await video_service.start_async_video_job_record(db, video_relative_path=body.video_relative_path)
    await db.commit()
    logger.info("Persisted processing_status job_id=%s (async video queued)", job_id)
    job_queue.submit_video_job(job_id, body.video_relative_path)
    return AsyncVideoJobResponse(job_id=job_id, status_url=f"/api/v1/videos/processing-status/{job_id}")


@router.post("/upload-and-process", response_model=AsyncVideoJobResponse)
async def upload_and_process_video(
    db: DbSession,
    video: UploadFile = File(...),
) -> AsyncVideoJobResponse:
    settings = get_settings()
    relative_path = await video_service.save_uploaded_video(video, settings)
    job_id = await video_service.start_async_video_job_record(db, video_relative_path=relative_path)
    await db.commit()
    logger.info("Persisted processing_status job_id=%s upload=%s", job_id, relative_path)
    job_queue.submit_video_job(job_id, relative_path)
    return AsyncVideoJobResponse(job_id=job_id, status_url=f"/api/v1/videos/processing-status/{job_id}")


@router.get("/processing-status/{job_id}", response_model=ProcessingStatusOut)
async def get_processing_status(job_id: str, db: DbSession) -> ProcessingStatusOut:
    jid = job_id.strip()
    row = await video_service.get_processing_status_row(db, jid)
    if row is None:
        logger.warning("processing_status miss job_id=%r", jid)
        raise HTTPException(status_code=404, detail="Job not found")
    return ProcessingStatusOut.model_validate(row)


@router.get("/processing-preview/{job_id}")
async def get_processing_preview(job_id: str) -> dict[str, str | None]:
    settings = get_settings()
    jid = job_id.strip()
    relative = video_service.latest_preview_relative_path(jid, settings)
    if relative is None:
        return {"preview_path": None, "preview_url": None}
    return {"preview_path": relative, "preview_url": f"/{relative.replace(chr(92), '/')}"}


@router.get("/processing-rejected-faces/{job_id}")
async def get_processing_rejected_faces(job_id: str, limit: int = 30) -> dict[str, object]:
    settings = get_settings()
    jid = job_id.strip()
    rows = video_service.list_rejected_face_debug_images(jid, settings, limit=max(1, min(limit, 100)))
    return {"items": rows}

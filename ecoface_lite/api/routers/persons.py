from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ecoface_lite.api.deps import DbSession, RecognitionPipelineDep
from ecoface_lite.api.schemas import PersonEnrollOut, PersonOut
from ecoface_lite.core.config import get_settings
from ecoface_lite.services import person_service

router = APIRouter(prefix="/persons", tags=["persons"])


@router.get("", response_model=list[PersonOut])
async def list_persons(db: DbSession) -> list[PersonOut]:
    persons = await person_service.list_persons(db)
    return [PersonOut.model_validate(p) for p in persons]


@router.post("", response_model=PersonEnrollOut)
async def create_person(
    db: DbSession,
    pipeline: RecognitionPipelineDep,
    display_name: str = Form(...),
    notes: str | None = Form(default=None),
    image: UploadFile = File(...),
) -> PersonEnrollOut:
    settings = get_settings()
    raw = await image.read()
    if len(raw) > settings.max_image_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        person, deduplicated = await person_service.create_person_from_image(
            db,
            pipeline,
            settings,
            file_bytes=raw,
            original_filename=image.filename or "upload.jpg",
            display_name=display_name,
            notes=notes,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return PersonEnrollOut(person=PersonOut.model_validate(person), deduplicated=deduplicated)

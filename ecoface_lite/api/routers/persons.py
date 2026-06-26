from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import select

from ecoface_lite.api.deps import DbSession, RecognitionPipelineDep
from ecoface_lite.api.schemas import PersonEnrollMultiOut, PersonEnrollOut, PersonOut, PhotoValidationResult, BatchValidationOut
from ecoface_lite.core.config import get_settings
from ecoface_lite.db.models import Person
from ecoface_lite.services import person_service

router = APIRouter(prefix="/persons", tags=["persons"])


@router.post("/validate-batch", response_model=BatchValidationOut)
async def validate_person_photos(
    pipeline: RecognitionPipelineDep,
    images: list[UploadFile] = File(...),
) -> BatchValidationOut:
    """Pre-flight validation for multi-photo enrollment.

    Accepts 1-8 photos, runs face detection + quality check + identity clustering
    on each, and returns per-photo results with outlier flags.

    No person is created and nothing is written to the database.
    Outlier detection fires when >=3 photos are accepted (identity mismatch check).
    """
    settings = get_settings()
    if len(images) > 8:
        raise HTTPException(status_code=400, detail="Max 8 photos per validation call")

    raw_files: list[bytes] = []
    filenames: list[str] = []
    for image in images:
        raw = await image.read()
        if len(raw) > settings.max_image_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{image.filename}: image too large")
        raw_files.append(raw)
        filenames.append(image.filename or "upload.jpg")

    validations = await person_service.validate_enrollment_batch(
        pipeline, raw_files, filenames
    )

    photos_out = [
        PhotoValidationResult(
            index=v.index,
            filename=v.filename,
            status=v.status,
            reason=v.reason,
            is_outlier=v.is_outlier,
            thumbnail_b64=v.thumbnail_b64,
        )
        for v in validations
    ]

    valid_count = sum(1 for v in validations if v.status == "ok")
    rejected_count = sum(1 for v in validations if v.status == "rejected")
    outlier_indices = [v.index for v in validations if v.is_outlier]

    return BatchValidationOut(
        photos=photos_out,
        valid_count=valid_count,
        rejected_count=rejected_count,
        outlier_indices=outlier_indices,
        can_proceed=valid_count > 0,
    )


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
    force_create: bool = Form(default=False),
    force_enroll: bool = Query(default=False),
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
            skip_conflict_check=force_create,
            force_enroll=force_enroll,
        )
    except person_service.EnrollmentConflictError as e:
        raise HTTPException(status_code=409, detail={
            "conflict": True,
            "person_id": e.person_id,
            "person_name": e.person_name,
            "incident_id": e.incident_id,
            "incident_ref": e.incident_ref,
            "incident_title": e.incident_title,
            "incident_status": e.incident_status,
            "incident_opened_at": e.incident_opened_at.isoformat() if e.incident_opened_at else None,
            "similarity": round(e.similarity, 4),
        }) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # Extract and store gender from notes at enrollment time
    if person.enrolled_gender is None and person.notes:
        lower = (person.notes or "").lower()
        if "gender: female" in lower:
            person.enrolled_gender = 0
        elif "gender: male" in lower:
            person.enrolled_gender = 1
        await db.flush()
    return PersonEnrollOut(person=PersonOut.model_validate(person), deduplicated=deduplicated)


@router.post("/{person_id}/photos", response_model=PersonEnrollMultiOut)
async def add_person_photos(
    person_id: int,
    db: DbSession,
    pipeline: RecognitionPipelineDep,
    images: list[UploadFile] = File(...),
    force_enroll: bool = Query(default=False),
) -> PersonEnrollMultiOut:
    settings = get_settings()

    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one_or_none()
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    raw_files: list[bytes] = []
    filenames: list[str] = []
    for image in images:
        raw = await image.read()
        if len(raw) > settings.max_image_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{image.filename}: image too large")
        raw_files.append(raw)
        filenames.append(image.filename or "upload.jpg")

    accepted, rejected, reasons = await person_service.add_photos_to_person(
        db, pipeline, settings, person_id, raw_files, filenames, force_enroll=force_enroll
    )
    await db.refresh(person)
    return PersonEnrollMultiOut(
        person=PersonOut.model_validate(person),
        photos_accepted=accepted,
        photos_rejected=rejected,
        rejection_reasons=reasons,
    )

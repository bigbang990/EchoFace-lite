from __future__ import annotations

import json
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import select

from ecoface_lite.api.deps import DbSession, RecognitionPipelineDep
from ecoface_lite.api.schemas import (
    BatchConfirmOut,
    BatchDetectionOut,
    DetectedFaceOut,
    PersonEnrollMultiOut,
    PersonEnrollOut,
    PersonOut,
    PhotoDetectionResult,
)
from ecoface_lite.core.config import get_settings
from ecoface_lite.db.models import Person
from ecoface_lite.services import person_service

router = APIRouter(prefix="/persons", tags=["persons"])


@router.post("/detect-faces", response_model=BatchDetectionOut)
async def detect_faces_in_photos(
    pipeline: RecognitionPipelineDep,
    images: list[UploadFile] = File(...),
) -> BatchDetectionOut:
    """Stage 1 of multi-photo enrollment: detect ALL faces in each uploaded photo.

    Returns one PhotoDetectionResult per input image. The operator then picks ONE
    face per photo (by face_id) in the frontend and submits the selection back via
    POST /persons/confirm-enroll.

    No person is created here. Stateless — no server-side cache.
    """
    settings = get_settings()
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="At least 1 image required")
    if len(images) > 8:
        raise HTTPException(status_code=400, detail="Max 8 photos per call")

    raw_files: list[bytes] = []
    filenames: list[str] = []
    for image in images:
        raw = await image.read()
        if len(raw) > settings.max_image_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{image.filename}: image too large")
        raw_files.append(raw)
        filenames.append(image.filename or "upload.jpg")

    results = await person_service.detect_faces_in_batch(pipeline, raw_files, filenames)

    photos_out = []
    photos_with_faces = 0
    photos_with_no_face = 0
    for r in results:
        photos_out.append(PhotoDetectionResult(
            photo_index=r["photo_index"],
            filename=r["filename"],
            image_hash=r["image_hash"],
            status=r["status"],
            reason=r["reason"],
            photo_thumbnail_b64=r["photo_thumbnail_b64"],
            faces=[DetectedFaceOut(**f) for f in r["faces"]],
        ))
        if r["status"] == "ok":
            photos_with_faces += 1
        elif r["status"] == "no_face":
            photos_with_no_face += 1

    return BatchDetectionOut(
        photos=photos_out,
        total_photos=len(results),
        photos_with_faces=photos_with_faces,
        photos_with_no_face=photos_with_no_face,
    )


@router.post("/confirm-enroll", response_model=PersonEnrollOut)
async def confirm_and_enroll(
    db: DbSession,
    pipeline: RecognitionPipelineDep,
    display_name: str = Form(...),
    notes: str | None = Form(default=None),
    selections_json: str = Form(...),
    images: list[UploadFile] = File(...),
    force_create: bool = Form(default=False),
    skip_outlier_check: bool = Form(default=False),
) -> PersonEnrollOut:
    """Stage 2 of multi-photo enrollment: persist operator selections.

    Re-uploads the same image bytes from detect-faces along with a JSON list of
    selections [{photo_index, image_hash, face_id}, ...]. Backend re-detects faces,
    matches selected face_ids, extracts embeddings, and creates the Person + N
    FaceEmbedding rows in a single transaction.

    Use skip_outlier_check=true to bypass the cross-photo identity check
    (operator has already reviewed and approved any flagged photos in the UI).
    """
    settings = get_settings()
    try:
        selections = json.loads(selections_json)
        if not isinstance(selections, list) or not selections:
            raise ValueError("selections must be a non-empty list")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid selections_json: {e}") from e

    raw_files: list[bytes] = []
    filenames: list[str] = []
    for image in images:
        raw = await image.read()
        if len(raw) > settings.max_image_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{image.filename}: image too large")
        raw_files.append(raw)
        filenames.append(image.filename or "upload.jpg")

    # Re-extract embeddings for each selection + outlier check
    confirmed = await person_service.confirm_face_selections(
        pipeline, raw_files, filenames, selections,
    )

    # If outliers detected and not bypassed, return 409 with details
    if confirmed["outlier_indices"] and not skip_outlier_check:
        clean = [{k: v for k, v in s.items() if not k.startswith("_")} for s in confirmed["selections"]]
        raise HTTPException(status_code=409, detail={
            "outliers_detected": True,
            "outlier_photo_indices": confirmed["outlier_indices"],
            "selections": clean,
            "message": "One or more selected faces appear to be different identities. Review and resubmit with skip_outlier_check=true to proceed anyway.",
        })

    # Per-selection errors
    errors = [s for s in confirmed["selections"] if s.get("_error")]
    if errors and len([s for s in confirmed["selections"] if s.get("_embedding") is not None]) == 0:
        raise HTTPException(status_code=400, detail={
            "all_failed": True,
            "errors": [{"photo_index": e["photo_index"], "error": e["_error"]} for e in errors],
        })

    try:
        person, written = await person_service.create_person_from_selections(
            db, settings,
            display_name=display_name,
            notes=notes,
            files=raw_files,
            filenames=filenames,
            confirmed=confirmed,
            skip_conflict_check=force_create,
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

    # Extract enrolled_gender from notes
    if person.enrolled_gender is None and person.notes:
        lower = (person.notes or "").lower()
        if "gender: female" in lower:
            person.enrolled_gender = 0
        elif "gender: male" in lower:
            person.enrolled_gender = 1
        await db.flush()

    return PersonEnrollOut(person=PersonOut.model_validate(person), deduplicated=False)


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
    """Legacy single-photo enrollment. New code should use detect-faces + confirm-enroll."""
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

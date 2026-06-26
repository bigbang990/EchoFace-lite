"""Business logic for persons and embeddings."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ecoface_lite.core.config import Settings
from ecoface_lite.core.logging import get_logger
from ecoface_lite.db.models import FaceEmbedding, Person

if TYPE_CHECKING:
    import numpy as np

    from ecoface_lite.ai_engine.pipeline import RecognitionPipeline

logger = get_logger(__name__)


@dataclass
class EnrollmentConflictError(Exception):
    """Raised when a new enrollment closely matches a person already in an open incident."""
    person_id: int
    person_name: str
    incident_id: int
    incident_ref: str
    incident_title: str
    incident_status: str
    incident_opened_at: datetime
    similarity: float


def _validate_enrollment_image(pipeline: "RecognitionPipeline", image: "np.ndarray") -> None:
    """Raise ValueError with a specific message if the image is unsuitable for enrollment.

    Checks are ordered cheapest-first:
      0 faces  → reject (no signal)
      >1 faces → reject (ambiguous identity — operator must crop to one face)
    Quality gate happens inside enroll_reference_embedding after this passes.
    """
    n = pipeline.count_enrollment_faces(image)
    if n == 0:
        raise ValueError("No face detected in reference photo")
    if n > 1:
        raise ValueError(f"Multiple faces detected ({n}) — crop to one face per photo")


def sha256_hex(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


async def _find_person_by_ingest_hash(session: AsyncSession, digest: str) -> Person | None:
    """Match prior enrollment by stored person hash or legacy embedding ingest hash."""
    r1 = await session.execute(select(Person).where(Person.source_image_hash == digest).limit(1))
    found = r1.scalar_one_or_none()
    if found is not None:
        return found

    r2 = await session.execute(
        select(Person)
        .join(FaceEmbedding, FaceEmbedding.person_id == Person.id)
        .where(FaceEmbedding.ingest_sha256 == digest)
        .limit(1)
    )
    return r2.scalar_one_or_none()


async def _check_identity_conflict(
    session: AsyncSession,
    new_embedding: "np.ndarray",
    threshold: float,
) -> None:
    """Raise EnrollmentConflictError if the embedding matches a person in an open incident.

    ArcFace embeddings are L2-normalised so cosine similarity == dot product.
    Only checks persons currently linked to at least one open, non-paused incident.
    """
    import numpy as np

    from ecoface_lite.db.models import Incident, incident_persons

    rows = (await session.execute(
        select(
            FaceEmbedding.person_id,
            FaceEmbedding.embedding,
            Person.display_name,
            Incident.id.label("incident_id"),
            Incident.title.label("incident_title"),
            Incident.status.label("incident_status"),
            Incident.created_at.label("incident_opened_at"),
        )
        .join(Person, Person.id == FaceEmbedding.person_id)
        .join(incident_persons, incident_persons.c.person_id == Person.id)
        .join(Incident, Incident.id == incident_persons.c.incident_id)
        .where(Incident.status == "open")
        .where(Incident.is_paused == False)
    )).all()

    best_sim = 0.0
    best_row = None
    for row in rows:
        vec = np.frombuffer(row.embedding, dtype=np.float32)
        sim = float(np.dot(new_embedding, vec))
        if sim > best_sim:
            best_sim = sim
            best_row = row

    if best_sim >= threshold and best_row is not None:
        inc_id = int(best_row.incident_id)
        raise EnrollmentConflictError(
            person_id=int(best_row.person_id),
            person_name=str(best_row.display_name),
            incident_id=inc_id,
            incident_ref=f"INC-{inc_id:03d}",
            incident_title=str(best_row.incident_title),
            incident_status=str(best_row.incident_status),
            incident_opened_at=best_row.incident_opened_at,
            similarity=best_sim,
        )


async def list_persons(session: AsyncSession) -> list[Person]:
    result = await session.execute(select(Person).order_by(Person.id.desc()))
    return list(result.scalars().all())


async def create_person_from_image(
    session: AsyncSession,
    pipeline: RecognitionPipeline,
    settings: Settings,
    *,
    file_bytes: bytes,
    original_filename: str,
    display_name: str,
    notes: str | None,
    skip_conflict_check: bool = False,
    force_enroll: bool = False,
) -> tuple[Person, bool]:
    """Create a person + embedding, or return an existing person when bytes match a prior upload.

    Returns (person, deduplicated).
    """
    import cv2
    import numpy as np

    digest = sha256_hex(file_bytes)
    existing = await _find_person_by_ingest_hash(session, digest)
    if existing is not None:
        logger.info("Enrollment dedupe hit hash=%s person_id=%s", digest[:12], existing.id)
        return existing, True

    uploads = settings.resolved_uploads_dir()
    uploads.mkdir(parents=True, exist_ok=True)
    ext = Path(original_filename).suffix.lower() or ".jpg"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = uploads / stored_name
    stored_path.write_bytes(file_bytes)

    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid or corrupted image file")

    _validate_enrollment_image(pipeline, image)
    try:
        embedding = pipeline.enroll_reference_embedding(image, enrollment_mode=force_enroll)
    except ValueError:
        raise  # quality rejection — already has a specific message

    if not skip_conflict_check:
        await _check_identity_conflict(session, embedding, settings.enrollment_conflict_threshold)

    rel_upload = str(Path("data/uploads") / stored_name)

    person = Person(
        display_name=display_name,
        notes=notes,
        source_image_path=rel_upload,
        source_image_hash=digest,
    )
    session.add(person)
    await session.flush()

    face = FaceEmbedding(
        person_id=person.id,
        ingest_sha256=digest,
        embedding=embedding.astype(np.float32).tobytes(),
        embedding_dim=int(embedding.shape[0]),
        model_name=settings.insightface_model_name,
    )
    session.add(face)
    await session.flush()
    await session.refresh(person)
    logger.info("Enrolled person id=%s name=%s", person.id, display_name)
    return person, False


async def add_photos_to_person(
    db: AsyncSession,
    pipeline: RecognitionPipeline,
    settings: Settings,
    person_id: int,
    files: list[bytes],
    filenames: list[str],
    force_enroll: bool = False,
) -> tuple[int, int, list[str]]:
    """Enroll additional reference photos for an existing person.

    Returns (accepted_count, rejected_count, rejection_reasons).
    Raises HTTPException 400 if more than 5 photos are submitted.
    """
    import json
    import cv2
    import numpy as np
    from fastapi import HTTPException

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Max 5 photos per call")

    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one_or_none()
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    accepted = 0
    rejected = 0
    reasons: list[str] = []
    new_paths: list[str] = []

    uploads = settings.resolved_uploads_dir()
    uploads.mkdir(parents=True, exist_ok=True)

    for file_bytes, filename in zip(files, filenames):
        digest = sha256_hex(file_bytes)

        dup = await db.execute(
            select(FaceEmbedding)
            .where(FaceEmbedding.person_id == person_id, FaceEmbedding.ingest_sha256 == digest)
            .limit(1)
        )
        if dup.scalar_one_or_none() is not None:
            rejected += 1
            reasons.append(f"{filename}: duplicate (already enrolled for this person)")
            continue

        buf = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            rejected += 1
            reasons.append(f"{filename}: invalid or corrupted image file")
            continue

        try:
            _validate_enrollment_image(pipeline, image)
            embedding = pipeline.enroll_reference_embedding(image, enrollment_mode=force_enroll)
        except ValueError as exc:
            rejected += 1
            reasons.append(f"{filename}: {exc}")
            continue

        # Save image file to uploads dir so it can be shown in the gallery
        ext = Path(filename).suffix.lower() or ".jpg"
        stored_name = f"{uuid.uuid4().hex}{ext}"
        (uploads / stored_name).write_bytes(file_bytes)
        rel_path = str(Path("data/uploads") / stored_name)
        new_paths.append(rel_path)

        face = FaceEmbedding(
            person_id=person_id,
            ingest_sha256=digest,
            embedding=embedding.astype(np.float32).tobytes(),
            embedding_dim=int(embedding.shape[0]),
            model_name=settings.insightface_model_name,
        )
        db.add(face)
        await db.flush()
        accepted += 1
        logger.info("Added photo for person id=%s hash=%s path=%s", person_id, digest[:12], rel_path)

    if new_paths:
        existing = json.loads(person.extra_photo_paths) if person.extra_photo_paths else []
        person.extra_photo_paths = json.dumps(existing + new_paths)
        await db.flush()

    return accepted, rejected, reasons


# ── Pre-flight batch validation (no DB writes) ────────────────────────────────

@dataclass
class PhotoValidation:
    index: int
    filename: str
    status: str           # "ok" | "rejected" | "outlier"
    reason: str | None
    embedding: "np.ndarray | None"
    thumbnail_b64: str | None
    is_outlier: bool = False


def _make_thumbnail_b64(image: "np.ndarray", max_width: int = 120) -> str | None:
    """Return base64-encoded JPEG thumbnail of the image, or None on failure."""
    try:
        import base64
        import cv2
        import numpy as np
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / w
            image = cv2.resize(image, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode()
    except Exception:
        return None


async def validate_enrollment_batch(
    pipeline: "RecognitionPipeline",
    files: list[bytes],
    filenames: list[str],
    outlier_threshold: float = 0.30,
) -> list[PhotoValidation]:
    """Validate N photos for enrollment without writing to DB.

    For each photo:
      - Decode image
      - Count faces using enrollment-grade threshold (0.50 det_score)
      - Extract embedding via enroll_reference_embedding (enrollment_mode=True)
      - Generate a thumbnail for inline preview

    After all photos are processed, run cross-photo outlier detection:
      - Compute pairwise cosine similarity between all accepted embeddings
      - Flag any photo whose mean similarity to all others is below outlier_threshold

    Returns a list of PhotoValidation results (one per input file, same order).
    Outlier detection only fires when >=3 photos are accepted.
    """
    import cv2
    import numpy as np

    results: list[PhotoValidation] = []

    for idx, (file_bytes, filename) in enumerate(zip(files, filenames)):
        buf = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            results.append(PhotoValidation(
                index=idx, filename=filename, status="rejected",
                reason="invalid or corrupted image file",
                embedding=None, thumbnail_b64=None,
            ))
            continue

        thumbnail = _make_thumbnail_b64(image)

        try:
            n = pipeline.count_enrollment_faces(image, min_det_score=0.65)
            if n == 0:
                results.append(PhotoValidation(
                    index=idx, filename=filename, status="rejected",
                    reason="no face detected",
                    embedding=None, thumbnail_b64=thumbnail,
                ))
                continue
            if n > 1:
                results.append(PhotoValidation(
                    index=idx, filename=filename, status="rejected",
                    reason=f"multiple faces detected ({n}) — crop to one face per photo",
                    embedding=None, thumbnail_b64=thumbnail,
                ))
                continue

            embedding = pipeline.enroll_reference_embedding(image, enrollment_mode=True)
            results.append(PhotoValidation(
                index=idx, filename=filename, status="ok",
                reason=None, embedding=embedding, thumbnail_b64=thumbnail,
            ))
        except ValueError as exc:
            results.append(PhotoValidation(
                index=idx, filename=filename, status="rejected",
                reason=str(exc).replace("Face quality rejected for enrollment: ", ""),
                embedding=None, thumbnail_b64=thumbnail,
            ))

    # Outlier detection — only meaningful with >=3 accepted photos
    ok_results = [r for r in results if r.status == "ok" and r.embedding is not None]
    if len(ok_results) >= 3:
        embeddings = np.stack([r.embedding.astype(np.float32) for r in ok_results])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-6)
        sim_matrix = embeddings @ embeddings.T
        n_ok = len(ok_results)
        for i, r in enumerate(ok_results):
            others = [sim_matrix[i, j] for j in range(n_ok) if j != i]
            mean_sim = float(np.mean(others)) if others else 1.0
            if mean_sim < outlier_threshold:
                r.is_outlier = True
                r.status = "outlier"
                r.reason = (
                    f"identity mismatch — this photo looks different from the others "
                    f"(similarity {mean_sim:.2f} vs threshold {outlier_threshold:.2f})"
                )

    return results

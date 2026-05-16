"""Business logic for persons and embeddings."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ecoface_lite.core.config import Settings
from ecoface_lite.core.logging import get_logger
from ecoface_lite.db.models import FaceEmbedding, Person

if TYPE_CHECKING:
    from ecoface_lite.ai_engine.pipeline import RecognitionPipeline

logger = get_logger(__name__)


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
        raise ValueError("Could not decode image bytes")

    embedding = pipeline.enroll_reference_embedding(image)
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

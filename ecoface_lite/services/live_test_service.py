"""Direct still-image recognition for local live testing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ecoface_lite.core.config import Settings
from ecoface_lite.core.logging import get_logger
from ecoface_lite.db.models import DetectionEvent, FaceEmbedding, Person

if TYPE_CHECKING:
    import numpy as np

    from ecoface_lite.ai_engine.pipeline import RecognitionPipeline

logger = get_logger(__name__)


@dataclass(frozen=True)
class NamedGalleryEmbedding:
    person_id: int
    person_name: str
    embedding: Any


@dataclass(frozen=True)
class LiveMatchResult:
    matched: bool
    person_id: int | None
    person_name: str | None
    similarity_score: float | None
    threshold: float
    detail: str
    snapshot_path: str | None = None


async def load_named_gallery(session: AsyncSession) -> list[NamedGalleryEmbedding]:
    import numpy as np

    stmt = (
        select(FaceEmbedding.person_id, Person.display_name, FaceEmbedding.embedding)
        .join(Person, Person.id == FaceEmbedding.person_id)
        .order_by(FaceEmbedding.id.asc())
    )
    rows = (await session.execute(stmt)).all()
    gallery: list[NamedGalleryEmbedding] = []
    for person_id, display_name, blob in rows:
        gallery.append(
            NamedGalleryEmbedding(
                person_id=person_id,
                person_name=display_name,
                embedding=np.frombuffer(blob, dtype=np.float32).copy(),
            )
        )
    return gallery


def decode_camera_image_to_bgr(file_bytes: bytes) -> Any:
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(BytesIO(file_bytes)) as img:
        rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


async def test_match_image(
    session: AsyncSession,
    pipeline: RecognitionPipeline,
    settings: Settings,
    *,
    file_bytes: bytes,
    persist_event: bool = False,
    source_label: str = "laptop_camera",
) -> LiveMatchResult:
    gallery_rows = await load_named_gallery(session)
    threshold = settings.match_confidence_threshold
    if not gallery_rows:
        return LiveMatchResult(False, None, None, None, threshold, "No enrolled faces found", None)

    frame_bgr = decode_camera_image_to_bgr(file_bytes)
    gallery = [(row.person_id, row.embedding) for row in gallery_rows]
    match = pipeline.test_match_frame(frame_bgr, gallery)
    if match is None:
        return LiveMatchResult(False, None, None, None, threshold, "No detectable face or no gallery match", None)

    person_name_by_id = {row.person_id: row.person_name for row in gallery_rows}
    person_name = person_name_by_id.get(match.person_id)
    logger.info(
        "Live test top similarity person_id=%s name=%s score=%.4f threshold=%.4f",
        match.person_id,
        person_name,
        match.confidence,
        threshold,
    )
    is_match = match.confidence >= threshold
    snapshot_path = None
    if is_match and persist_event:
        snapshot_path = await persist_live_detection_event(
            session,
            settings,
            frame_bgr=frame_bgr,
            person_id=match.person_id,
            confidence=match.confidence,
            threshold=threshold,
            source_label=source_label,
        )
    return LiveMatchResult(
        matched=is_match,
        person_id=match.person_id if is_match else None,
        person_name=person_name if is_match else None,
        similarity_score=match.confidence,
        threshold=threshold,
        detail="Match found" if is_match else "Top candidate below threshold",
        snapshot_path=snapshot_path,
    )


async def persist_live_detection_event(
    session: AsyncSession,
    settings: Settings,
    *,
    frame_bgr: Any,
    person_id: int,
    confidence: float,
    threshold: float,
    source_label: str,
) -> str | None:
    recent_cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.live_event_dedupe_seconds)
    if settings.live_event_dedupe_seconds > 0:
        existing = await session.execute(
            select(DetectionEvent.id)
            .where(DetectionEvent.person_id == person_id)
            .where(DetectionEvent.source_type == "webcam")
            .where(DetectionEvent.created_at >= recent_cutoff)
            .limit(1)
        )
        if existing.scalar_one_or_none() is not None:
            return None

    import cv2

    name = f"{uuid4().hex}.jpg"
    snap_path = settings.resolved_snapshots_dir() / name
    cv2.imwrite(str(snap_path), frame_bgr)
    rel_snap = str(Path("data/snapshots") / name)
    session.add(
        DetectionEvent(
            person_id=person_id,
            confidence=confidence,
            threshold_used=threshold,
            source_type="webcam",
            source_label=source_label,
            snapshot_path=rel_snap,
        )
    )
    await session.flush()
    return rel_snap

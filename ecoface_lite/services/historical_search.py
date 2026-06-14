"""Historical footage search (VSL Phase 4).

Pipeline:
  POST /incidents/{id}/historical-search
    → HistoricalSearchJob queued as asyncio background task
    → VideoFileSource.get_historical_stream(start_time, end_time)
    → RecognitionPipeline.process_frame()
    → Sighting(source="historical", alert_id=None)   — no live alert routing
    → ProcessingStatus updated every N frames

Sightings tagged source="historical" never pass through the alert session engine.
They surface only in the Case Management UI under the incident's case history.
The live alert feed is unaffected while a historical job is running.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from sqlalchemy.ext.asyncio import AsyncSession

from ecoface_lite.core.config import Settings, get_settings
from ecoface_lite.core.logging import get_logger
from ecoface_lite.db.session import get_session_factory
from ecoface_lite.services import processing_status_service

logger = get_logger(__name__)


async def _run_historical_search(
    *,
    job_id: str,
    incident_id: int,
    video_path: Path,
    start_time: datetime,
    end_time: datetime,
    source_id: str,
    video_epoch: datetime | None,
    frame_skip: int,
    settings: Settings,
) -> None:
    """Background coroutine — runs recognition on a historical footage window."""
    import cv2

    from ecoface_lite.ai_engine.bootstrap import get_recognition_pipeline
    from ecoface_lite.db.models import DetectionEvent, Sighting
    from ecoface_lite.input_sources.video_file import VideoFileSource
    from ecoface_lite.services.video_service import load_gallery

    session_factory = get_session_factory()
    pipeline = get_recognition_pipeline()

    try:
        async with session_factory() as session:
            gallery = await load_gallery(session)

        if not gallery:
            async with session_factory() as session:
                await processing_status_service.mark_failed(
                    session, job_id, "No enrolled persons — historical search aborted"
                )
                await session.commit()
            return

        source = VideoFileSource(
            path=video_path,
            frame_skip=max(1, frame_skip),
            source_id=source_id,
            video_epoch=video_epoch,
        )

        sighting_count = 0
        frame_count = 0
        started_at = perf_counter()

        for frame in source.get_historical_stream(start_time, end_time):
            frame_count += 1
            matches = pipeline.process_frame(frame.bgr, frame.index, gallery)

            for m in matches:
                if m.person_id is None or m.confidence is None or not m.should_alert:
                    continue

                async with session_factory() as session:
                    det = DetectionEvent(
                        person_id=m.person_id,
                        confidence=m.confidence,
                        threshold_used=settings.match_confidence_threshold,
                        source_type="historical",
                        source_label=str(video_path.name),
                        frame_index=frame.index,
                    )
                    session.add(det)
                    await session.flush()

                    sighting = Sighting(
                        incident_id=incident_id,
                        detection_id=det.id,
                        person_id=m.person_id,
                        confidence=m.confidence,
                        frame_index=frame.index,
                        source="historical",       # key: keeps out of live alert feed
                        alert_id=None,             # never routed through alert engine
                        status="pending",
                    )
                    session.add(sighting)
                    await session.commit()
                    sighting_count += 1

            # Progress every 50 frames
            if frame_count % 50 == 0:
                async with session_factory() as session:
                    await processing_status_service.set_progress(
                        session, job_id, processed_frames=frame_count, alerts_created=sighting_count
                    )
                    await session.commit()

            await asyncio.sleep(0)  # yield to event loop — live pipeline unaffected

        elapsed = perf_counter() - started_at
        async with session_factory() as session:
            await processing_status_service.mark_completed(
                session, job_id,
                processed_frames=frame_count,
                alerts_created=sighting_count,
            )
            await session.commit()

        logger.info(
            "Historical search job=%s complete: %d frames, %d sightings in %.1fs",
            job_id, frame_count, sighting_count, elapsed,
        )

    except Exception as exc:
        logger.exception("Historical search job=%s failed: %s", job_id, exc)
        try:
            async with session_factory() as session:
                await processing_status_service.mark_failed(session, job_id, str(exc))
                await session.commit()
        except Exception:
            pass


def submit_historical_search(
    *,
    incident_id: int,
    video_path: Path,
    start_time: datetime,
    end_time: datetime,
    video_epoch: datetime | None = None,
    frame_skip: int = 1,
    source_id: str = "historical",
    job_id: str | None = None,
) -> str:
    """Queue a historical search as an asyncio background task. Returns job_id.

    Pass job_id when the caller has already created a ProcessingStatus row;
    omit it to auto-generate (job row must then be created inside the coroutine).
    """
    job_id = job_id or uuid.uuid4().hex
    settings = get_settings()

    asyncio.create_task(
        _run_historical_search(
            job_id=job_id,
            incident_id=incident_id,
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            source_id=source_id,
            video_epoch=video_epoch,
            frame_skip=frame_skip,
            settings=settings,
        ),
        name=f"historical_search_{job_id[:8]}",
    )
    return job_id

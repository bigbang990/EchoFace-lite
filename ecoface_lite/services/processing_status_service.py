"""Persistence helpers for `processing_status` rows (video job progress)."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ecoface_lite.db.models import ProcessingStatus


async def create_job(
    session: AsyncSession,
    *,
    job_id: str,
    video_label: str | None,
) -> ProcessingStatus:
    row = ProcessingStatus(
        job_id=job_id,
        video_label=video_label,
        total_frames=0,
        processed_frames=0,
        status="queued",
    )
    session.add(row)
    await session.flush()
    return row


async def get_job_by_id(session: AsyncSession, job_id: str) -> ProcessingStatus | None:
    stmt = select(ProcessingStatus).where(ProcessingStatus.job_id == job_id)
    return (await session.execute(stmt)).scalar_one_or_none()


async def set_total_frames_and_running(
    session: AsyncSession,
    job_id: str,
    *,
    total_frames: int,
) -> None:
    await session.execute(
        update(ProcessingStatus)
        .where(ProcessingStatus.job_id == job_id)
        .values(total_frames=total_frames, status="running", processed_frames=0)
    )


async def set_processed_frames(session: AsyncSession, job_id: str, processed_frames: int) -> None:
    await session.execute(
        update(ProcessingStatus)
        .where(ProcessingStatus.job_id == job_id)
        .values(processed_frames=processed_frames)
    )


async def set_progress(
    session: AsyncSession,
    job_id: str,
    *,
    processed_frames: int,
    alerts_created: int,
) -> None:
    await session.execute(
        update(ProcessingStatus)
        .where(ProcessingStatus.job_id == job_id)
        .values(processed_frames=processed_frames, alerts_created=alerts_created)
    )


async def mark_completed(
    session: AsyncSession,
    job_id: str,
    *,
    processed_frames: int,
    alerts_created: int,
    analytics: dict[str, float | int] | None = None,
) -> None:
    values = {
        "processed_frames": processed_frames,
        "alerts_created": alerts_created,
        "status": "completed",
        "error_message": None,
    }
    if analytics:
        values.update(analytics)
    await session.execute(
        update(ProcessingStatus)
        .where(ProcessingStatus.job_id == job_id)
        .values(**values)
    )


async def mark_failed(session: AsyncSession, job_id: str, message: str) -> None:
    await session.execute(
        update(ProcessingStatus)
        .where(ProcessingStatus.job_id == job_id)
        .values(status="failed", error_message=message[:4000])
    )

"""Detection history reads."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ecoface_lite.db.models import DetectionEvent


async def list_recent_detections(session: AsyncSession, *, limit: int = 200) -> list[DetectionEvent]:
    stmt = (
        select(DetectionEvent)
        .options(selectinload(DetectionEvent.person))
        .order_by(DetectionEvent.id.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())

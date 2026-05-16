"""FastAPI dependencies — DB session lifecycle and shared AI pipeline."""

from __future__ import annotations

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ecoface_lite.db.session import get_session_factory


def get_recognition_pipeline():
    """Lazy import so `/health` and tests do not require NumPy/InsightFace at import time."""
    from ecoface_lite.ai_engine.bootstrap import get_recognition_pipeline as _get_pipeline

    return _get_pipeline()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DbSession = Annotated[AsyncSession, Depends(get_db)]
RecognitionPipelineDep = Annotated["RecognitionPipeline", Depends(get_recognition_pipeline)]

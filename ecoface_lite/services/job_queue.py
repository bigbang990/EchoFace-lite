"""In-process background job dispatcher for laptop-friendly local video processing."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from ecoface_lite.core.logging import get_logger
from ecoface_lite.services import video_service

logger = get_logger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ecoface-video-worker")


def submit_video_job(job_id: str, video_relative_path: str) -> None:
    future = _executor.submit(_run_video_job_in_worker_thread, job_id, video_relative_path)
    future.add_done_callback(_log_worker_failure)


def _run_video_job_in_worker_thread(job_id: str, video_relative_path: str) -> None:
    asyncio.run(video_service.run_async_video_job(job_id, video_relative_path))


def _log_worker_failure(future) -> None:
    exc = future.exception()
    if exc is not None:
        logger.exception("Video worker crashed", exc_info=exc)

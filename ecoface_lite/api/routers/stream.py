"""Live MJPEG streaming endpoints.

GET /stream/live/{camera_id}   multipart/x-mixed-replace — live camera via LiveCameraSession
GET /stream/job/{job_id}       multipart/x-mixed-replace — file/RTSP processing job preview
GET /debug/stream-metrics      lightweight rolling counters for active sessions
"""

from __future__ import annotations

import asyncio
import pathlib

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from ecoface_lite.api.deps import DbSession
from ecoface_lite.core.config import get_settings
from ecoface_lite.core.logging import get_logger
from ecoface_lite.core.metrics import metrics
from ecoface_lite.db.models import Camera
from ecoface_lite.services.live_camera_session import get_live_session_manager

logger = get_logger(__name__)

router = APIRouter(tags=["stream"])

_BOUNDARY_PREFIX = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
_BOUNDARY_SUFFIX = b"\r\n"
_TARGET_FPS_SLEEP = 0.033  # ~30 FPS ceiling


@router.get("/stream/live/{camera_id}")
async def stream_live(camera_id: int, db: DbSession):
    from ecoface_lite.ai_engine.bootstrap import get_recognition_pipeline

    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not camera.stream_url:
        raise HTTPException(status_code=422, detail="Camera has no stream_url configured")

    manager = get_live_session_manager()
    pipeline = get_recognition_pipeline()
    try:
        session = manager.acquire(camera_id, camera.stream_url, pipeline)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    async def _frame_generator():
        try:
            while True:
                frame = session.get_annotated_frame()
                if frame is None:
                    await asyncio.sleep(_TARGET_FPS_SLEEP)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    await asyncio.sleep(_TARGET_FPS_SLEEP)
                    continue
                yield _BOUNDARY_PREFIX + buf.tobytes() + _BOUNDARY_SUFFIX
                await asyncio.sleep(_TARGET_FPS_SLEEP)
        finally:
            # G3 — runs on client disconnect, generator close, or any exception.
            manager.release(camera_id)

    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/stream/job/{job_id}")
async def stream_job_annotated_frames(job_id: str):
    """MJPEG stream for any file-upload or camera processing job.

    Polls {PREVIEWS_DIR}/{job_id}/latest.jpg at ~10 FPS and yields each
    new frame as a MJPEG boundary when mtime or size changes.  Keeps the
    last good frame on screen while the pipeline is slower than the poll
    interval, and terminates after ~60 s with no file present.
    """
    settings = get_settings()
    preview_path = pathlib.Path(settings.resolved_previews_dir()) / job_id / "latest.jpg"

    async def _generator():
        last_mtime: float = 0.0
        last_size: int = 0
        idle_ticks: int = 0
        while True:
            try:
                if preview_path.is_file():
                    st = preview_path.stat()
                    if st.st_mtime != last_mtime or st.st_size != last_size:
                        last_mtime = st.st_mtime
                        last_size = st.st_size
                        data = preview_path.read_bytes()
                        if data:
                            yield _BOUNDARY_PREFIX + data + _BOUNDARY_SUFFIX
                    idle_ticks = 0
                else:
                    idle_ticks += 1
                    if idle_ticks > 600:  # ~60 s with no file → close
                        break
            except OSError:
                pass
            await asyncio.sleep(0.1)  # poll at ~10 FPS

    return StreamingResponse(
        _generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/debug/stream-metrics")
async def stream_metrics() -> dict[str, float | bool]:
    manager = get_live_session_manager()
    sessions = manager.all_sessions()
    if not sessions:
        return {"stream_fps": 0.0, "ai_fps": 0.0, "frame_age_ms": -1.0, "ai_busy": False}
    # Single-camera demo scope: report the most recently created active session.
    session = next(iter(sessions.values()))
    snap = metrics.snapshot()
    return {
        "stream_fps": session.stream_fps.fps,
        "ai_fps": session.ai_fps.fps,
        "frame_age_ms": session.frame_age_ms,
        "ai_busy": session.ai_busy,
        "max_queue_size_seen": snap.recent_values.get("max_queue_size_seen", [0.0])[-1],
        "avg_queue_size_seen": snap.averages.get("avg_queue_size_seen", 0.0),
        "queue_full_duration_ms": float(snap.counters.get("queue_full_duration_ms", 0)),
    }

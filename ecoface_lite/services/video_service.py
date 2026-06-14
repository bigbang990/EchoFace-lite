"""Video processing orchestration."""

from __future__ import annotations

import uuid
import json
from time import perf_counter
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ecoface_lite.ai_engine.progress import should_persist_progress
from ecoface_lite.ai_engine.processing_diagnostics import VideoJobDiagnostics
from ecoface_lite.ai_engine.visualization import OverlayItem, VideoPreviewWriter
from ecoface_lite.core.config import Settings, get_settings
from ecoface_lite.core.logging import get_logger
from ecoface_lite.core.metrics import metrics
from ecoface_lite.db.session import get_session_factory
from ecoface_lite.services import processing_status_service
from ecoface_lite.services.alert_session_engine import get_alert_session_engine

if TYPE_CHECKING:
    import numpy as np

    from ecoface_lite.ai_engine.pipeline import RecognitionPipeline

logger = get_logger(__name__)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def safe_video_path(settings: Settings, relative_path: str) -> Path:
    """Resolve a path under VIDEOS_DIR; raises ValueError if traversal or missing base."""
    base = settings.resolved_videos_dir().resolve()
    candidate = (base / relative_path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("Invalid video path") from exc
    return candidate


def _safe_video_path(settings: Settings, relative_path: str) -> Path:
    try:
        return safe_video_path(settings, relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _is_stream_url(path: str) -> bool:
    return path.startswith(("http://", "https://", "rtsp://"))


def count_emitted_frames(video_path: Path, frame_skip: int) -> int:
    """Approximate emitted frame count (matches VideoFileSource logic when CAP_PROP works)."""
    import cv2

    from ecoface_lite.input_sources.video_file import VideoFileSource

    skip = max(1, frame_skip)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n > 0:
        return (n + skip - 1) // skip
    return sum(1 for _ in VideoFileSource(video_path, skip).frames())


async def load_gallery(session: AsyncSession) -> list[tuple[int, "np.ndarray"]]:
    """
    Load embeddings for persons attached to at least one OPEN incident.
    Falls back to ALL enrolled persons if no incidents exist yet
    (preserves backward compatibility for tests and simple demos).
    """
    import numpy as np
    from sqlalchemy import select

    from ecoface_lite.db.models import FaceEmbedding, Incident, incident_persons

    # Check if any incidents exist at all
    has_incidents = (await session.execute(
        select(Incident.id).limit(1)
    )).first() is not None

    if has_incidents:
        # Only load persons attached to at least one OPEN and NOT PAUSED incident
        open_person_ids = (await session.execute(
            select(incident_persons.c.person_id)
            .join(Incident, Incident.id == incident_persons.c.incident_id)
            .where(Incident.status == "open")
            .where(Incident.is_paused == False)
            .distinct()
        )).scalars().all()

        if not open_person_ids:
            return []  # All incidents closed — nothing to search

        stmt = (
            select(FaceEmbedding.person_id, FaceEmbedding.embedding)
            .where(FaceEmbedding.person_id.in_(open_person_ids))
            .order_by(FaceEmbedding.id.asc())
        )
    else:
        # No incident system in use — load everything (backward compat)
        stmt = select(FaceEmbedding.person_id, FaceEmbedding.embedding).order_by(FaceEmbedding.id.asc())

    rows = (await session.execute(stmt)).all()
    gallery: list[tuple[int, np.ndarray]] = []
    for person_id, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32).copy()
        gallery.append((person_id, vec))
    return gallery


async def _persist_progress_if_needed(
    session: AsyncSession,
    job_id: str,
    emitted_count: int,
    alerts: int,
    *,
    every_n: int,
) -> None:
    if not should_persist_progress(emitted_count, every_n=every_n):
        return
    await processing_status_service.set_progress(
        session,
        job_id,
        processed_frames=emitted_count,
        alerts_created=alerts,
    )
    await session.commit()


def _resize_for_inference(frame_bgr: "np.ndarray", target_width: int) -> "np.ndarray":
    if target_width <= 0 or frame_bgr.shape[1] <= target_width:
        return frame_bgr
    import cv2

    ratio = target_width / frame_bgr.shape[1]
    height = max(1, int(frame_bgr.shape[0] * ratio))
    return cv2.resize(frame_bgr, (target_width, height), interpolation=cv2.INTER_AREA)


async def save_uploaded_video(upload: UploadFile, settings: Settings) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video type")
    name = f"{uuid.uuid4().hex}{suffix}"
    target = settings.resolved_videos_dir() / name
    with target.open("wb") as f:
        while chunk := await upload.read(1024 * 1024):
            f.write(chunk)
    logger.info("Saved uploaded video %s", target)
    return name


async def process_prerecorded_video(
    session: AsyncSession,
    pipeline: RecognitionPipeline,
    settings: Settings,
    *,
    video_relative_path: str,
    job_id: str | None = None,
) -> dict[str, int | str]:
    """Process a video; optionally update `processing_status` every N emitted frames."""
    import cv2

    from ecoface_lite.input_sources.video_file import FramePacket, VideoFileSource

    from sqlalchemy import select as _select

    from ecoface_lite.db.models import DetectionEvent, Incident, incident_persons

    if _is_stream_url(video_relative_path):
        _cap = cv2.VideoCapture(video_relative_path)
        if not _cap.isOpened():
            _cap.release()
            raise HTTPException(status_code=422, detail=f"Cannot connect to stream: {video_relative_path}")
        video_path = video_relative_path

        def _make_frame_iter():
            _idx = 0
            _emit = 0
            try:
                while True:
                    ok, frame = _cap.read()
                    if not ok:
                        break
                    if _idx % settings.video_frame_skip == 0:
                        yield FramePacket(index=_emit, bgr=frame)
                        _emit += 1
                    _idx += 1
            finally:
                _cap.release()

        _frame_iter = _make_frame_iter()
    else:
        video_path = _safe_video_path(settings, video_relative_path)
        if not video_path.is_file():
            raise HTTPException(status_code=404, detail="Video file not found under configured videos directory")
        _frame_iter = VideoFileSource(video_path, settings.video_frame_skip).frames()

    gallery = await load_gallery(session)
    if not gallery:
        raise HTTPException(status_code=400, detail="No enrolled persons in gallery")

    alert_engine = get_alert_session_engine()
    alerts = 0
    new_sessions = 0
    emitted_count = 0
    job_diagnostics = VideoJobDiagnostics(job_id=job_id)
    preview_writer = VideoPreviewWriter(settings, job_id)
    last_sighting_frame_by_person: dict[int, int] = {}  # controls sighting write frequency
    started_at = perf_counter()
    for packet in _frame_iter:
        emitted_count += 1
        job_diagnostics.frames_processed = emitted_count
        inference_frame = _resize_for_inference(packet.bgr, settings.video_inference_width)
        frame_matches = pipeline.process_frame(inference_frame, packet.index, gallery)
        _save_rejected_debug_crops(settings, job_id, inference_frame, frame_matches, packet.index, emitted_count)
        if preview_writer.should_write(packet.index):
            preview_writer.write(
                inference_frame,
                _overlay_items_from_matches(frame_matches),
                packet.index,
            )
        for m in frame_matches:
            if m.reason in {"blurry_face", "face_too_small", "low_detection_confidence", "below_adaptive_threshold"}:
                job_diagnostics.faces_rejected += 1
                if m.reason == "blurry_face":
                    job_diagnostics.blur_rejections += 1
                elif m.reason == "face_too_small":
                    job_diagnostics.size_rejections += 1
                elif m.reason in {"low_detection_confidence", "below_adaptive_threshold"}:
                    job_diagnostics.low_confidence_rejections += 1
            if m.reason not in {"no_face_detected"}:
                job_diagnostics.faces_detected += 1
            job_diagnostics.observe_confidence(m.confidence)
            if m.person_id is None or m.confidence is None:
                continue
            if not m.should_alert:
                continue
            # Rate-limit sighting writes: one DB write per video_event_dedupe_frames frames.
            # The alert session tracks last_seen_at in memory between writes.
            previous = last_sighting_frame_by_person.get(m.person_id)
            if previous is not None and packet.index - previous < settings.video_event_dedupe_frames:
                job_diagnostics.duplicate_suppressions += 1
                metrics.increment("duplicate_alerts_suppressed")
                continue
            last_sighting_frame_by_person[m.person_id] = packet.index

            # Save face crop snapshot
            name = f"{uuid.uuid4().hex}.jpg"
            snap_path = settings.resolved_snapshots_dir() / name
            _ih, _iw = inference_frame.shape[:2]
            _x1 = max(0, int(m.face.bbox.x1))
            _y1 = max(0, int(m.face.bbox.y1))
            _x2 = min(_iw, int(m.face.bbox.x2))
            _y2 = min(_ih, int(m.face.bbox.y2))
            _px = max(4, int((_x2 - _x1) * 0.15))
            _py = max(4, int((_y2 - _y1) * 0.15))
            _face_crop = inference_frame[
                max(0, _y1 - _py):min(_ih, _y2 + _py),
                max(0, _x1 - _px):min(_iw, _x2 + _px),
            ]
            cv2.imwrite(str(snap_path), _face_crop)
            rel_snap = str(Path("data/snapshots") / name)

            # Audit log: one DetectionEvent per written frame (source of truth for raw detections)
            det = DetectionEvent(
                person_id=m.person_id,
                confidence=m.confidence,
                threshold_used=m.threshold,
                source_type="video_file",
                source_label=str(video_path),
                frame_index=m.frame_index,
                snapshot_path=rel_snap,
            )
            session.add(det)
            await session.flush()  # assign det.id

            # Only fire alerts for OPEN, non-paused incidents — never match against closed cases
            inc_rows = await session.execute(
                _select(incident_persons.c.incident_id)
                .join(Incident, Incident.id == incident_persons.c.incident_id)
                .where(incident_persons.c.person_id == m.person_id)
                .where(Incident.status == "open")
                .where(Incident.is_paused == False)
            )
            for (inc_id,) in inc_rows.all():
                alert, _sighting = await alert_engine.record_match(
                    session,
                    incident_id=inc_id,
                    person_id=m.person_id,
                    camera_id=None,  # populated by VSL Phase 1 when source carries camera metadata
                    confidence=m.confidence,
                    detection_id=det.id,
                    frame_index=m.frame_index,
                    snapshot_path=rel_snap,
                    source="live",
                )
                if alert is not None and alert.sighting_count == 1:
                    new_sessions += 1
            alerts += 1
            job_diagnostics.alerts_created = new_sessions
            metrics.increment("detection_events_created")
        if job_id:
            await _persist_progress_if_needed(
                session,
                job_id,
                emitted_count,
                alerts,
                every_n=settings.video_progress_interval,
            )

    if job_id:
        await processing_status_service.mark_completed(
            session,
            job_id,
            processed_frames=emitted_count,
            alerts_created=alerts,
            analytics=job_diagnostics.as_analytics(),
        )
        await session.commit()

    duration = perf_counter() - started_at
    metrics.observe("video_job_duration", duration)
    metrics.observe("average_processing_fps", emitted_count / duration if duration > 0 else 0.0)
    metrics.observe("alerts_per_video", alerts)
    logger.info(
        "Processed video %s alerts=%s frames=%s duration=%.3fs fps=%.3f job_id=%s",
        video_path,
        alerts,
        emitted_count,
        duration,
        emitted_count / duration if duration > 0 else 0.0,
        job_id,
    )
    return {"alerts_created": alerts, "video_path": str(video_path)}


def _overlay_items_from_matches(matches) -> list[OverlayItem]:
    items: list[OverlayItem] = []
    for match in matches:
        if match.face is None:
            continue
        trace = match.trace
        state = trace.state if trace is not None else ("green" if match.should_alert else "red")
        size_text = f"{trace.face_width}x{trace.face_height}" if trace is not None else ""
        blur_text = f" | blur {trace.blur_score:.1f}" if trace is not None and trace.blur_score is not None else ""
        track_text = f"T{match.track_id} | " if match.track_id is not None else ""
        if state == "green":
            pct = f"{(match.confidence or 0) * 100:.0f}%"
            label = f"{track_text}{match.person_id} | {pct}"
        elif state == "yellow":
            reason = match.reason or (trace.rejection_reason if trace is not None else "unstable")
            if reason and not reason.startswith("REJECTED:"):
                reason = f"REJECTED: {reason}"
            score = f"{trace.detector_confidence:.2f}" if trace is not None and trace.detector_confidence is not None else "n/a"
            label = f"{track_text}{reason} | det {score}"
        elif match.track_id is not None:
            det = trace.detector_confidence if trace is not None and trace.detector_confidence is not None else 0.0
            label = f"T{match.track_id} | det {det:.2f}"
        elif match.confidence is not None:
            label = f"{track_text}RED | unknown | {match.confidence:.2f}"
        else:
            label = f"{track_text}RED | {match.reason or 'unknown'}"
        items.append(OverlayItem(face=match.face, match=match, label=label, state=state))
    return items


def _save_rejected_debug_crops(
    settings: Settings,
    job_id: str | None,
    frame_bgr: "np.ndarray",
    matches,
    frame_index: int,
    emitted_count: int,
) -> None:
    if emitted_count % max(1, settings.rejected_face_snapshot_interval) != 0:
        return
    import cv2

    target_dir = settings.resolved_rejected_faces_dir() / (job_id or "sync")
    target_dir.mkdir(parents=True, exist_ok=True)
    h, w = frame_bgr.shape[:2]
    for idx, match in enumerate(matches):
        if match.face is None or match.trace is None or match.trace.state != "yellow":
            continue
        x1 = max(0, min(w - 1, int(match.face.bbox.x1)))
        y1 = max(0, min(h - 1, int(match.face.bbox.y1)))
        x2 = max(x1 + 1, min(w, int(match.face.bbox.x2)))
        y2 = max(y1 + 1, min(h, int(match.face.bbox.y2)))
        crop = frame_bgr[y1:y2, x1:x2]
        reason = (match.reason or match.trace.rejection_reason or "rejected").replace("/", "_").replace("\\", "_")
        stem = f"frame_{frame_index:06d}_{idx}_{reason}_{match.trace.face_width}x{match.trace.face_height}_{match.trace.detector_confidence or 0:.2f}"
        image_path = target_dir / f"{stem}.jpg"
        meta_path = target_dir / f"{stem}.json"
        cv2.imwrite(str(image_path), crop)
        meta_path.write_text(
            json.dumps(
                {
                    "frame_index": frame_index,
                    "reason": match.reason or match.trace.rejection_reason,
                    "detector_confidence": match.trace.detector_confidence,
                    "face_width": match.trace.face_width,
                    "face_height": match.trace.face_height,
                    "blur_score": match.trace.blur_score,
                    "stages": list(match.trace.stages),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        metrics.increment("rejected_face_snapshots_saved")


async def start_async_video_job_record(session: AsyncSession, *, video_relative_path: str) -> str:
    """Insert a queued job row and return `job_id` (caller schedules background work)."""
    job_id = uuid.uuid4().hex
    await processing_status_service.create_job(
        session,
        job_id=job_id,
        video_label=video_relative_path,
    )
    return job_id


async def run_async_video_job(job_id: str, video_relative_path: str) -> None:
    """Background task: short DB sessions for job metadata, separate session for heavy CV work.

    Splitting sessions avoids holding a SQLite connection across long GPU/CPU inference,
    which reduces lock contention with the main API thread.
    """
    from ecoface_lite.ai_engine.bootstrap import get_recognition_pipeline

    settings = get_settings()
    factory = get_session_factory()
    pipeline = get_recognition_pipeline()

    async with factory() as meta_session:
        if _is_stream_url(video_relative_path):
            total = 0  # frame count unknown ahead of time for live streams
        else:
            try:
                video_path = safe_video_path(settings, video_relative_path)
            except ValueError as exc:
                await processing_status_service.mark_failed(meta_session, job_id, str(exc))
                await meta_session.commit()
                return

            if not video_path.is_file():
                await processing_status_service.mark_failed(
                    meta_session,
                    job_id,
                    "Video file not found under configured videos directory",
                )
                await meta_session.commit()
                return

            total = count_emitted_frames(video_path, settings.video_frame_skip)
        await processing_status_service.set_total_frames_and_running(
            meta_session,
            job_id,
            total_frames=max(total, 1),
        )
        await meta_session.commit()

    from ecoface_lite.core.runtime_state import clear_session_id, new_session_id

    session_id = new_session_id()
    metrics.reset()
    logger.info("Session started session_id=%s", session_id)
    try:
        async with factory() as proc_session:
            await process_prerecorded_video(
                proc_session,
                pipeline,
                settings,
                video_relative_path=video_relative_path,
                job_id=job_id,
            )
    except HTTPException as exc:
        msg = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        async with factory() as fail_session:
            await processing_status_service.mark_failed(fail_session, job_id, msg)
            await fail_session.commit()
    except Exception as exc:  # noqa: BLE001 — surface last-resort failure to job row
        logger.exception("Async video job failed job_id=%s", job_id)
        async with factory() as fail_session:
            await processing_status_service.mark_failed(fail_session, job_id, str(exc))
            await fail_session.commit()
    finally:
        clear_session_id()
        logger.info("Session ended session_id=%s", session_id)


async def get_processing_status_row(session: AsyncSession, job_id: str):
    return await processing_status_service.get_job_by_id(session, job_id)


def latest_preview_relative_path(job_id: str, settings: Settings) -> str | None:
    path = settings.resolved_previews_dir() / job_id / "latest.jpg"
    if not path.is_file():
        return None
    return str(path.relative_to(settings.project_root))


def list_rejected_face_debug_images(job_id: str, settings: Settings, limit: int = 30) -> list[dict[str, object]]:
    target_dir = settings.resolved_rejected_faces_dir() / job_id
    if not target_dir.is_dir():
        return []
    rows: list[dict[str, object]] = []
    for image_path in sorted(target_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        meta_path = image_path.with_suffix(".json")
        metadata: dict[str, object] = {}
        if meta_path.is_file():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        relative = str(image_path.relative_to(settings.project_root)).replace("\\", "/")
        rows.append({"image_url": f"/{relative}", "image_path": relative, "metadata": metadata})
    return rows

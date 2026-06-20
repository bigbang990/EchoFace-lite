"""Live camera session — decoupled capture + AI inference for MJPEG streaming.

Two daemon threads per session, intentionally decoupled so a slow AI cycle never
blocks frame capture and a slow camera never blocks the stream endpoint:

  - capture thread  — pulls frames from cv2.VideoCapture as fast as the camera
                       delivers them (no sleep). Always writes a fresh `.copy()`
                       into `latest_frame` under `_frame_lock`.
  - AI thread        — when not busy, copies the latest frame and submits it to a
                        single-worker ThreadPoolExecutor that runs
                        `pipeline.process_frame()` (synchronous, never modified —
                        see RecognitionPipeline hard constraint). Back-pressure:
                        if the previous inference hasn't finished, this cycle is
                        skipped entirely (G1). The frame handed to inference is
                        always an isolated copy (G2), never the shared reference.

`get_annotated_frame()` draws the latest `overlay_state` onto a fresh copy of
`latest_frame` for the MJPEG endpoint to encode — never mutates `latest_frame`
itself, so the AI thread's next `.copy()` is unaffected.

Session lifecycle (G3) is owned by `LiveSessionManager` below, which ref-counts
concurrent viewers per camera_id and calls `session.stop()` only when the last
viewer disconnects — so multiple browser tabs share one capture/AI pair instead
of spawning duplicate threads, and `stop()` always runs (no orphan threads) once
nobody is watching.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from ecoface_lite.core.config import get_settings
from ecoface_lite.core.logging import get_logger
from ecoface_lite.core.metrics import metrics
from ecoface_lite.db.session import get_session_factory
from ecoface_lite.services import video_service
from ecoface_lite.services.alert_session_engine import get_alert_session_engine

if TYPE_CHECKING:
    import numpy as np

    from ecoface_lite.ai_engine.pipeline import RecognitionPipeline
    from ecoface_lite.ai_engine.pipeline_types import FrameMatch

logger = get_logger(__name__)

_OVERLAY_COLORS = {
    "matched": (0, 180, 0),     # green — confirmed identity match
    "tracking": (0, 220, 220),  # yellow — tracked, not yet matched/confirmed
    "rejected": (0, 0, 220),    # red — detected but rejected by validator
}


class _RollingRate:
    """Rolling-average tick rate over the last `window` samples (debug metrics)."""

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span


class LiveCameraSession:
    """Owns one camera's capture + AI threads; exposes the latest annotated frame."""

    def __init__(self, camera_id: int, pipeline: "RecognitionPipeline") -> None:
        self.camera_id = camera_id
        self._pipeline = pipeline

        self.latest_frame: "np.ndarray | None" = None
        self._frame_lock = threading.Lock()
        self._frame_captured_at: float | None = None

        self.overlay_state: dict[int, dict[str, Any]] = {}
        self._overlay_lock = threading.Lock()

        self._ai_busy = threading.Event()
        self._running = False

        self._capture_thread: threading.Thread | None = None
        self._ai_thread: threading.Thread | None = None
        self._cap: "cv2.VideoCapture | None" = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"ai-infer-cam{camera_id}"
        )

        self._gallery: list[tuple[int, "np.ndarray"]] = []
        self._frame_index = 0

        # debug metrics (G-adjacent, required by /debug/stream-metrics)
        self.stream_fps = _RollingRate()
        self.ai_fps = _RollingRate()

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self, source_url: str) -> None:
        if self._running:
            return
        cap = cv2.VideoCapture(source_url)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera source: {source_url}")
        self._cap = cap

        try:
            self._gallery = self._load_gallery_sync()
        except Exception:
            logger.exception("LiveCameraSession camera_id=%s: gallery load failed", self.camera_id)
            self._gallery = []

        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name=f"capture-cam{self.camera_id}", daemon=True
        )
        self._ai_thread = threading.Thread(
            target=self._ai_loop, name=f"ai-loop-cam{self.camera_id}", daemon=True
        )
        self._capture_thread.start()
        self._ai_thread.start()
        logger.info("LiveCameraSession started camera_id=%s source=%s", self.camera_id, source_url)

    def stop(self) -> None:
        """Stop both threads, release the executor and camera. Idempotent."""
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        if self._ai_thread is not None:
            self._ai_thread.join(timeout=2.0)
            self._ai_thread = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("LiveCameraSession stopped camera_id=%s", self.camera_id)

    # ── capture loop (thread 1) ─────────────────────────────────────────

    def _capture_loop(self) -> None:
        cap = self._cap
        while self._running and cap is not None:
            ok, frame = cap.read()
            if not ok:
                logger.warning(
                    "LiveCameraSession camera_id=%s: capture read failed, stopping", self.camera_id
                )
                break
            with self._frame_lock:
                self.latest_frame = frame.copy()
                self._frame_captured_at = time.monotonic()
            self.stream_fps.tick()
        self._running = False

    # ── AI loop (thread 2) ──────────────────────────────────────────────

    def _ai_loop(self) -> None:
        while self._running:
            if self._ai_busy.is_set():
                # G1 — previous inference still running; skip this cycle entirely.
                time.sleep(0.01)
                continue
            with self._frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()  # G2
            if frame is None:
                time.sleep(0.01)
                continue
            self._ai_busy.set()
            self._executor.submit(self._run_inference, frame)
            # Loop returns immediately to the top — never blocks waiting on the
            # executor, so capture/stream loops are never stalled by AI latency.

    def _run_inference(self, frame: "np.ndarray") -> None:
        try:
            frame_index = self._frame_index
            self._frame_index += 1
            with metrics.timer("live_session_inference_duration"):
                matches = self._pipeline.process_frame(frame, frame_index, self._gallery)
            self.update_overlay(matches)
            self.ai_fps.tick()
            self._persist_alerts(matches, frame, frame_index)
        except Exception:
            logger.exception(
                "LiveCameraSession camera_id=%s: inference cycle failed", self.camera_id
            )
        finally:
            self._ai_busy.clear()

    # ── overlay state (written by AI thread, read by stream consumers) ──

    def update_overlay(self, matches: list["FrameMatch"]) -> None:
        new_state: dict[int, dict[str, Any]] = {}
        for m in matches:
            if m.face is None:
                continue
            key = m.track_id if m.track_id is not None else id(m.face)
            if m.should_alert and m.person_id is not None:
                status = "matched"
            elif m.track_id is not None:
                status = "tracking"
            else:
                status = "rejected"
            new_state[key] = {
                "bbox": (
                    int(m.face.bbox.x1), int(m.face.bbox.y1),
                    int(m.face.bbox.x2), int(m.face.bbox.y2),
                ),
                "name": str(m.person_id) if m.person_id is not None else None,
                "confidence": m.confidence,
                "status": status,
            }
        with self._overlay_lock:
            self.overlay_state = new_state

    # ── annotated frame for the MJPEG endpoint ───────────────────────────

    def get_annotated_frame(self) -> "np.ndarray | None":
        with self._frame_lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
        with self._overlay_lock:
            overlay = dict(self.overlay_state)
        for item in overlay.values():
            x1, y1, x2, y2 = item["bbox"]
            color = _OVERLAY_COLORS.get(item["status"], (0, 0, 220))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            parts = []
            if item["name"]:
                parts.append(item["name"])
            if item["confidence"] is not None:
                parts.append(f"{item['confidence'] * 100:.0f}%")
            label = " | ".join(parts) or item["status"]
            cv2.putText(
                frame, label, (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return frame

    # ── debug metrics ──────────────────────────────────────────────────

    @property
    def frame_age_ms(self) -> float:
        with self._frame_lock:
            captured_at = self._frame_captured_at
        if captured_at is None:
            return -1.0
        return (time.monotonic() - captured_at) * 1000.0

    @property
    def ai_busy(self) -> bool:
        return self._ai_busy.is_set()

    # ── gallery load (once at session start) ────────────────────────────

    def _load_gallery_sync(self) -> list[tuple[int, "np.ndarray"]]:
        async def _load() -> list[tuple[int, "np.ndarray"]]:
            factory = get_session_factory()
            async with factory() as session:
                return await video_service.load_gallery(session)

        return asyncio.run(_load())

    # ── alert persistence (mirrors process_prerecorded_video's per-match logic) ──

    def _persist_alerts(self, matches: list["FrameMatch"], frame: "np.ndarray", frame_index: int) -> None:
        alertable = [m for m in matches if m.should_alert and m.person_id is not None]
        if not alertable:
            return
        try:
            asyncio.run(self._persist_alerts_async(alertable, frame, frame_index))
        except Exception:
            logger.exception(
                "LiveCameraSession camera_id=%s: alert persistence failed", self.camera_id
            )

    async def _persist_alerts_async(
        self, alertable: list["FrameMatch"], frame: "np.ndarray", frame_index: int
    ) -> None:
        from sqlalchemy import select as _select

        from ecoface_lite.db.models import DetectionEvent, Incident, incident_persons

        settings = get_settings()
        alert_engine = get_alert_session_engine()
        factory = get_session_factory()

        async with factory() as session:
            for m in alertable:
                if m.face is None:
                    continue
                name = f"{uuid.uuid4().hex}.jpg"
                snap_path = settings.resolved_snapshots_dir() / name
                ih, iw = frame.shape[:2]
                x1 = max(0, int(m.face.bbox.x1))
                y1 = max(0, int(m.face.bbox.y1))
                x2 = min(iw, int(m.face.bbox.x2))
                y2 = min(ih, int(m.face.bbox.y2))
                px = max(4, int((x2 - x1) * 0.15))
                py = max(4, int((y2 - y1) * 0.15))
                crop = frame[max(0, y1 - py):min(ih, y2 + py), max(0, x1 - px):min(iw, x2 + px)]
                cv2.imwrite(str(snap_path), crop)
                rel_snap = str(Path("data/snapshots") / name)

                det = DetectionEvent(
                    person_id=m.person_id,
                    confidence=m.confidence,
                    threshold_used=m.threshold,
                    source_type="live_camera",
                    source_label=f"camera:{self.camera_id}",
                    frame_index=frame_index,
                    snapshot_path=rel_snap,
                )
                session.add(det)
                await session.flush()

                inc_rows = await session.execute(
                    _select(incident_persons.c.incident_id)
                    .join(Incident, Incident.id == incident_persons.c.incident_id)
                    .where(incident_persons.c.person_id == m.person_id)
                    .where(Incident.status == "open")
                    .where(Incident.is_paused == False)  # noqa: E712
                )
                for (inc_id,) in inc_rows.all():
                    await alert_engine.record_match(
                        session,
                        incident_id=inc_id,
                        person_id=m.person_id,
                        camera_id=self.camera_id,
                        confidence=m.confidence,
                        detection_id=det.id,
                        frame_index=frame_index,
                        snapshot_path=rel_snap,
                        source="live",
                    )
                metrics.increment("live_detection_events_created")
            await session.commit()


class LiveSessionManager:
    """Ref-counts viewers per camera_id so concurrent tabs share one capture/AI pair.

    Session.stop() runs exactly when the last viewer disconnects — never before,
    never orphaned (G3).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[int, LiveCameraSession] = {}
        self._refcounts: dict[int, int] = {}

    def acquire(
        self, camera_id: int, source_url: str, pipeline: "RecognitionPipeline"
    ) -> LiveCameraSession:
        with self._lock:
            session = self._sessions.get(camera_id)
            if session is None:
                session = LiveCameraSession(camera_id, pipeline)
                session.start(source_url)  # raises on failure — not registered if it does
                self._sessions[camera_id] = session
                self._refcounts[camera_id] = 0
            self._refcounts[camera_id] += 1
            return session

    def release(self, camera_id: int) -> None:
        with self._lock:
            if camera_id not in self._refcounts:
                return
            self._refcounts[camera_id] -= 1
            if self._refcounts[camera_id] <= 0:
                session = self._sessions.pop(camera_id, None)
                self._refcounts.pop(camera_id, None)
                if session is not None:
                    session.stop()

    def all_sessions(self) -> dict[int, LiveCameraSession]:
        with self._lock:
            return dict(self._sessions)


_manager: LiveSessionManager | None = None
_manager_lock = threading.Lock()


def get_live_session_manager() -> LiveSessionManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = LiveSessionManager()
    return _manager

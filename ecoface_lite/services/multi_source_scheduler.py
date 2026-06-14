"""MultiSourceScheduler — round-robin frame acquisition across BaseVideoSource instances (VSL Phase 3).

Hard constraints (CLAUDE.md VSL Phase 3 hard stops):
  1. connect() called ONCE before the loop, disconnect() ONCE after — never per-frame.
     ByteTrack/SORT state resets if the capture re-opens mid-stream.
  2. One source returning None or raising must not interrupt other sources.
  3. This scheduler is only active when USE_VSL_FRAME_PATH=True (default False).
     Single-source pipeline continues using the frames() iterator until the flag is
     enabled after 3 consecutive green GPU regression runs.

Usage:
    scheduler = MultiSourceScheduler(sources, fps_window=30)
    connect_results = scheduler.start()          # connect() once per source
    try:
        while True:
            frame = scheduler.get_next_frame()   # round-robin across sources
            if frame is None:
                break                            # all sources exhausted
            pipeline.process_frame(frame.bgr, frame.index, gallery)
    finally:
        scheduler.stop()                         # disconnect() once per source
"""

from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import NamedTuple

from ecoface_lite.core.logging import get_logger
from ecoface_lite.input_sources.base import BaseVideoSource, Frame, HealthStatus

logger = get_logger(__name__)


class SourceStats(NamedTuple):
    source_id: str
    name: str
    fps: float
    frames_delivered: int
    connected: bool
    health: HealthStatus


class _FpsTracker:
    """Rolling-window FPS measurement per source."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)
        self.frames_delivered: int = 0

    def record(self) -> None:
        self._times.append(perf_counter())
        self.frames_delivered += 1

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


class MultiSourceScheduler:
    """Round-robin frame scheduler across multiple BaseVideoSource instances.

    Each source gets exactly one get_frame() call per round. If a source returns
    None or raises, it is skipped and the next source is tried. The scheduler
    returns None only when ALL sources are exhausted or failed.
    """

    def __init__(self, sources: list[BaseVideoSource], fps_window: int = 30) -> None:
        if not sources:
            raise ValueError("MultiSourceScheduler requires at least one source")
        self._sources = sources
        self._trackers = [_FpsTracker(fps_window) for _ in sources]
        self._cursor = 0
        self._started = False

    # ------------------------------------------------------------------ #
    #  Lifecycle — connect/disconnect called ONCE, never per-frame        #
    # ------------------------------------------------------------------ #

    def start(self) -> dict[int, bool]:
        """Connect all sources. Returns {index: connected}. Partial failure is non-fatal."""
        results: dict[int, bool] = {}
        for i, src in enumerate(self._sources):
            try:
                ok = src.connect()
                results[i] = ok
                if ok:
                    meta = src.get_metadata()
                    logger.info(
                        "Scheduler: source %d connected — %s (%s)",
                        i, meta.name, meta.source_type,
                    )
                else:
                    logger.warning("Scheduler: source %d failed to connect", i)
            except Exception as exc:
                results[i] = False
                logger.error("Scheduler: source %d connect error: %s", i, exc)
        self._started = True
        connected = sum(1 for v in results.values() if v)
        logger.info("Scheduler started: %d/%d sources connected", connected, len(self._sources))
        return results

    def stop(self) -> None:
        """Disconnect all sources — called once after the frame loop exits."""
        for i, src in enumerate(self._sources):
            try:
                src.disconnect()
            except Exception as exc:
                logger.error("Scheduler: source %d disconnect error: %s", i, exc)
        self._started = False
        logger.info("Scheduler stopped: %d sources disconnected", len(self._sources))

    # ------------------------------------------------------------------ #
    #  Frame acquisition — round-robin, fully isolated per source         #
    # ------------------------------------------------------------------ #

    def get_next_frame(self) -> Frame | None:
        """Return the next frame from the next available source in round-robin order.

        Tries each source exactly once per call. Returns None only when all
        sources are exhausted in this pass (end of file or all offline).
        """
        if not self._started:
            raise RuntimeError("Call start() before get_next_frame()")

        n = len(self._sources)
        for _ in range(n):
            idx = self._cursor
            self._cursor = (self._cursor + 1) % n
            src = self._sources[idx]
            try:
                frame = src.get_frame()
            except Exception as exc:
                logger.warning(
                    "Scheduler: source %d get_frame() exception (skipping): %s", idx, exc,
                )
                continue
            if frame is not None:
                self._trackers[idx].record()
                return frame
        return None  # all sources returned None this pass

    # ------------------------------------------------------------------ #
    #  Observability                                                       #
    # ------------------------------------------------------------------ #

    def stats(self) -> list[SourceStats]:
        """Per-source FPS, frame count, and health snapshot."""
        result = []
        for i, src in enumerate(self._sources):
            tracker = self._trackers[i]
            try:
                health = src.health_check()
                meta = src.get_metadata()
                name = meta.name
            except Exception:
                health = None
                name = f"source-{i}"
            result.append(SourceStats(
                source_id=str(i),
                name=name,
                fps=tracker.fps,
                frames_delivered=tracker.frames_delivered,
                connected=self._started,
                health=health,
            ))
        return result

    def total_fps(self) -> float:
        """Aggregate FPS across all sources."""
        return sum(t.fps for t in self._trackers)

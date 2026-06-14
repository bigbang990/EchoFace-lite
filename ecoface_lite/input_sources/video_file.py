"""VideoFileSource — wraps OpenCV file reading as a BaseVideoSource (VSL Phase 1).

Backward compat: frames() iterator preserved for existing callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ecoface_lite.core.logging import get_logger
from ecoface_lite.input_sources.base import (
    BaseVideoSource,
    CameraMetadata,
    Frame,
    HealthStatus,
    SourceStatus,
    SourceType,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class FramePacket:
    index: int
    bgr: np.ndarray


class VideoSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[FramePacket]:
        """Yield frames until stream ends."""


class VideoFileSource(VideoSource, BaseVideoSource):
    """Prerecorded file source.

    Supports both the legacy frames() iterator (existing pipeline) and the
    BaseVideoSource streaming interface (VSL Phase 1+).
    """

    def __init__(
        self,
        path: Path,
        frame_skip: int = 1,
        source_id: str | None = None,
        name: str | None = None,
        zone: str | None = None,
        location: str | None = None,
    ) -> None:
        self._path = path.resolve()
        self._frame_skip = max(1, frame_skip)
        self._source_id = source_id or str(self._path)
        self._name = name or self._path.name
        self._zone = zone
        self._location = location
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._connected = False

    # ------------------------------------------------------------------ #
    #  BaseVideoSource interface                                           #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        self._cap = cv2.VideoCapture(str(self._path))
        self._connected = self._cap.isOpened()
        if not self._connected:
            logger.warning("VideoFileSource: cannot open %s", self._path)
        return self._connected

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False
        logger.info("VideoFileSource: released %s", self._path)

    def get_frame(self) -> Frame | None:
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, bgr = self._cap.read()
        if not ok:
            return None
        now = datetime.now(timezone.utc)
        frame = Frame(
            index=self._frame_index,
            bgr=bgr,
            captured_at=now,
            source_id=self._source_id,
        )
        self._frame_index += 1
        return frame

    def get_metadata(self) -> CameraMetadata:
        fps = width = height = None
        if self._cap and self._cap.isOpened():
            fps = self._cap.get(cv2.CAP_PROP_FPS) or None
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = w or None
            height = h or None
        return CameraMetadata(
            source_id=self._source_id,
            name=self._name,
            source_type=SourceType.FILE,
            stream_url=str(self._path),
            zone=self._zone,
            location=self._location,
            fps=fps,
            width=width,
            height=height,
        )

    def health_check(self) -> HealthStatus:
        status = SourceStatus.ONLINE if self._connected else SourceStatus.OFFLINE
        return HealthStatus(
            source_id=self._source_id,
            status=status,
            last_seen=datetime.now(timezone.utc) if self._connected else None,
        )

    # ------------------------------------------------------------------ #
    #  Legacy iterator (backward compat — existing pipeline callers)       #
    # ------------------------------------------------------------------ #

    def frames(self) -> Iterator[FramePacket]:
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self._path}")
        idx = 0
        emitted = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % self._frame_skip == 0:
                    yield FramePacket(index=emitted, bgr=frame)
                    emitted += 1
                idx += 1
        finally:
            cap.release()
            logger.info("Released video capture for %s", self._path)

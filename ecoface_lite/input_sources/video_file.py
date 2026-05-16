"""Input source implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FramePacket:
    index: int
    bgr: np.ndarray


class VideoSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[FramePacket]:
        """Yield frames until stream ends."""


class VideoFileSource(VideoSource):
    """Prerecorded file source (webcam/RTSP can mirror this interface later)."""

    def __init__(self, path: Path, frame_skip: int = 1) -> None:
        self._path = path.resolve()
        self._frame_skip = max(1, frame_skip)

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

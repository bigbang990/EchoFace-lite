"""BaseVideoSource — the contract every video source must satisfy (VSL Phase 1).

Architecture contract (never changes):
  The AI Engine, Tracking Engine, Alert Engine, and Incident Platform call
  get_frame() and get_historical_stream() on a BaseVideoSource.
  They never know if it's a file, a phone, an RTSP camera, or an NVR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Generator

import numpy as np


class SourceType(str, Enum):
    FILE = "file"
    RTSP = "rtsp"
    ANDROID = "android"   # VSL Phase 3
    NVR = "nvr"           # VSL Phase 5 — ONVIF device, live + historical via GetReplayUri
    DVR = "dvr"           # VSL Phase 5 — legacy DVR, live RTSP + operator-exported clips for historical


class SourceStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    RECONNECTING = "reconnecting"
    UNKNOWN = "unknown"


@dataclass
class Frame:
    index: int
    bgr: np.ndarray
    captured_at: datetime
    source_id: str


@dataclass
class CameraMetadata:
    source_id: str
    name: str
    source_type: SourceType
    stream_url: str | None
    zone: str | None
    location: str | None
    fps: float | None
    width: int | None
    height: int | None


@dataclass
class HealthStatus:
    source_id: str
    status: SourceStatus
    last_seen: datetime | None
    error: str | None = None


class BaseVideoSource(ABC):
    """Abstract interface every video source must implement."""

    @abstractmethod
    def connect(self) -> bool:
        """Open the source. Returns True on success."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release all resources."""

    @abstractmethod
    def get_frame(self) -> Frame | None:
        """Return the next frame, or None if the stream has ended or is unavailable."""

    @abstractmethod
    def get_metadata(self) -> CameraMetadata:
        """Return static metadata for this source."""

    @abstractmethod
    def health_check(self) -> HealthStatus:
        """Return current health — called by the health monitor (VSL Phase 2)."""

    # ------------------------------------------------------------------ #
    #  Capability properties — override per source class.                 #
    #  The registry gates get_historical_stream() calls using these       #
    #  before they can reach the NotImplementedError stub.                #
    # ------------------------------------------------------------------ #

    @property
    def supports_live(self) -> bool:
        return True

    @property
    def supports_historical(self) -> bool:
        """False until VSL Phase 4 implements get_historical_stream()."""
        return False

    @property
    def supports_ptz(self) -> bool:
        return False

    def get_historical_stream(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> Generator[Frame, None, None]:
        """Yield frames from historical footage. Implemented in VSL Phase 4."""
        raise NotImplementedError("Historical stream not available until VSL Phase 4")
        # unreachable; satisfies Generator return type annotation
        yield  # type: ignore[misc]

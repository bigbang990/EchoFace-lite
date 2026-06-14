"""RTSPSource — live RTSP/IP camera stream (VSL Phase 1).

Handles Hikvision, Dahua, and Android IP Webcam RTSP endpoints.
Frame buffer: always serves latest frame — stale frames are dropped, never queued.
Auto-reconnect on stream drop: exponential backoff capped at 30s.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import cv2

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

_RECONNECT_BACKOFF_INITIAL = 2.0
_RECONNECT_BACKOFF_CAP = 30.0


class RTSPSource(BaseVideoSource):
    """Live RTSP stream source.

    Android IP Webcam: use URL pattern rtsp://<phone-ip>:8080/h264_ulaw.sdp
    Hikvision: rtsp://<user>:<pass>@<ip>:554/Streaming/Channels/101
    """

    def __init__(
        self,
        source_id: str,
        name: str,
        stream_url: str,
        zone: str | None = None,
        location: str | None = None,
        source_type: SourceType = SourceType.RTSP,
    ) -> None:
        self._source_id = source_id
        self._name = name
        self._stream_url = stream_url
        self._zone = zone
        self._location = location
        self._source_type = source_type
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._status = SourceStatus.UNKNOWN
        self._last_seen: datetime | None = None
        self._error: str | None = None

    # ------------------------------------------------------------------ #
    #  BaseVideoSource interface                                           #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        try:
            cap = cv2.VideoCapture(self._stream_url)
            if cap.isOpened():
                self._cap = cap
                self._status = SourceStatus.ONLINE
                self._last_seen = datetime.now(timezone.utc)
                self._error = None
                logger.info("RTSPSource connected: %s (%s)", self._name, self._stream_url)
                return True
            cap.release()
            self._status = SourceStatus.OFFLINE
            self._error = f"Cannot open stream: {self._stream_url}"
            logger.warning("RTSPSource failed to open: %s", self._stream_url)
            return False
        except Exception as exc:
            self._status = SourceStatus.OFFLINE
            self._error = str(exc)
            logger.exception("RTSPSource connect error: %s", self._name)
            return False

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._status = SourceStatus.OFFLINE
        logger.info("RTSPSource disconnected: %s", self._name)

    def get_frame(self) -> Frame | None:
        if self._cap is None or not self._cap.isOpened():
            self._status = SourceStatus.OFFLINE
            return None
        ok, bgr = self._cap.read()
        if not ok:
            self._status = SourceStatus.OFFLINE
            logger.warning("RTSPSource lost frame: %s — stream may have dropped", self._name)
            return None
        now = datetime.now(timezone.utc)
        self._last_seen = now
        self._status = SourceStatus.ONLINE
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
            source_type=self._source_type,
            stream_url=self._stream_url,
            zone=self._zone,
            location=self._location,
            fps=fps,
            width=width,
            height=height,
        )

    def health_check(self) -> HealthStatus:
        if self._cap is not None and not self._cap.isOpened():
            self._status = SourceStatus.OFFLINE
        return HealthStatus(
            source_id=self._source_id,
            status=self._status,
            last_seen=self._last_seen,
            error=self._error,
        )

    # ------------------------------------------------------------------ #
    #  Capability properties (live streams don't support historical)      #
    # ------------------------------------------------------------------ #

    @property
    def supports_historical(self) -> bool:
        return False  # RTSP/Android live streams have no on-source history (NVR does — VSL Phase 5)

    # ------------------------------------------------------------------ #
    #  Reconnect helper (called by health monitor in VSL Phase 2)         #
    # ------------------------------------------------------------------ #

    def reconnect_with_backoff(self, max_attempts: int = 5) -> bool:
        """Try to reconnect with exponential backoff. Returns True on success."""
        self._status = SourceStatus.RECONNECTING
        backoff = _RECONNECT_BACKOFF_INITIAL
        for attempt in range(1, max_attempts + 1):
            logger.info(
                "RTSPSource reconnect attempt %d/%d: %s (backoff %.1fs)",
                attempt, max_attempts, self._name, backoff,
            )
            self.disconnect()
            if self.connect():
                return True
            time.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_BACKOFF_CAP)
        self._status = SourceStatus.OFFLINE
        return False

"""AndroidCameraSource — live MJPEG stream from Android IP Webcam app (VSL Phase 3).

Tested stream URL: http://192.168.1.21:8080/video  (MJPEG endpoint, IP Webcam app)
The /video endpoint delivers MJPEG which OpenCV reads without H.264 decode overhead.
Avoid the RTSP endpoint from IP Webcam on Colab — NAT punch-through is unreliable.

Stale-frame policy: IP Webcam maintains an internal MJPEG frame buffer. Every
get_frame() call flushes it with grab() before retrieve() so the pipeline always
receives the most recent camera frame, never a queued stale one.

Reconnect policy: 3 consecutive read failures trigger an inline reconnect attempt.
Backoff sequence: 5 s → 10 s → 30 s (capped). Counter resets on any successful read.

HealthStatus mapping (SourceStatus has no WARNING variant):
  consecutive_failures == 0   → ONLINE
  consecutive_failures 1–2    → UNKNOWN  (degraded; reconnect not yet triggered)
  consecutive_failures >= 3   → RECONNECTING
  cap is None                 → OFFLINE
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Generator

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

_FAILURE_THRESHOLD = 3
# Backoff in seconds; index capped at last entry → 30 s maximum
_RECONNECT_BACKOFF = [5.0, 10.0, 30.0]


class AndroidCameraSource(BaseVideoSource):
    """Live MJPEG stream source for an Android device running IP Webcam.

    Tested URL format: http://<phone-ip>:8080/video
    Example:           http://192.168.1.21:8080/video
    """

    def __init__(
        self,
        source_id: str,
        name: str,
        stream_url: str,
        zone: str | None = None,
        location: str | None = None,
    ) -> None:
        self._source_id = source_id
        self._name = name
        self._stream_url = stream_url
        self._zone = zone
        self._location = location
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._consecutive_failures = 0
        self._reconnect_attempts = 0
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
                self._consecutive_failures = 0
                self._reconnect_attempts = 0
                self._status = SourceStatus.ONLINE
                self._last_seen = datetime.now(timezone.utc)
                self._error = None
                logger.info(
                    "AndroidCameraSource connected: %s (%s)",
                    self._name, self._stream_url,
                )
                return True
            cap.release()
            self._status = SourceStatus.OFFLINE
            self._error = f"Cannot open stream: {self._stream_url}"
            logger.warning("AndroidCameraSource failed to open: %s", self._stream_url)
            return False
        except Exception as exc:
            self._status = SourceStatus.OFFLINE
            self._error = str(exc)
            logger.exception("AndroidCameraSource connect error: %s", self._name)
            return False

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._status = SourceStatus.OFFLINE
        logger.info("AndroidCameraSource disconnected: %s", self._name)

    def get_frame(self) -> Frame | None:
        if self._cap is None or not self._cap.isOpened():
            self._status = SourceStatus.OFFLINE
            return None

        # Flush IP Webcam's internal MJPEG buffer.
        # grab() discards the buffered frame; retrieve() then decodes the newest one.
        # Without this, frames lag by several seconds under typical phone capture rates.
        if not self._cap.grab():
            return self._handle_failure()

        ok, bgr = self._cap.retrieve()
        if not ok or bgr is None:
            return self._handle_failure()

        self._consecutive_failures = 0
        self._reconnect_attempts = 0
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
            raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
            fps = raw_fps if raw_fps and raw_fps > 0 else None
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = w or None
            height = h or None
        return CameraMetadata(
            source_id=self._source_id,
            name=self._name,
            source_type=SourceType.ANDROID,
            stream_url=self._stream_url,
            zone=self._zone,
            location=self._location,
            fps=fps,
            width=width,
            height=height,
        )

    def health_check(self) -> HealthStatus:
        return HealthStatus(
            source_id=self._source_id,
            status=self._derive_status(),
            last_seen=self._last_seen,
            error=self._error,
        )

    # ------------------------------------------------------------------ #
    #  Capability properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def supports_live(self) -> bool:
        return True

    @property
    def supports_historical(self) -> bool:
        return False

    @property
    def supports_ptz(self) -> bool:
        return False

    def get_historical_stream(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> Generator[Frame, None, None]:
        raise NotImplementedError("AndroidCameraSource does not support historical stream")
        yield  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _derive_status(self) -> SourceStatus:
        if self._cap is None:
            return SourceStatus.OFFLINE
        if self._consecutive_failures == 0:
            return SourceStatus.ONLINE
        if self._consecutive_failures < _FAILURE_THRESHOLD:
            # Degraded but reconnect not yet triggered; UNKNOWN is the closest
            # SourceStatus variant (base has no WARNING).
            return SourceStatus.UNKNOWN
        return SourceStatus.RECONNECTING

    def _handle_failure(self) -> None:
        self._consecutive_failures += 1
        logger.warning(
            "AndroidCameraSource read failure %d/%d: %s",
            self._consecutive_failures, _FAILURE_THRESHOLD, self._name,
        )
        if self._consecutive_failures >= _FAILURE_THRESHOLD:
            self._reconnect()
        return None

    def _reconnect(self) -> None:
        backoff = _RECONNECT_BACKOFF[min(self._reconnect_attempts, len(_RECONNECT_BACKOFF) - 1)]
        self._reconnect_attempts += 1
        self._status = SourceStatus.RECONNECTING
        logger.info(
            "AndroidCameraSource reconnecting: %s (attempt %d, backoff %.0fs)",
            self._name, self._reconnect_attempts, backoff,
        )
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        time.sleep(backoff)
        self.connect()

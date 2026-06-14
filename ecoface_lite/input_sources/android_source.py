"""AndroidCameraSource — Android IP Webcam / DroidCam via RTSP (VSL Phase 3).

Functionally identical to RTSPSource. The ANDROID source_type tag enables
source-specific health reporting and future camera-topology filtering.

Tested URL patterns:
  rtsp://<phone-ip>:8080/h264_ulaw.sdp   (IP Webcam app — most common)
  rtsp://<phone-ip>:4747/video            (DroidCam)
  rtsp://<phone-ip>:8080/video            (IP Webcam, some versions)

Auto-reconnect: Android RTSP servers restart quickly after app resumes —
initial backoff is 1s (half of RTSP default) with the same 30s cap.
"""

from __future__ import annotations

from ecoface_lite.core.logging import get_logger
from ecoface_lite.input_sources.base import SourceType
from ecoface_lite.input_sources.rtsp_source import RTSPSource

logger = get_logger(__name__)

_ANDROID_BACKOFF_INITIAL = 1.0  # shorter than RTSP — Android RTSP restarts fast


class AndroidCameraSource(RTSPSource):
    """Android IP Webcam source (RTSP, VSL Phase 3)."""

    def __init__(
        self,
        source_id: str,
        name: str,
        stream_url: str,
        zone: str | None = None,
        location: str | None = None,
    ) -> None:
        super().__init__(
            source_id=source_id,
            name=name,
            stream_url=stream_url,
            zone=zone,
            location=location,
            source_type=SourceType.ANDROID,
        )

    def reconnect_with_backoff(self, max_attempts: int = 5) -> bool:
        """Android RTSP restarts faster — use shorter initial backoff."""
        import time
        from ecoface_lite.input_sources.base import SourceStatus
        from ecoface_lite.input_sources.rtsp_source import _RECONNECT_BACKOFF_CAP

        self._status = SourceStatus.RECONNECTING
        backoff = _ANDROID_BACKOFF_INITIAL
        for attempt in range(1, max_attempts + 1):
            logger.info(
                "AndroidCameraSource reconnect %d/%d: %s (backoff %.1fs)",
                attempt, max_attempts, self._name, backoff,
            )
            self.disconnect()
            if self.connect():
                return True
            time.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_BACKOFF_CAP)
        self._status = SourceStatus.OFFLINE
        return False

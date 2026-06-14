"""NVRSource and DVRSource — VSL Phase 5: enterprise NVR/DVR integration.

NVRSource
---------
Live stream: inherits RTSPSource (NVRs expose live feeds via RTSP).
Historical:  ONVIF GetReplayUri() returns a time-windowed RTSP playback URL
             which OpenCV opens exactly like a VideoFileSource — same seeking
             logic, same Frame output.

Requires `onvif-zeep` for historical access:
    pip install onvif-zeep

Without it, connect() works for live; get_historical_stream() raises ImportError
with the install instruction. The module loads regardless — no hard import.

DVRSource
---------
Live stream: inherits RTSPSource.
Historical:  operator drops exported clips into `dvr_clip_dir`; this source
             finds the best-matching file by mtime and delegates to
             VideoFileSource.get_historical_stream(). Zero new packages needed.

ONVIF discovery
---------------
NVRSource.discover(timeout_seconds) is a class method that sends a
WS-Discovery probe and returns a list of device candidates. Also requires
`onvif-zeep` (which bundles wsdiscovery). Raises ImportError if absent.
"""

from __future__ import annotations

import base64
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from ecoface_lite.core.logging import get_logger
from ecoface_lite.input_sources.base import (
    BaseVideoSource,
    CameraMetadata,
    Frame,
    HealthStatus,
    SourceStatus,
    SourceType,
)
from ecoface_lite.input_sources.rtsp_source import RTSPSource

logger = get_logger(__name__)

_ONVIF_MISSING_MSG = (
    "onvif-zeep is not installed. "
    "Install it with: pip install onvif-zeep"
)


def _decode_password(onvif_password_enc: str | None) -> str:
    """Decode base64-stored ONVIF password. Returns empty string if None."""
    if not onvif_password_enc:
        return ""
    try:
        return base64.b64decode(onvif_password_enc.encode()).decode()
    except Exception:
        return onvif_password_enc  # fall through if not base64


def _encode_password(plaintext: str) -> str:
    """Base64-encode a password for storage in SQLite."""
    return base64.b64encode(plaintext.encode()).decode()


class NVRSource(RTSPSource):
    """ONVIF-capable NVR source.

    Live path:       RTSPSource.get_frame() — unchanged.
    Historical path: ONVIF GetReplayUri() → time-windowed RTSP URL → OpenCV.
    """

    def __init__(
        self,
        source_id: str,
        name: str,
        stream_url: str,           # live RTSP URL (main stream)
        onvif_host: str,
        onvif_port: int = 80,
        onvif_username: str = "admin",
        onvif_password_enc: str | None = None,
        zone: str | None = None,
        location: str | None = None,
    ) -> None:
        super().__init__(
            source_id=source_id,
            name=name,
            stream_url=stream_url,
            zone=zone,
            location=location,
            source_type=SourceType.NVR,
        )
        self._onvif_host = onvif_host
        self._onvif_port = onvif_port
        self._onvif_username = onvif_username
        self._onvif_password = _decode_password(onvif_password_enc)

    @property
    def supports_historical(self) -> bool:
        return True

    def get_metadata(self) -> CameraMetadata:
        meta = super().get_metadata()
        return CameraMetadata(
            source_id=meta.source_id,
            name=meta.name,
            source_type=SourceType.NVR,
            stream_url=meta.stream_url,
            zone=meta.zone,
            location=meta.location,
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
        )

    def get_historical_stream(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> Generator[Frame, None, None]:
        """Yield frames from NVR recording via ONVIF GetReplayUri.

        Flow:
          1. Connect ONVIF media service (requires onvif-zeep).
          2. Get first media profile token.
          3. Call GetReplayUri with StartDateTime/EndDateTime.
          4. Open the returned RTSP URL with OpenCV.
          5. Yield Frame objects until EOF or end_time reached.
        """
        try:
            from onvif import ONVIFCamera  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_ONVIF_MISSING_MSG) from exc

        import cv2

        logger.info(
            "NVRSource %s: connecting ONVIF at %s:%d for historical stream",
            self._source_id, self._onvif_host, self._onvif_port,
        )
        try:
            cam = ONVIFCamera(
                self._onvif_host,
                self._onvif_port,
                self._onvif_username,
                self._onvif_password,
            )
            media = cam.create_media_service()
            replay = cam.create_replay_service()
            profiles = media.GetProfiles()
            if not profiles:
                raise RuntimeError("NVR returned no media profiles")
            token = profiles[0].token

            # ONVIF datetime strings: ISO 8601 UTC
            def _onvif_dt(dt: datetime) -> str:
                return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            request = replay.create_type("GetReplayUri")
            request.StreamSetup = {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            }
            request.RecordingToken = token
            # Some NVRs support time-range in the replay URI request
            if hasattr(request, "StartDateTime"):
                request.StartDateTime = _onvif_dt(start_time)
                request.EndDateTime = _onvif_dt(end_time)

            response = replay.GetReplayUri(request)
            replay_url = response.Uri
            logger.info("NVRSource %s: replay URL obtained", self._source_id)
        except ImportError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"NVRSource {self._source_id}: ONVIF GetReplayUri failed — {exc}"
            ) from exc

        # Open replay RTSP with OpenCV and stream frames
        cap = cv2.VideoCapture(replay_url)
        if not cap.isOpened():
            raise RuntimeError(
                f"NVRSource {self._source_id}: cannot open replay stream {replay_url}"
            )

        idx = 0
        end_ts = end_time.timestamp()
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                now = datetime.now(timezone.utc)
                # Stop if we've passed the requested window
                if now.timestamp() > end_ts + 5:
                    break
                yield Frame(
                    index=idx,
                    bgr=bgr,
                    captured_at=now,
                    source_id=self._source_id,
                )
                idx += 1
        finally:
            cap.release()

    # ------------------------------------------------------------------ #
    #  ONVIF device discovery (opt-in class method)                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def discover(cls, timeout_seconds: float = 5.0) -> list[dict]:
        """WS-Discovery probe for ONVIF devices on the local network.

        Returns a list of candidate dicts:
          [{"xaddrs": [...], "types": [...], "scopes": [...]}]

        Raises ImportError if onvif-zeep is not installed.
        This method is NEVER called automatically — operator-triggered only.
        """
        try:
            from wsdiscovery import WSDiscovery  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_ONVIF_MISSING_MSG) from exc

        wsd = WSDiscovery()
        wsd.start()
        try:
            services = wsd.searchServices(timeout=timeout_seconds)
            results = []
            for svc in services:
                results.append({
                    "xaddrs": list(svc.getXAddrs()),
                    "types": [str(t) for t in svc.getTypes()],
                    "scopes": [str(s) for s in svc.getScopes()],
                })
            logger.info("NVRSource.discover(): found %d device(s)", len(results))
            return results
        finally:
            wsd.stop()

    def get_device_info(self) -> dict:
        """Fetch ONVIF device information (model, firmware, serial).

        Raises ImportError if onvif-zeep is not installed.
        Used by POST /cameras/{id}/nvr/test-onvif.
        """
        try:
            from onvif import ONVIFCamera  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_ONVIF_MISSING_MSG) from exc

        cam = ONVIFCamera(
            self._onvif_host,
            self._onvif_port,
            self._onvif_username,
            self._onvif_password,
        )
        devicemgmt = cam.create_devicemgmt_service()
        info = devicemgmt.GetDeviceInformation()
        return {
            "manufacturer": getattr(info, "Manufacturer", None),
            "model": getattr(info, "Model", None),
            "firmware_version": getattr(info, "FirmwareVersion", None),
            "serial_number": getattr(info, "SerialNumber", None),
            "hardware_id": getattr(info, "HardwareId", None),
            "onvif_host": self._onvif_host,
            "onvif_port": self._onvif_port,
        }


class DVRSource(RTSPSource):
    """Legacy DVR source.

    Live path:       RTSPSource.get_frame() — unchanged.
    Historical path: operator exports a clip to dvr_clip_dir; this source
                     finds the best-matching file by mtime and delegates to
                     VideoFileSource.get_historical_stream().
    """

    def __init__(
        self,
        source_id: str,
        name: str,
        stream_url: str,
        dvr_clip_dir: str | Path,
        zone: str | None = None,
        location: str | None = None,
    ) -> None:
        super().__init__(
            source_id=source_id,
            name=name,
            stream_url=stream_url,
            zone=zone,
            location=location,
            source_type=SourceType.DVR,
        )
        self._dvr_clip_dir = Path(dvr_clip_dir)

    @property
    def supports_historical(self) -> bool:
        return True

    def get_metadata(self) -> CameraMetadata:
        meta = super().get_metadata()
        return CameraMetadata(
            source_id=meta.source_id,
            name=meta.name,
            source_type=SourceType.DVR,
            stream_url=meta.stream_url,
            zone=meta.zone,
            location=meta.location,
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
        )

    def _find_clip(self, start_time: datetime, end_time: datetime) -> Path | None:
        """Find the clip file in dvr_clip_dir whose mtime best covers the window.

        Strategy: collect all video files, sort by mtime descending,
        return the first whose mtime falls before end_time and is
        within 24 hours of start_time (clips are typically short exports).
        Falls back to the most recent file if no close match found.
        """
        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".m4v"}
        if not self._dvr_clip_dir.is_dir():
            logger.warning(
                "DVRSource %s: dvr_clip_dir %s does not exist",
                self._source_id, self._dvr_clip_dir,
            )
            return None

        candidates = [
            p for p in self._dvr_clip_dir.iterdir()
            if p.is_file() and p.suffix.lower() in video_exts
        ]
        if not candidates:
            logger.warning(
                "DVRSource %s: no video files in %s",
                self._source_id, self._dvr_clip_dir,
            )
            return None

        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()

        # Prefer files whose mtime is between start and end (clip recorded in window)
        window_matches = [
            p for p in candidates
            if start_ts - 86400 <= p.stat().st_mtime <= end_ts + 86400
        ]
        pool = window_matches if window_matches else candidates
        # Pick the file whose mtime is closest to the requested start
        best = min(pool, key=lambda p: abs(p.stat().st_mtime - start_ts))
        logger.info(
            "DVRSource %s: selected clip %s for window [%s, %s]",
            self._source_id, best.name,
            start_time.isoformat(), end_time.isoformat(),
        )
        return best

    def get_historical_stream(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> Generator[Frame, None, None]:
        """Yield frames from the best-matching operator-exported DVR clip."""
        from ecoface_lite.input_sources.video_file import VideoFileSource

        clip = self._find_clip(start_time, end_time)
        if clip is None:
            raise FileNotFoundError(
                f"DVRSource {self._source_id}: no clip found in {self._dvr_clip_dir} "
                f"for window [{start_time.isoformat()}, {end_time.isoformat()}]"
            )

        delegate = VideoFileSource(
            path=clip,
            source_id=self._source_id,
        )
        yield from delegate.get_historical_stream(start_time, end_time)

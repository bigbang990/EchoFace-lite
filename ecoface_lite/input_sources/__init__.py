from ecoface_lite.input_sources.android_source import AndroidCameraSource
from ecoface_lite.input_sources.base import (
    BaseVideoSource,
    CameraMetadata,
    Frame,
    HealthStatus,
    SourceStatus,
    SourceType,
)
from ecoface_lite.input_sources.nvr_source import DVRSource, NVRSource
from ecoface_lite.input_sources.rtsp_source import RTSPSource
from ecoface_lite.input_sources.source_registry import SourceRegistry, get_source_registry
from ecoface_lite.input_sources.video_file import FramePacket, VideoFileSource, VideoSource

__all__ = [
    # VSL Phase 3
    "AndroidCameraSource",
    # VSL Phase 5
    "NVRSource",
    "DVRSource",
    # VSL Phase 1 abstractions
    "BaseVideoSource",
    "Frame",
    "CameraMetadata",
    "HealthStatus",
    "SourceStatus",
    "SourceType",
    "RTSPSource",
    "SourceRegistry",
    "get_source_registry",
    # Legacy (backward compat)
    "FramePacket",
    "VideoSource",
    "VideoFileSource",
]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics

if TYPE_CHECKING:
    import numpy as np
    from ecoface_lite.ai_engine.detector import DetectedFace
    from ecoface_lite.ai_engine.pipeline_types import FrameMatch


@dataclass(frozen=True)
class OverlayItem:
    face: "DetectedFace"
    match: "FrameMatch | None"
    label: str
    state: str


class VideoPreviewWriter:
    def __init__(self, settings: Settings, job_id: str | None) -> None:
        self._settings = settings
        self._job_id = job_id or "sync"
        self._preview_dir = self._settings.resolved_previews_dir() / self._job_id
        self._preview_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self._preview_dir / "latest.jpg"

    def should_write(self, emitted_count: int) -> bool:
        return emitted_count == 1 or emitted_count % max(1, self._settings.video_preview_interval) == 0

    def write(self, frame_bgr: "np.ndarray", items: list[OverlayItem], frame_index: int) -> str:
        import cv2

        with metrics.timer("overlay_render_time"):
            annotated = frame_bgr.copy()
            for item in items:
                self._draw_item(annotated, item)
            metrics.increment("overlay_debug_frame_count")
        with metrics.timer("preview_generation_time"):
            latest = self.latest_path
            cv2.imwrite(str(latest), annotated)
            periodic = self._preview_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(periodic), annotated)
        return str(latest.relative_to(self._settings.project_root))

    def _draw_item(self, frame_bgr: "np.ndarray", item: OverlayItem) -> None:
        import cv2

        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, int(item.face.bbox.x1)))
        y1 = max(0, min(h - 1, int(item.face.bbox.y1)))
        x2 = max(x1 + 1, min(w, int(item.face.bbox.x2)))
        y2 = max(y1 + 1, min(h, int(item.face.bbox.y2)))
        color = {
            "yellow": (0, 220, 220),
            "red": (0, 0, 220),
            "green": (0, 180, 0),
        }.get(item.state, (0, 0, 220))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = item.label[:80]
        cv2.putText(
            frame_bgr,
            label,
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

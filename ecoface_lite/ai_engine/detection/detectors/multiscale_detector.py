"""Multi-scale detector for tiny and distant face recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ecoface_lite.ai_engine.detector import DetectedFace
from ecoface_lite.ai_engine.detection.detectors.base_detector import BaseDetector, DetectionConfig
from ecoface_lite.ai_engine.detection.detectors.tile_detector import TileDetector
from ecoface_lite.core.config import Settings
from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScaleSelectionResult:
    """Result of scale selection logic."""
    selected_scales: list[float]
    reason: str


class MultiScaleDetector(BaseDetector):
    """Multi-scale detector with adaptive scale selection."""

    def __init__(
        self,
        base_detector: BaseDetector,
        settings: Settings,
    ) -> None:
        self._base_detector = base_detector
        self._settings = settings
        self._scales = settings.multiscale_scales if hasattr(settings, 'multiscale_scales') else [1.0, 1.5, 2.0]
        self._adaptive_activation = settings.multiscale_adaptive_activation if hasattr(settings, 'multiscale_adaptive_activation') else True
        self._max_scales_per_frame = settings.multiscale_max_scales_per_frame if hasattr(settings, 'multiscale_max_scales_per_frame') else 2
        self._tiny_face_threshold = settings.multiscale_tiny_face_threshold if hasattr(settings, 'multiscale_tiny_face_threshold') else 30
        self._small_face_threshold = settings.multiscale_small_face_threshold if hasattr(settings, 'multiscale_small_face_threshold') else 60
        self._gpu_batching = settings.multiscale_gpu_batching if hasattr(settings, 'multiscale_gpu_batching') else True

        # Track recent face sizes for adaptive selection
        self._recent_face_sizes: list[float] = []
        self._max_history = 32

        # ── Phase 2A.3: Tile detector for crowd recovery ─────────────────────────
        if settings.enable_tile_detection:
            self._tile_detector = TileDetector(self._base_detector, settings)
            logger.info("Tile detection enabled for crowd recovery")
        else:
            self._tile_detector = None

    def detect(
        self,
        frame_bgr: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> list[DetectedFace]:
        """Detect faces using multi-scale inference.

        Args:
            frame_bgr: Input frame in BGR format
            config: Optional detection configuration

        Returns:
            List of detected faces from all scales
        """
        if config is None:
            config = DetectionConfig(scale=1.0, det_size=self._base_detector.get_input_size())

        # Select scales to run
        scale_result = self._select_scales(frame_bgr)
        selected_scales = scale_result.selected_scales

        logger.debug(
            "Multi-scale detection: selected_scales=%s reason=%s",
            selected_scales,
            scale_result.reason,
        )

        # Run detection at selected scales
        all_faces = []
        for scale in selected_scales:
            scale_config = DetectionConfig(
                scale=scale,
                det_size=config.det_size,
                batch_size=config.batch_size,
                enable_enhancement=config.enable_enhancement,
            )
            faces = self._base_detector.detect(frame_bgr, scale_config)
            all_faces.extend(faces)

        # Update face size history
        if all_faces:
            avg_size = np.mean([max(f.bbox.x2 - f.bbox.x1, f.bbox.y2 - f.bbox.y1) for f in all_faces])
            self._recent_face_sizes.append(avg_size)
            if len(self._recent_face_sizes) > self._max_history:
                self._recent_face_sizes.pop(0)

        # ── Phase 2A.3: Apply tile detection for crowd recovery ─────────────────
        if self._tile_detector:
            frame_index = getattr(config, 'frame_index', 0) if config else 0
            all_faces = self._tile_detector.detect_with_tiles(frame_bgr, all_faces, frame_index)

        return all_faces

    def _select_scales(self, frame_bgr: np.ndarray) -> ScaleSelectionResult:
        """Select which scales to run based on scene analysis.

        Args:
            frame_bgr: Input frame for scene analysis

        Returns:
            ScaleSelectionResult with selected scales and reason
        """
        if not self._adaptive_activation:
            # Run all scales if adaptive activation is disabled
            return ScaleSelectionResult(
                selected_scales=self._scales[:self._max_scales_per_frame],
                reason="adaptive_disabled",
            )

        # Always run baseline 1.0x
        selected_scales = [1.0]

        # Analyze recent face sizes
        if self._recent_face_sizes:
            avg_size = np.mean(self._recent_face_sizes)

            # Activate 1.5x if small faces detected
            if avg_size < self._small_face_threshold and 1.5 in self._scales:
                selected_scales.append(1.5)

            # Activate 2.0x if tiny faces detected
            if avg_size < self._tiny_face_threshold and 2.0 in self._scales:
                selected_scales.append(2.0)

            reason = f"adaptive_avg_size_{avg_size:.1f}"
        else:
            # No history yet, run 1.5x as conservative default
            if 1.5 in self._scales and len(selected_scales) < self._max_scales_per_frame:
                selected_scales.append(1.5)
            reason = "no_history_conservative"

        # Limit to max scales per frame
        selected_scales = selected_scales[:self._max_scales_per_frame]

        return ScaleSelectionResult(selected_scales=selected_scales, reason=reason)

    def get_model_name(self) -> str:
        """Return the model name."""
        return f"multiscale_{self._base_detector.get_model_name()}"

    def get_input_size(self) -> tuple[int, int]:
        """Return the default input size."""
        return self._base_detector.get_input_size()

    def enable_adaptive_activation(self, enabled: bool) -> None:
        """Enable or disable adaptive scale selection.

        Args:
            enabled: Whether to enable adaptive activation
        """
        self._adaptive_activation = enabled

    def set_max_scales_per_frame(self, max_scales: int) -> None:
        """Set maximum number of scales to run per frame.

        Args:
            max_scales: Maximum scales per frame
        """
        self._max_scales_per_frame = max_scales

    def reset_face_size_history(self) -> None:
        """Reset the face size history (useful for scene changes)."""
        self._recent_face_sizes.clear()

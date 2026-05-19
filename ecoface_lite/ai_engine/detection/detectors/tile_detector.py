"""Tile-based crowd recovery for dense scene detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace
from ecoface_lite.ai_engine.detection.detectors.base_detector import BaseDetector, DetectionConfig
from ecoface_lite.core.config import Settings
from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TileConfig:
    """Configuration for tile-based detection."""
    tile_size: int = 640
    overlap: float = 0.20
    crowd_threshold: int = 8
    max_tiles: int = 9
    edge_padding: int = 32
    priority_center: bool = True


@dataclass(frozen=True)
class TileInfo:
    """Information about a detection tile."""
    x: int
    y: int
    width: int
    height: int
    scale_x: float
    scale_y: float


class TileDetector:
    """Tile-based detector for crowd scene recovery."""

    def __init__(
        self,
        base_detector: BaseDetector,
        settings: Settings,
        config: TileConfig | None = None,
    ) -> None:
        self._base_detector = base_detector
        self._settings = settings
        self._config = config or TileConfig(
            tile_size=settings.tile_size if hasattr(settings, 'tile_size') else 640,
            overlap=settings.tile_overlap if hasattr(settings, 'tile_overlap') else 0.20,
            crowd_threshold=settings.tile_crowd_threshold if hasattr(settings, 'tile_crowd_threshold') else 8,
            max_tiles=settings.tile_max_tiles if hasattr(settings, 'tile_max_tiles') else 9,
            edge_padding=settings.tile_edge_padding if hasattr(settings, 'tile_edge_padding') else 32,
            priority_center=settings.tile_priority_center if hasattr(settings, 'tile_priority_center') else True,
        )

    def detect_with_tiles(
        self,
        frame_bgr: np.ndarray,
        global_faces: list[DetectedFace],
        frame_index: int,
    ) -> list[DetectedFace]:
        """Run tile detection if scene indicates crowd.

        Args:
            frame_bgr: Input frame in BGR format
            global_faces: Faces detected from global detection
            frame_index: Current frame index

        Returns:
            Combined list of global and tile-detected faces
        """
        # Check if tile detection should be activated
        if not self._should_activate_tiles(global_faces):
            return global_faces

        logger.debug("Tile detection activated: global_faces=%s", len(global_faces))

        # Generate tiles
        tiles = self._generate_tiles(frame_bgr.shape)

        # Run detection on tiles
        tile_faces = []
        for tile_info in tiles:
            tile_faces.extend(self._detect_tile(frame_bgr, tile_info))

        # Merge tile faces with global faces
        combined_faces = global_faces + tile_faces

        logger.debug(
            "Tile detection complete: global=%s tile=%s total=%s",
            len(global_faces),
            len(tile_faces),
            len(combined_faces),
        )

        return combined_faces

    def _should_activate_tiles(self, global_faces: list[DetectedFace]) -> bool:
        """Determine if tile detection should be activated.

        Args:
            global_faces: Faces detected from global detection

        Returns:
            True if tile detection should be activated
        """
        # Activate if crowd threshold exceeded
        if len(global_faces) >= self._config.crowd_threshold:
            return True

        return False

    def _generate_tiles(self, frame_shape: tuple[int, int]) -> list[TileInfo]:
        """Generate tile coordinates for the frame.

        Args:
            frame_shape: Frame dimensions (height, width)

        Returns:
            List of tile information
        """
        h, w = frame_shape[:2]
        tile_size = self._config.tile_size
        overlap = int(tile_size * self._config.overlap)

        tiles: list[TileInfo] = []

        # Calculate number of tiles needed
        n_cols = max(1, int(np.ceil(w / (tile_size - overlap))))
        n_rows = max(1, int(np.ceil(h / (tile_size - overlap))))

        # Limit to max tiles
        max_tiles = self._config.max_tiles
        if n_cols * n_rows > max_tiles:
            # Use center priority: prioritize center tiles
            center_x = w // 2
            center_y = h // 2

            # Generate all tiles first
            all_tiles = []
            for row in range(n_rows):
                for col in range(n_cols):
                    x = col * (tile_size - overlap)
                    y = row * (tile_size - overlap)

                    # Calculate distance from center
                    tile_center_x = x + tile_size // 2
                    tile_center_y = y + tile_size // 2
                    distance = np.sqrt((tile_center_x - center_x) ** 2 + (tile_center_y - center_y) ** 2)

                    all_tiles.append((distance, x, y))

            # Sort by distance and take top N
            all_tiles.sort(key=lambda t: t[0])
            selected_tiles = all_tiles[:max_tiles]

            for _, x, y in selected_tiles:
                tiles.append(self._create_tile_info(x, y, tile_size, frame_shape))
        else:
            # Generate all tiles
            for row in range(n_rows):
                for col in range(n_cols):
                    x = col * (tile_size - overlap)
                    y = row * (tile_size - overlap)
                    tiles.append(self._create_tile_info(x, y, tile_size, frame_shape))

        return tiles

    def _create_tile_info(
        self,
        x: int,
        y: int,
        tile_size: int,
        frame_shape: tuple[int, int],
    ) -> TileInfo:
        """Create tile information with edge padding.

        Args:
            x: Tile x coordinate
            y: Tile y coordinate
            tile_size: Tile size
            frame_shape: Frame dimensions

        Returns:
            TileInfo with coordinates and scale factors
        """
        h, w = frame_shape[:2]
        padding = self._config.edge_padding

        # Add padding
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        x2_pad = min(w, x + tile_size + padding)
        y2_pad = min(h, y + tile_size + padding)

        width = x2_pad - x_pad
        height = y2_pad - y_pad

        # Scale factors to map back to original frame
        scale_x = w / width
        scale_y = h / height

        return TileInfo(
            x=x_pad,
            y=y_pad,
            width=width,
            height=height,
            scale_x=scale_x,
            scale_y=scale_y,
        )

    def _detect_tile(
        self,
        frame_bgr: np.ndarray,
        tile_info: TileInfo,
    ) -> list[DetectedFace]:
        """Detect faces in a single tile.

        Args:
            frame_bgr: Full frame
            tile_info: Tile information

        Returns:
            List of detected faces with coordinates mapped to original frame
        """
        # Extract tile
        tile = frame_bgr[
            tile_info.y : tile_info.y + tile_info.height,
            tile_info.x : tile_info.x + tile_info.width,
        ]

        # Run detection
        faces = self._base_detector.detect(tile)

        # Map coordinates back to original frame
        mapped_faces = []
        for face in faces:
            mapped_bbox = BoundingBox(
                x1=(face.bbox.x1 * tile_info.scale_x) + tile_info.x,
                y1=(face.bbox.y1 * tile_info.scale_y) + tile_info.y,
                x2=(face.bbox.x2 * tile_info.scale_x) + tile_info.x,
                y2=(face.bbox.y2 * tile_info.scale_y) + tile_info.y,
            )

            mapped_face = DetectedFace(
                bbox=mapped_bbox,
                det_score=face.det_score,
                aligned_face=face.aligned_face,
                embedding=face.embedding,
                landmarks=face.landmarks,
                temporal_score=face.temporal_score,
            )
            mapped_faces.append(mapped_face)

        return mapped_faces

from __future__ import annotations

from dataclasses import dataclass

from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace


@dataclass(frozen=True)
class FaceGeometry:
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    area: int
    center_x: float
    center_y: float
    aspect_ratio: float


def scale_bbox_to_original(bbox: BoundingBox, scale: float) -> BoundingBox:
    if scale == 1.0:
        return bbox
    inv = 1.0 / max(scale, 1e-6)
    return BoundingBox(
        x1=bbox.x1 * inv,
        y1=bbox.y1 * inv,
        x2=bbox.x2 * inv,
        y2=bbox.y2 * inv,
    )


def scale_face_to_original(face: DetectedFace, scale: float) -> DetectedFace:
    return DetectedFace(
        bbox=scale_bbox_to_original(face.bbox, scale),
        det_score=face.det_score,
        aligned_face=face.aligned_face,
        embedding=face.embedding,
    )


def clip_bbox_to_frame(bbox: BoundingBox, frame_shape: tuple[int, ...]) -> BoundingBox:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x1 = max(0.0, min(float(width - 1), float(bbox.x1)))
    y1 = max(0.0, min(float(height - 1), float(bbox.y1)))
    x2 = max(x1 + 1.0, min(float(width), float(bbox.x2)))
    y2 = max(y1 + 1.0, min(float(height), float(bbox.y2)))
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def compute_face_geometry(face: DetectedFace, frame_shape: tuple[int, ...]) -> FaceGeometry:
    clipped = clip_bbox_to_frame(face.bbox, frame_shape)
    x1 = int(round(clipped.x1))
    y1 = int(round(clipped.y1))
    x2 = int(round(clipped.x2))
    y2 = int(round(clipped.y2))
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    area = width * height
    aspect = max(width / max(height, 1), height / max(width, 1))
    return FaceGeometry(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        width=width,
        height=height,
        area=area,
        center_x=(x1 + x2) / 2.0,
        center_y=(y1 + y2) / 2.0,
        aspect_ratio=aspect,
    )


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)

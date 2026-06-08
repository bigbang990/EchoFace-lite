"""Small-face acquisition curve — Phase 2D.

Synthetic measurement: for each target face size (150, 120, 100, 80, 60, 40, 20 px width),
paste a real face crop at that size onto a plain background and run the detector.
20 trials per size, varying position (random placement within center region).

Output: data/eval/validation/small_face_acquisition.json
  - The `reliable_threshold_px` value is the Phase 2D baseline that Phase 4 must beat.

Usage:
    python -m ecoface_lite.scripts.validate_small_faces [--face-image PATH]
    # If no face image given, attempts to use any .jpg in data/uploads/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

TARGET_SIZES_PX = [150, 120, 100, 80, 60, 40, 20]
TRIALS_PER_SIZE = 20
CANVAS_SIZE = (640, 640)  # plain background frame


def _build_detector():
    from ecoface_lite.core.config import get_settings
    from ecoface_lite.ai_engine.detector import InsightFaceDetector
    from insightface.app import FaceAnalysis

    s = get_settings()
    app = FaceAnalysis(name=s.insightface_model_name, providers=["CPUExecutionProvider"])
    det_size = (s.detector_input_width, s.detector_input_height)
    app.prepare(ctx_id=s.insightface_ctx_id, det_size=det_size)
    detector = InsightFaceDetector(
        model_name=s.insightface_model_name, ctx_id=s.insightface_ctx_id,
        face_app=app, det_size=det_size,
    )
    return detector, s


def _extract_face_crop(image_path: Path) -> np.ndarray | None:
    """Extract best face crop from an image using the detector."""
    detector, settings = _build_detector()
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    faces = detector.detect(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: f.det_score)
    x1, y1 = max(0, int(best.bbox.x1)), max(0, int(best.bbox.y1))
    x2, y2 = min(img.shape[1], int(best.bbox.x2)), min(img.shape[0], int(best.bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _find_source_image() -> Path | None:
    """Find any usable face image in the project."""
    search_dirs = [
        _PROJECT_ROOT / "data" / "uploads",
        _PROJECT_ROOT / "data" / "snapshots",
        _PROJECT_ROOT / "validation" / "single_person",
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            matches = list(d.glob(ext))
            if matches:
                return matches[0]
    return None


def run_acquisition_curve(face_crop: np.ndarray, detector, settings) -> dict:
    """Paste the crop at each target size 20 times, record hit/miss per size."""
    from ecoface_lite.ai_engine.detection_optimizer import DetectionOptimizer

    optimizer = DetectionOptimizer(settings)
    is_gpu = settings.insightface_ctx_id >= 0
    canvas_h, canvas_w = CANVAS_SIZE
    rng = random.Random(42)

    size_results: list[dict] = []
    reliable_threshold_px: int | None = None

    print(f"\nAcquisition curve — {TRIALS_PER_SIZE} trials per size:")
    print(f"{'Size (px)':<12} {'Hits':<8} {'Misses':<8} {'Hit rate':<12} {'Avg det ms'}")

    for target_px in TARGET_SIZES_PX:
        hits = 0
        runtimes: list[float] = []

        for _ in range(TRIALS_PER_SIZE):
            # Resize crop to target pixel width, keep aspect ratio
            h_crop, w_crop = face_crop.shape[:2]
            scale = target_px / max(w_crop, 1)
            new_w = target_px
            new_h = max(1, int(h_crop * scale))

            resized = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Random placement within centre 60% of canvas to avoid edge filtering
            margin_x = int(canvas_w * 0.2)
            margin_y = int(canvas_h * 0.2)
            max_x = canvas_w - margin_x - new_w
            max_y = canvas_h - margin_y - new_h
            if max_x < margin_x:
                max_x = margin_x
            if max_y < margin_y:
                max_y = margin_y
            px = rng.randint(margin_x, max_x)
            py = rng.randint(margin_y, max_y)

            # Plain grey background
            canvas = np.full((canvas_h, canvas_w, 3), 128, dtype=np.uint8)
            canvas[py:py + new_h, px:px + new_w] = resized

            # Run detector (no pre-processing, direct canvas)
            det_frame, scale_factor = optimizer.prepare_for_detection(canvas)
            t0 = time.perf_counter()
            faces = detector.detect(det_frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            runtimes.append(elapsed_ms)

            # Check if any detected face overlaps our planted bbox
            planted_x1, planted_y1 = px, py
            planted_x2, planted_y2 = px + new_w, py + new_h

            # Scale detections back
            faces = optimizer.scale_faces(faces, scale_factor)
            detected = False
            for face in faces:
                # IoU between detected bbox and planted bbox
                ix1 = max(face.bbox.x1, planted_x1)
                iy1 = max(face.bbox.y1, planted_y1)
                ix2 = min(face.bbox.x2, planted_x2)
                iy2 = min(face.bbox.y2, planted_y2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter > 0:
                    union = (
                        (planted_x2 - planted_x1) * (planted_y2 - planted_y1)
                        + (face.bbox.x2 - face.bbox.x1) * (face.bbox.y2 - face.bbox.y1)
                        - inter
                    )
                    iou = inter / max(union, 1)
                    if iou > 0.3:
                        detected = True
                        break
            if detected:
                hits += 1

        misses = TRIALS_PER_SIZE - hits
        hit_rate = hits / TRIALS_PER_SIZE
        avg_ms = sum(runtimes) / max(len(runtimes), 1)
        print(f"  {target_px:<10} {hits:<8} {misses:<8} {hit_rate*100:<10.1f}%  {avg_ms:.0f}ms")

        size_results.append({
            "target_px": target_px,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hit_rate, 4),
            "avg_detector_ms": round(avg_ms, 1),
        })

        # Reliable threshold = smallest size with hit_rate >= 0.80
        if hit_rate >= 0.80 and reliable_threshold_px is None:
            pass  # keep scanning to find smallest reliable
        if hit_rate >= 0.80:
            reliable_threshold_px = target_px

    # reliable_threshold_px = smallest size where hit_rate >= 80%
    # Walk sorted list smallest-to-largest to find the bottom
    reliable_threshold_px = None
    for entry in reversed(size_results):  # reversed = smallest last
        if entry["hit_rate"] >= 0.80:
            reliable_threshold_px = entry["target_px"]
            break

    if reliable_threshold_px is None:
        reliable_threshold_px = 999  # can't reliably detect anything

    print(f"\n  → reliable_threshold_px (80% hit rate): {reliable_threshold_px}px")
    print(f"  → Phase 4 must beat this value (lower is better)")

    return {
        "acquisition_curve": size_results,
        "reliable_threshold_px": reliable_threshold_px,
        "trials_per_size": TRIALS_PER_SIZE,
        "canvas_size": list(CANVAS_SIZE),
        "hardware_backend": "GPU" if settings.insightface_ctx_id >= 0 else "CPU",
        "det_size": [settings.detector_input_width, settings.detector_input_height],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Small face acquisition curve (Phase 2D baseline)")
    ap.add_argument("--face-image", default=None,
                    help="Path to a photo containing a real face (auto-detected if omitted)")
    ap.add_argument("--out-dir", default="data/eval/validation",
                    help="Output directory for JSON results")
    args = ap.parse_args()

    face_image_path = Path(args.face_image) if args.face_image else _find_source_image()
    if face_image_path is None or not face_image_path.exists():
        print("ERROR: No face image found. Provide --face-image path/to/photo.jpg", file=sys.stderr)
        print("  (or place a photo in data/uploads/)", file=sys.stderr)
        sys.exit(1)

    print(f"Source image: {face_image_path}")
    print("Extracting face crop...")
    face_crop = _extract_face_crop(face_image_path)
    if face_crop is None:
        print(f"ERROR: No face detected in {face_image_path}", file=sys.stderr)
        sys.exit(1)

    h, w = face_crop.shape[:2]
    print(f"Face crop extracted: {w}x{h}px")

    # Re-build detector (clean instance for the curve run)
    detector, settings = _build_detector()
    result = run_acquisition_curve(face_crop, detector, settings)

    out_root = _PROJECT_ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "small_face_acquisition.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult written to: {out_path}")


if __name__ == "__main__":
    main()

"""Standalone detection validation runner — Phase 2D.

Runs the detector DIRECTLY on every frame of a video (no pipeline, no governance,
no adaptive interval, no load-shedding).  Records raw detections AND
validator-passed detections separately so we can diagnose whether low recall
is a detector problem or a validator problem.

Usage:
    python -m ecoface_lite.scripts.run_validation path/to/video.mp4 [--sample-rate N]
    python -m ecoface_lite.scripts.run_validation validation/crowds/ [--sample-rate 5]

Output: JSON per video written to data/eval/validation/<category>/<video>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on the path when invoked as a module
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _build_detector():
    """Build an InsightFace detector with frozen Phase-2C config (no overrides)."""
    from ecoface_lite.core.config import get_settings
    from ecoface_lite.ai_engine.detector import InsightFaceDetector

    s = get_settings()
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=s.insightface_model_name, providers=["CPUExecutionProvider"])
    det_size = (s.detector_input_width, s.detector_input_height)
    app.prepare(ctx_id=s.insightface_ctx_id, det_size=det_size)
    detector = InsightFaceDetector(
        model_name=s.insightface_model_name,
        ctx_id=s.insightface_ctx_id,
        face_app=app,
        det_size=det_size,
    )
    return detector, s


def _build_validator(settings):
    """Build unified FaceValidator from frozen config."""
    from ecoface_lite.core.validator import FaceValidator

    return FaceValidator(settings)


def validate_video(
    video_path: Path,
    detector,
    validator,
    settings,
    sample_rate: int = 1,
) -> dict:
    """Run detector + validator on every sampled frame of a video.

    Returns a result dict matching the Phase-2D schema:
    {
        video, hardware_backend, confidence_threshold, frames_processed,
        total_raw_detections, total_passed_validator, validator_rejection_rate,
        rejection_breakdown, avg_raw_faces_per_frame, avg_passed_faces_per_frame,
        max_raw_faces_per_frame, detector_avg_runtime_ms, false_positive_suspects
    }
    """
    from ecoface_lite.core.validator import ValidationTier
    from ecoface_lite.ai_engine.detection_optimizer import DetectionOptimizer

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    is_gpu = settings.insightface_ctx_id >= 0
    backend = "GPU" if is_gpu else "CPU"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(
        f"  Video: {video_path.name}  ({frame_w}x{frame_h}  {total_frames} frames  "
        f"{fps:.1f}fps  sample_rate=1/{sample_rate})"
    )
    print(f"  Backend: {backend}  det_size=({settings.detector_input_width},{settings.detector_input_height})")

    # Build optimizer (used only for the pre-detection frame resize — not for interval decisions)
    optimizer = DetectionOptimizer(settings)

    frames_processed = 0
    total_raw = 0
    total_passed = 0
    max_raw_per_frame = 0
    detector_runtimes: list[float] = []

    rejection_counts: dict[str, int] = {
        "REJECT": 0,
        "TRACK_ONLY": 0,
        "WEAK_PASS": 0,
        "other": 0,
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        # Pre-process: same resize path the pipeline uses (preserves aspect, caps pixels)
        detection_frame, scale = optimizer.prepare_for_detection(frame)

        t0 = time.perf_counter()
        raw_faces = detector.detect(detection_frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Scale bboxes back to original frame coords
        raw_faces = optimizer.scale_faces(raw_faces, scale)

        detector_runtimes.append(elapsed_ms)
        raw_count = len(raw_faces)
        total_raw += raw_count
        max_raw_per_frame = max(max_raw_per_frame, raw_count)

        # Run validator on each raw face (same validator the pipeline uses)
        passed = 0
        for face in raw_faces:
            result = validator.validate(
                face, frame, frame.shape, frame_idx,
                min_det_confidence=settings.detection_confidence_threshold,
                strict_cutoff=settings.validator_strict_cutoff,
                emergency_rebuild_active=False,
            )
            if result.tier == ValidationTier.STRICT_PASS:
                passed += 1
            elif result.tier == ValidationTier.WEAK_PASS:
                passed += 1
                rejection_counts["WEAK_PASS"] += 1
            elif result.tier == ValidationTier.TRACK_ONLY:
                rejection_counts["TRACK_ONLY"] += 1
            else:
                rejection_counts["REJECT"] += 1

        total_passed += passed
        frames_processed += 1
        frame_idx += 1

    cap.release()

    raw_rate = total_raw / max(frames_processed, 1)
    passed_rate = total_passed / max(frames_processed, 1)
    rejection_rate = (total_raw - total_passed) / max(total_raw, 1)
    avg_runtime = sum(detector_runtimes) / max(len(detector_runtimes), 1)

    result = {
        "video": str(video_path),
        "hardware_backend": backend,
        "confidence_threshold": settings.detection_confidence_threshold,
        "det_size": [settings.detector_input_width, settings.detector_input_height],
        "frame_resolution": [frame_w, frame_h],
        "frames_processed": frames_processed,
        "total_frames_in_video": total_frames,
        "sample_rate": sample_rate,
        "total_raw_detections": total_raw,
        "total_passed_validator": total_passed,
        "validator_rejection_rate": round(rejection_rate, 4),
        "rejection_breakdown": {
            "rejected_outright": rejection_counts["REJECT"],
            "track_only_tier": rejection_counts["TRACK_ONLY"],
            "weak_pass_tier": rejection_counts["WEAK_PASS"],
        },
        "avg_raw_faces_per_frame": round(raw_rate, 3),
        "avg_passed_faces_per_frame": round(passed_rate, 3),
        "max_raw_faces_per_frame": max_raw_per_frame,
        "detector_avg_runtime_ms": round(avg_runtime, 1),
        "false_positive_suspects": 0,  # populated by validate_false_positives.py
    }

    print(
        f"  → raw={total_raw} ({raw_rate:.2f}/frame)  passed={total_passed} ({passed_rate:.2f}/frame)  "
        f"rejection={rejection_rate*100:.1f}%  avg_det={avg_runtime:.0f}ms"
    )
    return result


def run(video_input: str, sample_rate: int = 1, out_dir: str = "data/eval/validation") -> None:
    detector, settings = _build_detector()
    validator = _build_validator(settings)

    root = _PROJECT_ROOT
    out_root = root / out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    p = Path(video_input)
    if p.is_file():
        paths = [p]
    elif p.is_dir():
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
            paths.extend(p.rglob(ext))
        paths.sort()
    else:
        print(f"ERROR: {video_input} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    if not paths:
        print(f"No video files found under {video_input}", file=sys.stderr)
        sys.exit(1)

    results = []
    for vp in paths:
        print(f"\nProcessing: {vp}")
        try:
            r = validate_video(vp, detector, validator, settings, sample_rate=sample_rate)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            continue

        # Write per-video result
        rel = vp.relative_to(root) if vp.is_relative_to(root) else Path(vp.name)
        out_path = out_root / rel.with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(r, indent=2))
        print(f"  Wrote: {out_path}")
        results.append(r)

    if results:
        summary_path = out_root / "run_summary.json"
        summary_path.write_text(json.dumps(results, indent=2))
        print(f"\nSummary written to {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EchoFace validation runner (raw vs validator-passed)")
    ap.add_argument("input", help="Path to video file or directory of videos")
    ap.add_argument("--sample-rate", type=int, default=1,
                    help="Process 1 frame in every N (default: 1 = every frame)")
    ap.add_argument("--out-dir", default="data/eval/validation",
                    help="Output directory for JSON results")
    args = ap.parse_args()
    run(args.input, sample_rate=args.sample_rate, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

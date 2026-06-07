"""Crowd recall validation — Phase 2D.

Reads videos from validation/crowds/ and their manifest (which includes
manually-counted expected_faces_visible per video).  Runs detection on sampled
frames and outputs raw recall and validator-passed recall SEPARATELY — the split
reveals whether low recall is a detector problem or a validator problem.

Usage:
    python -m ecoface_lite.scripts.validate_crowds [--sample-rate N]

Output: data/eval/validation/crowds/crowd_recall.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Crowd recall validation (Phase 2D)")
    ap.add_argument("--sample-rate", type=int, default=5,
                    help="Process 1 frame in every N (default: 5)")
    ap.add_argument("--out-dir", default="data/eval/validation",
                    help="Output directory for results")
    args = ap.parse_args()

    crowds_dir = _PROJECT_ROOT / "validation" / "crowds"
    manifest_path = crowds_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    expected_map: dict[str, int] = {
        v.get("filename", ""): v.get("expected_faces_visible", 0)
        for v in manifest.get("videos", [])
    }

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_paths = [p for p in crowds_dir.iterdir() if p.suffix.lower() in video_exts]

    if not video_paths:
        print("No crowd videos found in validation/crowds/")
        print("Add videos to validation/crowds/ and run again.")
        print(json.dumps({
            "category": "crowds",
            "videos_found": 0,
            "note": "No videos available — add crowd videos to validation/crowds/ to run this check."
        }, indent=2))
        return

    from ecoface_lite.scripts.run_validation import _build_detector, _build_validator, validate_video

    detector, settings = _build_detector()
    validator = _build_validator(settings)

    out_root = _PROJECT_ROOT / args.out_dir / "crowds"
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    for vp in sorted(video_paths):
        expected = expected_map.get(vp.name, None)
        print(f"\nCrowd video: {vp.name}  (expected_faces_visible={expected})")

        result = validate_video(vp, detector, validator, settings, sample_rate=args.sample_rate)
        result["expected_faces_visible"] = expected

        if expected is not None and expected > 0:
            raw_recall = result["avg_raw_faces_per_frame"] / expected
            passed_recall = result["avg_passed_faces_per_frame"] / expected
            result["raw_recall"] = round(raw_recall, 4)
            result["validator_passed_recall"] = round(passed_recall, 4)
            result["recall_gap"] = round(raw_recall - passed_recall, 4)
            if result["recall_gap"] > 0.2:
                result["diagnosis"] = "VALIDATOR_SUPPRESSION: detector found faces but validator rejected >20% — validator gates are too strict"
            elif raw_recall < 0.5:
                result["diagnosis"] = "DETECTOR_MISS: detector itself is missing faces — check det_size, resolution cap, or model quality"
            else:
                result["diagnosis"] = "OK"
            print(
                f"  → raw_recall={raw_recall*100:.1f}%  "
                f"passed_recall={passed_recall*100:.1f}%  "
                f"gap={result['recall_gap']*100:.1f}%  "
                f"diagnosis={result['diagnosis']}"
            )
        else:
            print("  (no expected_faces_visible in manifest — recall not computed)")

        out_path = out_root / vp.with_suffix(".json").name
        out_path.write_text(json.dumps(result, indent=2))
        all_results.append(result)

    summary_path = out_root / "crowd_recall_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

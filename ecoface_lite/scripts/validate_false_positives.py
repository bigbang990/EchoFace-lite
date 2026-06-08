"""False positive validation — Phase 2D.

Runs detector on the FP categories: posters, paintings, statues, tshirts.
These folders contain ZERO real humans — any detection is a false positive.

Records raw FP count AND validator-passed FP count separately.
Key insight: if raw FPR is high but validator-passed FPR is low, the validator
is doing its job.  If validator-passed FPR is also high, we have a deeper problem.

Usage:
    python -m ecoface_lite.scripts.validate_false_positives [--sample-rate N]

Output: data/eval/validation/false_positive_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

FP_CATEGORIES = ["posters", "paintings", "statues", "tshirts"]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="False positive validation (Phase 2D)")
    ap.add_argument("--sample-rate", type=int, default=3,
                    help="Process 1 frame in every N (default: 3)")
    ap.add_argument("--out-dir", default="data/eval/validation",
                    help="Output directory for results")
    args = ap.parse_args()

    from ecoface_lite.scripts.run_validation import _build_detector, _build_validator, validate_video

    detector, settings = _build_detector()
    validator = _build_validator(settings)

    out_root = _PROJECT_ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png"}
    category_summaries = []
    total_raw_fp = 0
    total_passed_fp = 0
    total_frames = 0

    for cat in FP_CATEGORIES:
        cat_dir = _PROJECT_ROOT / "validation" / cat
        if not cat_dir.exists():
            continue

        cat_videos = [p for p in cat_dir.iterdir() if p.suffix.lower() in video_exts]
        if not cat_videos:
            print(f"[{cat}] No files found — skipping")
            continue

        cat_raw_fp = 0
        cat_passed_fp = 0
        cat_frames = 0
        cat_results = []

        for fp in sorted(cat_videos):
            print(f"\n[{cat}] Processing: {fp.name}  (expected_faces_visible=0)")
            try:
                result = validate_video(fp, detector, validator, settings,
                                        sample_rate=args.sample_rate)
                result["expected_faces_visible"] = 0
                result["category"] = cat
                result["fp_raw"] = result["total_raw_detections"]
                result["fp_passed"] = result["total_passed_validator"]
                result["fp_raw_per_frame"] = result["avg_raw_faces_per_frame"]
                result["fp_passed_per_frame"] = result["avg_passed_faces_per_frame"]

                if result["fp_passed"] == 0:
                    result["fp_verdict"] = "PASS: validator suppressed all raw FPs"
                elif result["fp_raw"] > 0 and result["fp_passed"] / result["fp_raw"] < 0.1:
                    result["fp_verdict"] = "WARN: validator passed <10% of raw FPs (mild leakage)"
                else:
                    result["fp_verdict"] = f"FAIL: {result['fp_passed']} validated FPs in {cat}"

                print(f"  → raw_fp={result['fp_raw']}  passed_fp={result['fp_passed']}  verdict={result['fp_verdict']}")
                cat_raw_fp += result["fp_raw"]
                cat_passed_fp += result["fp_passed"]
                cat_frames += result["frames_processed"]
                cat_results.append(result)
            except Exception as e:
                print(f"  FAILED: {e}", file=sys.stderr)

        total_raw_fp += cat_raw_fp
        total_passed_fp += cat_passed_fp
        total_frames += cat_frames

        cat_summary = {
            "category": cat,
            "videos_tested": len(cat_results),
            "total_frames": cat_frames,
            "total_raw_fp": cat_raw_fp,
            "total_passed_fp": cat_passed_fp,
            "raw_fp_per_frame": round(cat_raw_fp / max(cat_frames, 1), 4),
            "passed_fp_per_frame": round(cat_passed_fp / max(cat_frames, 1), 4),
            "validator_suppression_rate": round(
                1 - (cat_passed_fp / max(cat_raw_fp, 1)), 4
            ),
        }
        category_summaries.append(cat_summary)

        cat_out = out_root / cat
        cat_out.mkdir(parents=True, exist_ok=True)
        (cat_out / "fp_results.json").write_text(json.dumps(cat_results, indent=2))

    overall = {
        "total_raw_fp": total_raw_fp,
        "total_passed_fp": total_passed_fp,
        "total_frames": total_frames,
        "overall_raw_fp_rate": round(total_raw_fp / max(total_frames, 1), 4),
        "overall_passed_fp_rate": round(total_passed_fp / max(total_frames, 1), 4),
        "validator_suppression_rate": round(
            1 - (total_passed_fp / max(total_raw_fp, 1)), 4
        ),
        "categories": category_summaries,
    }

    out_path = out_root / "false_positive_summary.json"
    out_path.write_text(json.dumps(overall, indent=2))
    print(f"\n{'='*60}")
    print(f"FP SUMMARY: raw={total_raw_fp}  passed={total_passed_fp}  "
          f"validator_suppression={overall['validator_suppression_rate']*100:.1f}%")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()

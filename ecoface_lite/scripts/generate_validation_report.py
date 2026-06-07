"""Aggregate validation report — Phase 2D.

Reads all JSON outputs from the validation scripts and merges them into
data/eval/validation_baseline.json.  Commit this file — it is the evidence
Phase 3 improvements must beat.

Usage:
    python -m ecoface_lite.scripts.generate_validation_report
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Generate Phase 2D validation baseline report")
    ap.add_argument("--val-dir", default="data/eval/validation",
                    help="Root of validation output (default: data/eval/validation)")
    ap.add_argument("--out", default="data/eval/validation_baseline.json",
                    help="Output path for baseline report")
    args = ap.parse_args()

    val_root = _PROJECT_ROOT / args.val_dir
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "2D",
        "sections": {},
    }

    # ── Small face acquisition curve ─────────────────────────────────────────
    acq = _load_json(val_root / "small_face_acquisition.json")
    if acq:
        report["sections"]["small_face_acquisition"] = {
            "reliable_threshold_px": acq.get("reliable_threshold_px"),
            "hardware_backend": acq.get("hardware_backend"),
            "curve": acq.get("acquisition_curve", []),
        }
        print(f"[OK] Small face acquisition: reliable_threshold_px={acq.get('reliable_threshold_px')}px")
    else:
        report["sections"]["small_face_acquisition"] = {"status": "not_run"}
        print("[--] Small face acquisition: not run (run validate_small_faces.py first)")

    # ── Crowd recall ─────────────────────────────────────────────────────────
    crowd_dir = val_root / "crowds"
    crowd_summary = _load_json(crowd_dir / "crowd_recall_summary.json")
    if crowd_summary and isinstance(crowd_summary, list) and crowd_summary:
        videos_with_expected = [v for v in crowd_summary if v.get("expected_faces_visible")]
        if videos_with_expected:
            avg_raw_recall = sum(v.get("raw_recall", 0) for v in videos_with_expected) / len(videos_with_expected)
            avg_passed_recall = sum(v.get("validator_passed_recall", 0) for v in videos_with_expected) / len(videos_with_expected)
            avg_gap = sum(v.get("recall_gap", 0) for v in videos_with_expected) / len(videos_with_expected)
        else:
            avg_raw_recall = avg_passed_recall = avg_gap = None

        report["sections"]["crowd_recall"] = {
            "videos_tested": len(crowd_summary),
            "avg_raw_recall": round(avg_raw_recall, 4) if avg_raw_recall is not None else None,
            "avg_validator_passed_recall": round(avg_passed_recall, 4) if avg_passed_recall is not None else None,
            "avg_recall_gap": round(avg_gap, 4) if avg_gap is not None else None,
            "per_video": crowd_summary,
        }
        print(f"[OK] Crowd recall: {len(crowd_summary)} videos, "
              f"avg_raw={avg_raw_recall*100:.1f}% " if avg_raw_recall else "[OK] Crowd recall: no expected_faces counts in manifest")
    else:
        report["sections"]["crowd_recall"] = {"status": "not_run"}
        print("[--] Crowd recall: not run (add videos to validation/crowds/ first)")

    # ── False positive summary ────────────────────────────────────────────────
    fp_summary = _load_json(val_root / "false_positive_summary.json")
    if fp_summary:
        report["sections"]["false_positives"] = {
            "total_raw_fp": fp_summary.get("total_raw_fp"),
            "total_passed_fp": fp_summary.get("total_passed_fp"),
            "overall_passed_fp_rate": fp_summary.get("overall_passed_fp_rate"),
            "validator_suppression_rate": fp_summary.get("validator_suppression_rate"),
            "categories": fp_summary.get("categories", []),
        }
        print(f"[OK] False positives: "
              f"raw={fp_summary.get('total_raw_fp')} "
              f"passed={fp_summary.get('total_passed_fp')} "
              f"suppression={fp_summary.get('validator_suppression_rate', 0)*100:.1f}%")
    else:
        report["sections"]["false_positives"] = {"status": "not_run"}
        print("[--] False positives: not run (add FP videos and run validate_false_positives.py)")

    # ── Run summary (all videos combined) ────────────────────────────────────
    run_summary = _load_json(val_root / "run_summary.json")
    if run_summary and isinstance(run_summary, list):
        total_raw = sum(v.get("total_raw_detections", 0) for v in run_summary)
        total_passed = sum(v.get("total_passed_validator", 0) for v in run_summary)
        total_frames = sum(v.get("frames_processed", 0) for v in run_summary)
        avg_reject_rate = (total_raw - total_passed) / max(total_raw, 1)
        avg_det_ms = sum(v.get("detector_avg_runtime_ms", 0) for v in run_summary) / max(len(run_summary), 1)
        report["sections"]["overall_run"] = {
            "videos_processed": len(run_summary),
            "total_frames": total_frames,
            "total_raw_detections": total_raw,
            "total_passed_validator": total_passed,
            "overall_validator_rejection_rate": round(avg_reject_rate, 4),
            "avg_detector_runtime_ms": round(avg_det_ms, 1),
        }
        print(f"[OK] Overall: {len(run_summary)} videos, "
              f"{total_frames} frames, "
              f"raw={total_raw}, passed={total_passed}, "
              f"rejection={avg_reject_rate*100:.1f}%, "
              f"avg_det={avg_det_ms:.0f}ms")
    else:
        report["sections"]["overall_run"] = {"status": "not_run"}
        print("[--] Overall run summary: not available")

    # ── Write baseline ────────────────────────────────────────────────────────
    out_path = _PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n{'='*60}")
    print(f"Phase 2D baseline written to: {out_path}")
    print("Commit this file — Phase 3 improvements must beat these numbers.")


if __name__ == "__main__":
    main()

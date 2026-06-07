"""Compare current eval metrics against the committed baseline.

Usage:
    python -m ecoface_lite.scripts.check_regression \
        --baseline data/eval/baseline_metrics.json \
        --current  ci_metrics.json \
        [--tolerance 0.02]

Exit codes:
    0  all metrics within tolerance
    1  one or more metrics regressed beyond tolerance

Metrics checked (higher is better):
    recall_at_1, precision, small_face_recall

Metrics checked (lower is better):
    false_positive_rate

Metrics skipped if null in baseline (not yet measured):
    any metric with a null value in the baseline file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HIGHER_IS_BETTER = {"recall_at_1", "precision", "small_face_recall"}
LOWER_IS_BETTER = {"false_positive_rate"}


def check(baseline: dict, current: dict, tolerance: float) -> list[str]:
    failures: list[str] = []

    all_keys = HIGHER_IS_BETTER | LOWER_IS_BETTER
    for key in sorted(all_keys):
        base_val = baseline.get(key)
        curr_val = current.get(key)

        if base_val is None:
            print(f"  SKIP  {key}: not in baseline")
            continue
        if curr_val is None:
            print(f"  SKIP  {key}: not in current metrics")
            continue

        if key in HIGHER_IS_BETTER:
            delta = curr_val - base_val
            regressed = delta < -tolerance
            direction = "↓ regressed" if regressed else "✓ ok"
        else:
            delta = curr_val - base_val
            regressed = delta > tolerance
            direction = "↑ regressed" if regressed else "✓ ok"

        symbol = "FAIL" if regressed else "PASS"
        print(
            f"  [{symbol}] {key}: baseline={base_val:.6f}  current={curr_val:.6f}  "
            f"delta={delta:+.6f}  tolerance=±{tolerance:.3f}  {direction}"
        )

        if regressed:
            failures.append(
                f"{key}: baseline={base_val:.6f}  current={curr_val:.6f}  "
                f"delta={delta:+.6f} (limit ±{tolerance:.3f})"
            )

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoFace metric regression checker")
    parser.add_argument("--baseline", default="data/eval/baseline_metrics.json")
    parser.add_argument("--current", default="ci_metrics.json")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Max absolute delta before a metric is considered regressed")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)

    if not baseline_path.exists():
        print(f"ERROR: baseline file not found: {baseline_path}")
        sys.exit(1)
    if not current_path.exists():
        print(f"ERROR: current metrics file not found: {current_path}")
        sys.exit(1)

    baseline = json.loads(baseline_path.read_text())
    current = json.loads(current_path.read_text())

    print(f"\nRegression check: tolerance = ±{args.tolerance:.3f}")
    print(f"  baseline : {baseline_path}")
    print(f"  current  : {current_path}\n")

    failures = check(baseline, current, args.tolerance)

    print()
    if failures:
        print(f"REGRESSION DETECTED — {len(failures)} metric(s) failed:")
        for f in failures:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print("All metrics within tolerance. No regression detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()

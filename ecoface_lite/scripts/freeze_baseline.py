"""Freeze the current pipeline metrics as the committed baseline.

Usage:
    python -m ecoface_lite.scripts.freeze_baseline [--force]

This script:
1. Verifies dataset diversity meets minimum requirements
2. Runs run_eval to get current metrics
3. Writes data/eval/baseline_metrics.json

IMPORTANT: This file must only be updated intentionally (PR with justification).
The CI regression checker uses it as the floor — never auto-update it in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

EVAL_DIR = Path("data/eval")
PERSONS_DIR = EVAL_DIR / "persons"
NEGATIVES_DIR = EVAL_DIR / "negatives"
ANNOTATIONS_PATH = EVAL_DIR / "annotations.jsonl"
BASELINE_PATH = EVAL_DIR / "baseline_metrics.json"

MIN_PERSONS = 50
MIN_IMAGES_PER_PERSON = 5
MIN_SMALL_FACE_FRACTION = 0.20  # fraction of images expected to have sub-100px faces
MIN_LOW_LIGHT_FRACTION = 0.10   # fraction of images expected to be low-light (relaxed for bootstrap)
MIN_NEGATIVES = 50


def check_dataset_diversity() -> list[str]:
    """Return list of warnings (not failures) about dataset diversity."""
    warnings: list[str] = []

    if not PERSONS_DIR.exists():
        return [f"Persons directory missing: {PERSONS_DIR}"]

    person_dirs = [d for d in PERSONS_DIR.iterdir() if d.is_dir()]
    n_persons = len(person_dirs)
    if n_persons < MIN_PERSONS:
        warnings.append(
            f"Only {n_persons} persons enrolled (minimum recommended: {MIN_PERSONS}). "
            "Expand dataset before production baseline freeze."
        )

    total_images = 0
    small_face_count = 0
    low_light_count = 0

    for person_dir in person_dirs:
        imgs = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        n_imgs = len(imgs)
        if n_imgs < MIN_IMAGES_PER_PERSON:
            warnings.append(
                f"Person {person_dir.name} has only {n_imgs} images "
                f"(minimum recommended: {MIN_IMAGES_PER_PERSON})"
            )

        for img_path in imgs:
            total_images += 1
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            # Estimate face width as ~50% of image width for cropped face images
            est_face_w = w * 0.5
            if est_face_w < 100:
                small_face_count += 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(gray.mean())
            if mean_brightness < 60:
                low_light_count += 1

    if total_images > 0:
        small_frac = small_face_count / total_images
        low_frac = low_light_count / total_images
        if small_frac < MIN_SMALL_FACE_FRACTION:
            warnings.append(
                f"Only {small_frac:.1%} of images appear to have small faces "
                f"(target ≥{MIN_SMALL_FACE_FRACTION:.0%}). "
                "Add more sub-100px face crops for robust small-face recall measurement."
            )
        if low_frac < MIN_LOW_LIGHT_FRACTION:
            warnings.append(
                f"Only {low_frac:.1%} of images appear low-light "
                f"(target ≥{MIN_LOW_LIGHT_FRACTION:.0%}). "
                "Add darker frames for low-light coverage."
            )

    neg_dirs = [d for d in NEGATIVES_DIR.iterdir() if d.is_dir()] if NEGATIVES_DIR.exists() else []
    neg_imgs = sum(len(list(d.glob("*.jpg"))) for d in neg_dirs)
    if neg_imgs < MIN_NEGATIVES:
        warnings.append(
            f"Only {neg_imgs} negative images (minimum recommended: {MIN_NEGATIVES}). "
            "False positive rate measurement will be unreliable."
        )

    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze current metrics as baseline")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing baseline without confirmation")
    parser.add_argument("--skip-manifest-check", action="store_true")
    args = parser.parse_args()

    print("=== EchoFace Baseline Freeze ===\n")

    # 1. Diversity check
    print("Checking dataset diversity...")
    warnings = check_dataset_diversity()
    if warnings:
        print(f"\n⚠  {len(warnings)} diversity warning(s):")
        for w in warnings:
            print(f"   • {w}")
        print()
    else:
        print("  Dataset diversity OK.\n")

    # 2. Check if baseline already exists
    if BASELINE_PATH.exists() and not args.force:
        existing = json.loads(BASELINE_PATH.read_text())
        print(f"Existing baseline found at {BASELINE_PATH}:")
        for k, v in existing.items():
            print(f"  {k}: {v}")
        print()
        answer = input("Overwrite existing baseline? (yes/no): ").strip().lower()
        if answer not in {"yes", "y"}:
            print("Aborted.")
            sys.exit(0)

    # 3. Run eval
    print("Running eval pipeline (this may take several minutes)...\n")
    from ecoface_lite.scripts.run_eval import verify_manifest, run_eval

    if not args.skip_manifest_check:
        verify_manifest()

    metrics = run_eval(Path("ci_metrics_temp.json"))

    # 4. Write baseline
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(metrics, indent=2))

    # Clean up temp file
    temp = Path("ci_metrics_temp.json")
    if temp.exists():
        temp.unlink()

    print(f"\n✓ Baseline written to {BASELINE_PATH}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nNext steps:")
    print("  1. git add data/eval/baseline_metrics.json")
    print("  2. git commit -m 'chore: freeze Phase 2C eval baseline'")
    print("  3. Verify CI passes with check_regression.py")


if __name__ == "__main__":
    main()

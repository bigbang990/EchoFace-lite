"""Evaluate the pipeline against the ground-truth dataset at data/eval/.

Usage:
    python -m ecoface_lite.scripts.run_eval [--output ci_metrics.json]

Dataset layout expected:
    data/eval/persons/{person_id}/{image_id}.jpg   <- enrolled persons
    data/eval/negatives/{person_id}/{image_id}.jpg <- non-enrolled persons (FPR)
    data/eval/annotations.jsonl                    <- optional metadata per image
    data/eval/MANIFEST.sha256                      <- integrity manifest

Metrics produced:
    recall_at_1           top-1 recall across all gallery pairs
    precision             fraction of above-threshold matches that are correct
    false_positive_rate   fraction of negative queries that fire above threshold
    small_face_recall     recall on faces with width < 80px in the original image
    avg_embedding_match_time_ms  mean gallery scan latency
    identity_switch_rate  0.0 (single-image eval; populated by stress suite)
    p95_fps               0.0 (populated by benchmark suite)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ecoface_lite.ai_engine.bootstrap import build_recognition_pipeline
from ecoface_lite.core.config import get_settings
from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)

EVAL_DIR = Path("data/eval")
PERSONS_DIR = EVAL_DIR / "persons"
NEGATIVES_DIR = EVAL_DIR / "negatives"
MANIFEST_PATH = EVAL_DIR / "MANIFEST.sha256"


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------

def verify_manifest() -> None:
    if not MANIFEST_PATH.exists():
        logger.warning("No MANIFEST.sha256 found — skipping integrity check")
        return
    manifest: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        sha, _, rel = line.partition("  ")
        manifest[rel.strip()] = sha.strip()

    mismatches: list[str] = []
    for rel_path, expected_sha in manifest.items():
        full = EVAL_DIR / rel_path
        if not full.exists():
            mismatches.append(f"MISSING: {rel_path}")
            continue
        actual = hashlib.sha256(full.read_bytes()).hexdigest()
        if actual != expected_sha:
            mismatches.append(f"CORRUPT: {rel_path} expected={expected_sha[:8]} got={actual[:8]}")

    if mismatches:
        raise RuntimeError("Dataset integrity check failed:\n" + "\n".join(mismatches))
    logger.info("Dataset integrity check passed (%d files)", len(manifest))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("Could not load image: %s", path)
    return img


def face_width_px(pipeline, img: np.ndarray) -> int:
    """Detect first face and return its pixel width; 0 if no detection."""
    try:
        faces = pipeline._detector.detect(img)
        if not faces:
            return 0
        best = max(faces, key=lambda f: f.det_score)
        return int(best.bbox.x2 - best.bbox.x1)
    except Exception:
        return 0


def embed_image(pipeline, img: np.ndarray) -> np.ndarray | None:
    """Preprocess and embed the best face; returns None if no face detected."""
    try:
        prepared = pipeline._preprocessor.process(img)
        faces = pipeline._detector.detect(prepared.bgr)
        if not faces:
            return None
        best = max(faces, key=lambda f: f.det_score)
        return pipeline._embedder.embed_face(prepared.bgr, best)
    except Exception as e:
        logger.debug("Embedding failed: %s", e)
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_eval(output_path: Path) -> dict:
    settings = get_settings()
    pipeline = build_recognition_pipeline(settings)
    threshold = settings.match_confidence_threshold

    logger.info("Loading enrolled persons from %s", PERSONS_DIR)
    if not PERSONS_DIR.exists():
        raise FileNotFoundError(
            f"Eval dataset not found at {PERSONS_DIR}. "
            "Run freeze_baseline.py after populating data/eval/persons/."
        )

    # Build gallery: {person_id_str: list[np.ndarray]}
    gallery_embeddings: dict[str, list[np.ndarray]] = {}
    for person_dir in sorted(PERSONS_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        embs: list[np.ndarray] = []
        for img_path in sorted(person_dir.glob("*.jpg")):
            img = load_image(img_path)
            if img is None:
                continue
            emb = embed_image(pipeline, img)
            if emb is not None:
                embs.append(emb)
        if embs:
            gallery_embeddings[person_id] = embs

    if not gallery_embeddings:
        raise RuntimeError("No embeddings could be computed from eval dataset.")

    logger.info("Gallery built: %d persons, %d total embeddings",
                len(gallery_embeddings),
                sum(len(v) for v in gallery_embeddings.values()))

    # Build mean gallery embedding per person for matching
    mean_gallery: list[tuple[str, np.ndarray]] = [
        (pid, np.mean(np.stack(embs), axis=0))
        for pid, embs in gallery_embeddings.items()
    ]

    # ---------------------------------------------------------------------------
    # Leave-one-out recall evaluation
    # ---------------------------------------------------------------------------
    true_positives = 0
    total_queries = 0
    small_face_tp = 0
    small_face_total = 0
    match_times_ms: list[float] = []

    for person_dir in sorted(PERSONS_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        img_paths = sorted(person_dir.glob("*.jpg"))

        for img_path in img_paths:
            img = load_image(img_path)
            if img is None:
                continue
            query_emb = embed_image(pipeline, img)
            if query_emb is None:
                continue

            width_px = face_width_px(pipeline, img)
            is_small = 0 < width_px < 80

            # Gallery scan (exclude this image's person's mean if leave-one-out — simplified:
            # we just measure if top-1 matches the correct person_id)
            t0 = time.perf_counter()
            best_id, best_sim = None, -1.0
            for pid, mean_emb in mean_gallery:
                sim = cosine_sim(query_emb, mean_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            match_times_ms.append(elapsed_ms)

            total_queries += 1
            matched = best_id == person_id and best_sim >= threshold
            if matched:
                true_positives += 1

            if is_small:
                small_face_total += 1
                if matched:
                    small_face_tp += 1

    recall_at_1 = true_positives / max(total_queries, 1)
    small_face_recall = small_face_tp / max(small_face_total, 1)
    avg_match_ms = float(np.mean(match_times_ms)) if match_times_ms else 0.0

    logger.info("Recall@1: %.4f (%d/%d)", recall_at_1, true_positives, total_queries)
    logger.info("Small-face recall: %.4f (%d/%d)", small_face_recall, small_face_tp, small_face_total)

    # ---------------------------------------------------------------------------
    # Precision (above-threshold queries)
    # ---------------------------------------------------------------------------
    above_threshold_correct = true_positives
    above_threshold_total = sum(
        1 for _ in range(total_queries)  # approximation: reuse recall loop count
    )
    # Recompute precisely
    above_threshold_total = 0
    above_threshold_correct = 0
    for person_dir in sorted(PERSONS_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        for img_path in sorted(person_dir.glob("*.jpg")):
            img = load_image(img_path)
            if img is None:
                continue
            query_emb = embed_image(pipeline, img)
            if query_emb is None:
                continue
            best_id, best_sim = None, -1.0
            for pid, mean_emb in mean_gallery:
                sim = cosine_sim(query_emb, mean_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid
            if best_sim >= threshold:
                above_threshold_total += 1
                if best_id == person_id:
                    above_threshold_correct += 1

    precision = above_threshold_correct / max(above_threshold_total, 1)
    logger.info("Precision: %.4f (%d/%d)", precision, above_threshold_correct, above_threshold_total)

    # ---------------------------------------------------------------------------
    # False positive rate on negatives gallery
    # ---------------------------------------------------------------------------
    false_positives = 0
    negative_total = 0
    if NEGATIVES_DIR.exists():
        for neg_dir in sorted(NEGATIVES_DIR.iterdir()):
            if not neg_dir.is_dir():
                continue
            for img_path in sorted(neg_dir.glob("*.jpg")):
                img = load_image(img_path)
                if img is None:
                    continue
                query_emb = embed_image(pipeline, img)
                if query_emb is None:
                    continue
                negative_total += 1
                best_sim = max(cosine_sim(query_emb, mean_emb) for _, mean_emb in mean_gallery)
                if best_sim >= threshold:
                    false_positives += 1
        logger.info("FPR: %d/%d = %.4f", false_positives, negative_total, false_positives / max(negative_total, 1))
    else:
        logger.warning("No negatives directory at %s — FPR set to null", NEGATIVES_DIR)

    false_positive_rate = false_positives / max(negative_total, 1) if negative_total > 0 else None

    # ---------------------------------------------------------------------------
    # Assemble output
    # ---------------------------------------------------------------------------
    result = {
        "recall_at_1": round(recall_at_1, 6),
        "precision": round(precision, 6),
        "false_positive_rate": round(false_positive_rate, 6) if false_positive_rate is not None else None,
        "small_face_recall": round(small_face_recall, 6),
        "avg_embedding_match_time_ms": round(avg_match_ms, 4),
        "identity_switch_rate": 0.0,
        "p95_fps": 0.0,
        "total_queries": total_queries,
        "small_face_total": small_face_total,
        "negative_total": negative_total,
        "gallery_size": len(gallery_embeddings),
        "threshold_used": threshold,
        "model_name": settings.insightface_model_name,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    logger.info("Eval results written to %s", output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EchoFace eval against ground-truth dataset")
    parser.add_argument("--output", default="ci_metrics.json", help="Output JSON path")
    parser.add_argument("--skip-manifest", action="store_true", help="Skip dataset integrity check")
    args = parser.parse_args()

    if not args.skip_manifest:
        verify_manifest()

    result = run_eval(Path(args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Toggleable monkey-patch instrumentation for confirmation-queue methods.

Activated at startup via ENABLE_CONFIRMATION_PROFILING=true. Never imported
unless that flag is set — zero runtime cost when disabled.
"""

from __future__ import annotations

import json
import os
import threading
from time import perf_counter
from typing import Any, Type

from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)

_stats: dict[str, list] = {
    "admit": [],
    "decay": [],
    "best_match": [],
}
_stats_lock = threading.Lock()
_total_calls = 0
_profiler_installed = False


def install_confirmation_profiler(
    track_manager_cls: Type[Any],
    dump_path: str,
    dump_interval_calls: int = 200,
) -> None:
    """Wrap _admit_or_queue_pending, _decay_pending, and _best_match on
    track_manager_cls with timing instrumentation. Idempotent — safe to call
    multiple times; only installs once.
    """
    global _profiler_installed

    if _profiler_installed:
        logger.warning("confirmation_profiler: already installed, skipping")
        return
    if getattr(track_manager_cls, "_profiler_installed", False):
        logger.warning("confirmation_profiler: class already patched, skipping")
        return

    try:
        _install(track_manager_cls, dump_path, dump_interval_calls)
    except Exception:
        logger.warning(
            "confirmation_profiler: installation failed — profiling disabled",
            exc_info=True,
        )
        return

    _profiler_installed = True
    track_manager_cls._profiler_installed = True
    logger.info(
        "confirmation_profiler: installed on %s, dump every %d calls → %s",
        track_manager_cls.__name__,
        dump_interval_calls,
        dump_path,
    )


# ── internals ────────────────────────────────────────────────────────────────

def _maybe_dump(dump_path: str) -> None:
    snapshot = {}
    with _stats_lock:
        snapshot = {k: list(v) for k, v in _stats.items()}
    tmp = dump_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(dump_path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)
        os.replace(tmp, dump_path)
    except Exception:
        logger.warning("confirmation_profiler: dump to %s failed", dump_path, exc_info=True)


def _install(
    cls: Type[Any],
    dump_path: str,
    dump_interval_calls: int,
) -> None:
    from ecoface_lite.ai_engine.geometry import bbox_iou

    _CENTROID_SPURIOUS_DIST = 40.0

    orig_admit = cls._admit_or_queue_pending
    orig_decay = cls._decay_pending
    orig_best = cls._best_match

    def _wrapped_admit(self, face, frame_index):
        t0 = perf_counter()
        try:
            result = orig_admit(self, face, frame_index)
        except Exception:
            logger.warning("confirmation_profiler: error in _admit_or_queue_pending", exc_info=True)
            return orig_admit(self, face, frame_index)

        duration_ms = (perf_counter() - t0) * 1000.0
        pending_len = len(getattr(self, "_pending", []))

        spurious = 0
        try:
            for p in getattr(self, "_pending", []):
                iou_val = bbox_iou(face.bbox, p.face.bbox)
                if iou_val >= getattr(self._settings, "temporal_min_track_iou", 0.0):
                    bc = p.centroid
                    fx = (face.bbox.x1 + face.bbox.x2) / 2.0
                    fy = (face.bbox.y1 + face.bbox.y2) / 2.0
                    dist = ((fx - bc[0]) ** 2 + (fy - bc[1]) ** 2) ** 0.5
                    if dist > _CENTROID_SPURIOUS_DIST:
                        spurious += 1
        except Exception:
            logger.warning("confirmation_profiler: spurious_match computation failed", exc_info=True)

        entry = {
            "duration_ms": round(duration_ms, 3),
            "pending_len_at_entry": pending_len,
            "spurious_match_count": spurious,
        }
        _record("admit", entry, dump_path, dump_interval_calls)
        return result

    def _wrapped_decay(self, frame_index):
        t0 = perf_counter()
        pending_len = len(getattr(self, "_pending", []))
        try:
            result = orig_decay(self, frame_index)
        except Exception:
            logger.warning("confirmation_profiler: error in _decay_pending", exc_info=True)
            return orig_decay(self, frame_index)
        duration_ms = (perf_counter() - t0) * 1000.0
        entry = {
            "duration_ms": round(duration_ms, 3),
            "pending_len_at_entry": pending_len,
        }
        _record("decay", entry, dump_path, dump_interval_calls)
        return result

    def _wrapped_best(self, face, centroid, frame_index, detector_interval=1):
        t0 = perf_counter()
        try:
            result = orig_best(self, face, centroid, frame_index, detector_interval)
        except Exception:
            logger.warning("confirmation_profiler: error in _best_match", exc_info=True)
            return orig_best(self, face, centroid, frame_index, detector_interval)
        duration_ms = (perf_counter() - t0) * 1000.0
        entry = {"duration_ms": round(duration_ms, 3)}
        _record("best_match", entry, dump_path, dump_interval_calls)
        return result

    cls._admit_or_queue_pending = _wrapped_admit
    cls._decay_pending = _wrapped_decay
    cls._best_match = _wrapped_best


def _record(
    key: str,
    entry: dict,
    dump_path: str,
    dump_interval_calls: int,
) -> None:
    global _total_calls
    with _stats_lock:
        _stats[key].append(entry)
        _total_calls += 1
        should_dump = (_total_calls % dump_interval_calls) == 0
    if should_dump:
        _maybe_dump(dump_path)

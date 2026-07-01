"""FrameEventLogger — per-scenario structured event log for the identity stress suite.

Writes one JSON object per line (JSONL) to ``<log_dir>/<scenario>_events.jsonl`` so the
per-frame track state (bbox, velocity, occlusion duration, identity) captured during a
real-video stress pass can be replayed or diffed afterwards.

Restored module: the stress-suite checkout was missing ``logs/logger.py`` (and
``reports/reporter.py``), which made ``identity_stress_suite/runner.py`` fail to import
and blocked the mandatory regression gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _json_default(o: Any):
    """Best-effort coercion of numpy scalars/arrays to JSON-native types."""
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


class FrameEventLogger:
    """Append-only JSONL logger, one instance per scenario."""

    def __init__(self, log_dir: Path | str, scenario_name: str) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{scenario_name}_events.jsonl"
        self._events: List[Dict[str, Any]] = []
        # Truncate any prior log for this scenario so each run starts clean.
        self._fh = self._path.open("w", encoding="utf-8")

    def log_event(self, event: Dict[str, Any]) -> None:
        self._events.append(event)
        self._fh.write(json.dumps(event, default=_json_default) + "\n")
        self._fh.flush()

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self._events

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        if getattr(self, "_fh", None) is not None and not self._fh.closed:
            self._fh.close()

    def __del__(self) -> None:  # best-effort flush/close on GC
        try:
            self.close()
        except Exception:
            pass

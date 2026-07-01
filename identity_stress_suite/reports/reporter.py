"""BenchmarkReporter — aggregate identity stress-suite scenario results into a JSON
report and flag regressions against a baseline / previous run.

Gate semantics mirror CLAUDE.md's regression gate: every metric is compared to the
baseline in the worse-is-worse direction; any metric that degrades beyond tolerance is a
regression and the gate fails.

Restored module: the stress-suite checkout was missing ``reports/reporter.py`` (and
``logs/logger.py``), which made ``identity_stress_suite/runner.py`` fail to import and
blocked the mandatory regression gate.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Per metric: +1 => higher is better, -1 => lower is better.
_METRIC_DIRECTION: Dict[str, int] = {
    "id_switch_count": -1,
    "bbox_jitter_avg": -1,
    "track_fragmentation_rate": -1,
    "occlusion_recovery_time": -1,
    "continuity_survival_duration": +1,
    "reidentification_success_rate": +1,
    "confidence_stability_score": +1,
    "stable_matches_avg": +1,
}

# Relative tolerance plus a small absolute floor so tiny/zero baselines do not
# produce false regressions on floating-point noise.
_REL_TOL = 0.05
_ABS_TOL = 1e-6


class BenchmarkReporter:
    def __init__(self, reports_dir: Path | str) -> None:
        self._dir = Path(reports_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── public API used by runner.py ──────────────────────────────────────────
    def generate_report(
        self,
        results: List[Dict[str, Any]],
        baseline: Optional[Dict[str, Any]] = None,
        previous: Optional[Dict[str, Any]] = None,
    ) -> str:
        scenarios = [
            {
                "name": r.get("name"),
                "type": r.get("type", "synthetic"),
                "metrics": r.get("metrics", {}),
            }
            for r in (results or [])
        ]
        aggregate = self._aggregate(scenarios)
        baseline_agg = self._extract_aggregate(baseline)
        previous_agg = self._extract_aggregate(previous)

        baseline_cmp, regressions = self._compare(aggregate, baseline_agg)
        previous_cmp, _ = self._compare(aggregate, previous_agg)

        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario_count": len(scenarios),
            "scenarios": scenarios,
            "aggregate": aggregate,
            "baseline_comparison": baseline_cmp,
            "previous_comparison": previous_cmp,
            "regressions": regressions,
            "gate_passed": len(regressions) == 0,
        }

        fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        (self._dir / fname).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return fname

    def recommend_actions(self, final_report: Dict[str, Any]) -> List[str]:
        recs: List[str] = []
        regressions = final_report.get("regressions", [])
        if not final_report.get("baseline_comparison"):
            recs.append(
                "No baseline comparison available — record this report as the baseline "
                "(reports/Phase_2C2_Baseline.json) before merging."
            )
        if regressions:
            recs.append(
                f"GATE FAILED — {len(regressions)} metric(s) regressed vs baseline. DO NOT MERGE."
            )
            recs.extend(f"  - {r}" for r in regressions)
        else:
            recs.append("GATE PASSED — all metrics within tolerance or improved vs baseline.")
        return recs

    # ── helpers ───────────────────────────────────────────────────────────────
    def _aggregate(self, scenarios: List[Dict[str, Any]]) -> Dict[str, float]:
        keys: set[str] = set()
        for s in scenarios:
            keys.update(s.get("metrics", {}).keys())
        agg: Dict[str, float] = {}
        for k in keys:
            vals = [
                float(s["metrics"][k])
                for s in scenarios
                if k in s.get("metrics", {}) and s["metrics"][k] is not None
            ]
            if vals:
                agg[k] = sum(vals) / len(vals)
        return agg

    def _extract_aggregate(self, report: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Pull an aggregate metric dict out of a prior report of unknown shape."""
        if not isinstance(report, dict):
            return None
        if isinstance(report.get("aggregate"), dict):
            return {k: float(v) for k, v in report["aggregate"].items()}
        if isinstance(report.get("scenarios"), list):
            return self._aggregate(
                [{"metrics": s.get("metrics", {})} for s in report["scenarios"]]
            )
        # Flat metric dict fallback (hand-authored baseline).
        flat = {k: float(v) for k, v in report.items() if isinstance(v, (int, float))}
        return flat or None

    def _compare(
        self,
        current: Dict[str, float],
        reference: Optional[Dict[str, float]],
    ) -> tuple[Dict[str, Any], List[str]]:
        if not reference:
            return {}, []
        cmp: Dict[str, Any] = {}
        regressions: List[str] = []
        for k, cur in current.items():
            if k not in reference:
                continue
            ref = float(reference[k])
            direction = _METRIC_DIRECTION.get(k, 0)
            delta = cur - ref
            tol = max(abs(ref) * _REL_TOL, _ABS_TOL)
            regressed = (direction < 0 and delta > tol) or (direction > 0 and delta < -tol)
            cmp[k] = {
                "current": cur,
                "baseline": ref,
                "delta": delta,
                "regressed": bool(regressed),
            }
            if regressed:
                regressions.append(f"{k}: {ref:.4f} -> {cur:.4f} (delta {delta:+.4f}, worse)")
        return cmp, regressions

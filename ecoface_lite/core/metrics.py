from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator


def safe_divide(a: float, b: float) -> float:
    """Divide a by b, returning 0.0 when b <= 0."""
    return a / max(b, 1.0)


@dataclass(frozen=True)
class MetricsSnapshot:
    counters: dict[str, int]
    averages: dict[str, float]
    recent_values: dict[str, list[float]]
    rates: dict[str, float]
    uptime_seconds: float


@dataclass
class _MetricState:
    counters: dict[str, int] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)
    samples: dict[str, int] = field(default_factory=dict)
    recent_values: dict[str, deque[float]] = field(default_factory=dict)
    rate_numerators: dict[str, float] = field(default_factory=dict)
    rate_denominators: dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)


class MetricsRegistry:
    def __init__(self, recent_window: int = 200) -> None:
        self._recent_window = recent_window
        self._state = _MetricState()
        self._lock = Lock()

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._state.counters[name] = self._state.counters.get(name, 0) + value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._state.totals[name] = self._state.totals.get(name, 0.0) + float(value)
            self._state.samples[name] = self._state.samples.get(name, 0) + 1
            bucket = self._state.recent_values.setdefault(name, deque(maxlen=self._recent_window))
            bucket.append(float(value))

    def observe_rate(self, name: str, numerator: float, denominator: float) -> None:
        """Track a rate by storing numerator/denominator separately.
        The ratio is computed at snapshot time to avoid cumulative corruption."""
        with self._lock:
            self._state.rate_numerators[name] = self._state.rate_numerators.get(name, 0.0) + float(numerator)
            self._state.rate_denominators[name] = self._state.rate_denominators.get(name, 0.0) + float(denominator)

    def observe_rolling(self, name: str, value: float) -> None:
        """Record a value in the rolling window only — no cumulative accumulation."""
        with self._lock:
            bucket = self._state.recent_values.setdefault(name, deque(maxlen=self._recent_window))
            bucket.append(float(value))

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - started)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            counters = dict(self._state.counters)
            averages = {
                name: safe_divide(total, float(self._state.samples.get(name, 0)))
                for name, total in self._state.totals.items()
            }
            recent_values = {name: list(values) for name, values in self._state.recent_values.items()}
            rates = {
                name: safe_divide(self._state.rate_numerators[name], self._state.rate_denominators.get(name, 0.0))
                for name in self._state.rate_numerators
            }
            uptime_seconds = time.perf_counter() - self._state.started_at
        return MetricsSnapshot(
            counters=counters,
            averages=averages,
            recent_values=recent_values,
            rates=rates,
            uptime_seconds=uptime_seconds,
        )

    def reset(self) -> None:
        with self._lock:
            self._state = _MetricState()

    def export(self) -> dict[str, object]:
        snap = self.snapshot()
        return {
            "counters": snap.counters,
            "averages": snap.averages,
            "recent_values": snap.recent_values,
            "rates": snap.rates,
            "uptime_seconds": snap.uptime_seconds,
        }


metrics = MetricsRegistry()

"""
Latency instrumentation and performance metrics.

Provides fine-grained timing for every subsystem: CLI startup, provider
requests, TTFT, tool execution, filesystem I/O, search, context assembly,
routing, verification, and cache performance.

All metrics are thread-safe and emit structured events through EventBus.
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional


@dataclass
class TimingEntry:
    """A single timing measurement."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time) * 1000


class MetricsCollector:
    """Collects and aggregates performance metrics.

    Thread-safe. Supports:
    - Named timers (start/stop)
    - Counters
    - Gauges
    - Summary statistics (min, max, avg, p50, p95, p99)
    """

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._active_timers: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def start(self, name: str) -> None:
        """Start a named timer."""
        with self._lock:
            self._active_timers[name] = time.monotonic()

    def stop(self, name: str) -> float:
        """Stop a named timer and record the duration. Returns duration in ms."""
        start = 0.0
        with self._lock:
            start = self._active_timers.pop(name, 0.0)
        if start == 0.0:
            return 0.0
        duration_ms = (time.monotonic() - start) * 1000
        with self._lock:
            self._timings[name].append(duration_ms)
        return duration_ms

    @contextmanager
    def timer(self, name: str) -> Generator[None, None, None]:
        """Context manager for timing a block."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def record_duration(self, name: str, duration_ms: float) -> None:
        """Record a pre-computed duration."""
        with self._lock:
            self._timings[name].append(duration_ms)

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def inc(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        with self._lock:
            self._counters[name] += value

    def dec(self, name: str, value: int = 1) -> None:
        """Decrement a counter."""
        with self._lock:
            self._counters[name] -= value

    def set_counter(self, name: str, value: int) -> None:
        with self._lock:
            self._counters[name] = value

    # ------------------------------------------------------------------
    # Gauges
    # ------------------------------------------------------------------

    def gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_timing_stats(self, name: str) -> dict[str, float]:
        """Get summary statistics for a named timer."""
        with self._lock:
            values = list(self._timings.get(name, []))
        if not values:
            return {}
        values.sort()
        n = len(values)
        return {
            'count': n,
            'min_ms': values[0],
            'max_ms': values[-1],
            'avg_ms': sum(values) / n,
            'p50_ms': values[n // 2],
            'p95_ms': values[int(n * 0.95)] if n >= 2 else values[-1],
            'p99_ms': values[int(n * 0.99)] if n >= 2 else values[-1],
            'total_ms': sum(values),
        }

    def get_counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        with self._lock:
            return self._gauges.get(name, 0.0)

    def get_all_counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def get_all_gauges(self) -> dict[str, float]:
        with self._lock:
            return dict(self._gauges)

    def get_all_timings(self) -> dict[str, dict[str, float]]:
        """Get stats for all timers."""
        with self._lock:
            names = list(self._timings.keys())
        return {name: self.get_timing_stats(name) for name in names}

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def dashboard(self) -> dict[str, Any]:
        """Generate a performance dashboard."""
        timings = self.get_all_timings()
        counters = self.get_all_counters()
        gauges = self.get_all_gauges()

        # Calculate key metrics
        model_calls = counters.get('model.calls', 0)
        tool_calls = counters.get('tool.calls', 0)
        cache_hits = counters.get('cache.hits', 0)
        cache_misses = counters.get('cache.misses', 0)
        cache_total = cache_hits + cache_misses

        return {
            'counters': counters,
            'gauges': gauges,
            'timings': timings,
            'summary': {
                'model_calls': model_calls,
                'tool_calls': tool_calls,
                'cache_hit_rate': cache_hits / cache_total if cache_total else 0.0,
                'total_tokens': gauges.get('tokens.total', 0),
                'estimated_cost': gauges.get('cost.estimated', 0.0),
            },
        }

    def format_dashboard(self) -> str:
        """Format dashboard as a readable string."""
        d = self.dashboard()
        lines = ['=== Harness Performance Metrics ===', '']

        # Counters
        if d['counters']:
            lines.append('Counters:')
            for k, v in sorted(d['counters'].items()):
                lines.append(f'  {k}: {v}')
            lines.append('')

        # Gauges
        if d['gauges']:
            lines.append('Gauges:')
            for k, v in sorted(d['gauges'].items()):
                if isinstance(v, float):
                    lines.append(f'  {k}: {v:.4f}')
                else:
                    lines.append(f'  {k}: {v}')
            lines.append('')

        # Timings
        if d['timings']:
            lines.append('Timings (ms):')
            for k in sorted(d['timings']):
                stats = d['timings'][k]
                if stats:
                    lines.append(
                        f'  {k}: avg={stats["avg_ms"]:.1f} '
                        f'min={stats["min_ms"]:.1f} '
                        f'max={stats["max_ms"]:.1f} '
                        f'n={stats["count"]}'
                    )
            lines.append('')

        # Summary
        s = d['summary']
        lines.append('Summary:')
        lines.append(f'  Model calls: {s["model_calls"]}')
        lines.append(f'  Tool calls: {s["tool_calls"]}')
        lines.append(f'  Cache hit rate: {s["cache_hit_rate"]:.1%}')
        lines.append(f'  Total tokens: {s["total_tokens"]}')
        lines.append(f'  Estimated cost: ${s["estimated_cost"]:.4f}')

        return '\n'.join(lines)

    def clear(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._timings.clear()
            self._counters.clear()
            self._gauges.clear()
            self._active_timers.clear()

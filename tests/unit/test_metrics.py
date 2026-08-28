"""Tests for latency instrumentation and metrics (Phase F).

Covers: MetricsCollector (timers, counters, gauges, dashboard).
"""

from __future__ import annotations

import time

from harness_core.observability.metrics import MetricsCollector


class TestMetricsCollectorTiming:
    def test_context_manager(self):
        m = MetricsCollector()
        with m.timer('block'):
            time.sleep(0.05)
        stats = m.get_timing_stats('block')
        assert stats['count'] == 1
        assert stats['avg_ms'] > 0

    def test_record_duration(self):
        m = MetricsCollector()
        m.record_duration('api_call', 42.5)
        stats = m.get_timing_stats('api_call')
        assert stats['count'] == 1
        assert stats['avg_ms'] == 42.5

    def test_multiple_timings(self):
        m = MetricsCollector()
        for i in range(5):
            m.record_duration('call', float(i * 10))
        stats = m.get_timing_stats('call')
        assert stats['count'] == 5
        assert stats['min_ms'] == 0.0
        assert stats['max_ms'] == 40.0
        assert stats['avg_ms'] == 20.0

    def test_empty_timing(self):
        m = MetricsCollector()
        stats = m.get_timing_stats('nonexistent')
        assert stats == {}

    def test_start_stop(self):
        m = MetricsCollector()
        m.start('test')
        m.record_duration('test', 10.0)
        m.stop('test')
        stats = m.get_timing_stats('test')
        assert stats['count'] >= 1


class TestMetricsCollectorCounters:
    def test_inc(self):
        m = MetricsCollector()
        m.inc('errors')
        m.inc('errors')
        assert m.get_counter('errors') == 2

    def test_inc_value(self):
        m = MetricsCollector()
        m.inc('tokens', 100)
        assert m.get_counter('tokens') == 100

    def test_dec(self):
        m = MetricsCollector()
        m.inc('in_flight', 5)
        m.dec('in_flight')
        assert m.get_counter('in_flight') == 4

    def test_set_counter(self):
        m = MetricsCollector()
        m.set_counter('total', 42)
        assert m.get_counter('total') == 42

    def test_get_all_counters(self):
        m = MetricsCollector()
        m.inc('a', 1)
        m.inc('b', 2)
        all_c = m.get_all_counters()
        assert all_c['a'] == 1
        assert all_c['b'] == 2


class TestMetricsCollectorGauges:
    def test_gauge(self):
        m = MetricsCollector()
        m.gauge('memory_mb', 128.5)
        assert m.get_gauge('memory_mb') == 128.5

    def test_get_all_gauges(self):
        m = MetricsCollector()
        m.gauge('a', 1.0)
        m.gauge('b', 2.0)
        all_g = m.get_all_gauges()
        assert all_g['a'] == 1.0
        assert all_g['b'] == 2.0


class TestMetricsCollectorDashboard:
    def test_dashboard_structure(self):
        m = MetricsCollector()
        m.inc('tool.calls', 5)
        m.gauge('tokens.total', 1000)
        m.record_duration('model.request', 200.0)

        d = m.dashboard()
        assert 'counters' in d
        assert 'gauges' in d
        assert 'timings' in d
        assert 'summary' in d
        assert d['summary']['tool_calls'] == 5
        assert d['summary']['total_tokens'] == 1000

    def test_format_dashboard(self):
        m = MetricsCollector()
        m.inc('model.calls', 3)
        m.gauge('cost.estimated', 0.05)
        text = m.format_dashboard()
        assert 'Harness Performance Metrics' in text
        assert 'Model calls: 3' in text


class TestMetricsCollectorClear:
    def test_clear(self):
        m = MetricsCollector()
        m.inc('counter')
        m.gauge('gauge', 1.0)
        m.record_duration('timer', 10.0)
        m.clear()
        assert m.get_counter('counter') == 0
        assert m.get_gauge('gauge') == 0.0
        assert m.get_timing_stats('timer') == {}

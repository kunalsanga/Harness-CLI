"""Phase 10.5 — ContextReuseManager unit tests (Part 17).

Prevents unnecessary re-reading: an unchanged file is reusable from the
last snapshot; a modified file invalidates the snapshot so the next read
is honest.
"""

from __future__ import annotations

from harness_core.context.reuse import ContextReuseManager


def test_unchanged_content_is_reusable():
    m = ContextReuseManager()
    m.record_read("script.js", content="const x = 1;")
    assert m.has_snapshot("script.js")
    assert m.is_unchanged("script.js", content="const x = 1;") is True


def test_changed_content_invalidates():
    m = ContextReuseManager()
    m.record_read("script.js", content="const x = 1;")
    assert m.is_unchanged("script.js", content="const x = 2;") is False
    # Still has a snapshot (stale) — the caller must decide to refresh.
    assert m.has_snapshot("script.js") is True


def test_size_and_mtime_fast_path():
    m = ContextReuseManager()
    m.record_read("app.py", content="print(1)", size=7, mtime_ns=1000)
    # Same size, same mtime, no content given → unchanged.
    assert m.is_unchanged("app.py", size=7, mtime_ns=1000) is True
    # Different mtime → changed.
    assert m.is_unchanged("app.py", size=7, mtime_ns=2000) is False
    # Different size → changed.
    assert m.is_unchanged("app.py", size=99, mtime_ns=1000) is False


def test_unknown_file_is_never_unchanged():
    m = ContextReuseManager()
    assert m.has_snapshot("missing.py") is False
    assert m.is_unchanged("missing.py", content="x") is False


def test_invalidate_drops_snapshot():
    m = ContextReuseManager()
    m.record_read("a.txt", content="one")
    assert m.is_unchanged("a.txt", content="one") is True
    m.invalidate("a.txt")  # e.g. a write happened to the file
    assert m.has_snapshot("a.txt") is False
    assert m.is_unchanged("a.txt", content="one") is False


def test_invalidate_all_resets():
    m = ContextReuseManager()
    m.record_read("a.txt", content="one")
    m.record_read("b.txt", content="two")
    m.invalidate_all()
    assert m.to_dict()["count"] == 0


def test_content_hash_stable_across_types():
    m = ContextReuseManager()
    assert m.content_hash("hello") == m.content_hash(b"hello")
    assert m.content_hash("hello") != m.content_hash("world")


def test_snapshot_cap_evicts_oldest():
    m = ContextReuseManager(max_snapshots=2)
    m.record_read("a", content="1")
    m.record_read("b", content="2")
    m.record_read("c", content="3")
    assert m.has_snapshot("a") is False
    assert m.has_snapshot("b") is True
    assert m.has_snapshot("c") is True


def test_to_dict_reports_paths():
    m = ContextReuseManager()
    m.record_read("z.py", content="z")
    m.record_read("a.py", content="a")
    assert m.to_dict()["paths"] == ["a.py", "z.py"]

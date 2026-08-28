"""Tests for the caching subsystem (Phase A).

Covers: FileContentCache (LRU, invalidation, secrets, staleness),
SearchCache (TTL, key generation, eviction).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from harness_core.cache.file_cache import (
    CacheStats,
    FileContentCache,
    FileMetadata,
    SKIP_DIRS,
)
from harness_core.cache.search_cache import SearchCache


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with test files."""
    # Create source files
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "src" / "utils.py").write_text("def helper(): pass\n", encoding="utf-8")
    (tmp_path / "src" / "auth.py").write_text("class Auth: pass\n", encoding="utf-8")

    # Create test files
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_hello(): pass\n", encoding="utf-8")

    # Create config
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Test\n", encoding="utf-8")

    # Create secret files
    (tmp_path / ".env").write_text("SECRET_KEY=abc123\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("LOCAL_SECRET=xyz\n", encoding="utf-8")

    # Create .git directory (should be skipped)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git config\n", encoding="utf-8")

    # Create __pycache__ (should be skipped)
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"\x00")

    return tmp_path


@pytest.fixture
def file_cache(tmp_project: Path) -> FileContentCache:
    """Create a FileContentCache rooted at the temp project."""
    return FileContentCache(max_entries=50, workspace_root=tmp_project)


@pytest.fixture
def search_cache() -> SearchCache:
    """Create a SearchCache with short TTL for testing."""
    return SearchCache(max_entries=50, ttl_seconds=2.0)


# ── FileContentCache Tests ──────────────────────────────────────────────────


class TestFileContentCacheBasic:
    def test_get_miss(self, file_cache: FileContentCache):
        result = file_cache.get("nonexistent.py")
        assert result is None

    def test_put_and_get(self, file_cache: FileContentCache, tmp_project: Path):
        path = tmp_project / "src" / "main.py"
        content = file_cache.put(path)
        assert content is not None
        assert "hello" in content

        # Second get should be cache hit
        cached = file_cache.get(path)
        assert cached == content
        assert file_cache.stats.hits == 1

    def test_relative_path(self, file_cache: FileContentCache, tmp_project: Path):
        os.chdir(tmp_project)
        content = file_cache.put("src/main.py")
        assert content is not None
        cached = file_cache.get("src/main.py")
        assert cached == content

    def test_cache_size(self, file_cache: FileContentCache, tmp_project: Path):
        assert file_cache.size == 0
        file_cache.put(tmp_project / "src" / "main.py")
        assert file_cache.size == 1
        file_cache.put(tmp_project / "src" / "utils.py")
        assert file_cache.size == 2


class TestFileContentCacheLRU:
    def test_lru_eviction(self, tmp_project: Path):
        cache = FileContentCache(max_entries=3, workspace_root=tmp_project)
        cache.put(tmp_project / "src" / "main.py")
        cache.put(tmp_project / "src" / "utils.py")
        cache.put(tmp_project / "src" / "auth.py")
        assert cache.size == 3

        # Adding a 4th should evict the oldest
        cache.put(tmp_project / "pyproject.toml")
        assert cache.size == 3
        assert cache.stats.evictions == 1

    def test_lru_access_promotes(self, tmp_project: Path):
        cache = FileContentCache(max_entries=3, workspace_root=tmp_project)
        cache.put(tmp_project / "src" / "main.py")  # oldest
        cache.put(tmp_project / "src" / "utils.py")
        cache.put(tmp_project / "src" / "auth.py")  # newest

        # Access oldest — should promote it
        cache.get(tmp_project / "src" / "main.py")

        # Now add a 4th — should evict utils.py (now oldest)
        cache.put(tmp_project / "pyproject.toml")
        assert cache.contains(tmp_project / "src" / "main.py")  # promoted
        assert not cache.contains(tmp_project / "src" / "utils.py")  # evicted


class TestFileContentCacheInvalidation:
    def test_invalidate_specific(self, file_cache: FileContentCache, tmp_project: Path):
        path = tmp_project / "src" / "main.py"
        file_cache.put(path)
        assert file_cache.size == 1

        result = file_cache.invalidate(path)
        assert result is True
        assert file_cache.size == 0

    def test_invalidate_nonexistent(self, file_cache: FileContentCache, tmp_project: Path):
        result = file_cache.invalidate(tmp_project / "nope.py")
        assert result is False

    def test_invalidate_all(self, file_cache: FileContentCache, tmp_project: Path):
        file_cache.put(tmp_project / "src" / "main.py")
        file_cache.put(tmp_project / "src" / "utils.py")
        count = file_cache.invalidate_all()
        assert count == 2
        assert file_cache.size == 0

    def test_staleness_detection(self, file_cache: FileContentCache, tmp_project: Path):
        path = tmp_project / "src" / "main.py"
        file_cache.put(path)

        # Modify file — change mtime
        time.sleep(0.05)
        path.write_text("print('modified')\n", encoding="utf-8")

        # Cache should detect staleness
        result = file_cache.get(path)
        assert result is None  # stale, treated as miss
        assert file_cache.stats.invalidations >= 1

    def test_file_deleted(self, file_cache: FileContentCache, tmp_project: Path):
        path = tmp_project / "src" / "main.py"
        file_cache.put(path)

        # Delete file
        path.unlink()

        # Cache should detect deletion
        result = file_cache.get(path)
        assert result is None


class TestFileContentCacheSecrets:
    def test_env_files_excluded(self, file_cache: FileContentCache, tmp_project: Path):
        content = file_cache.put(tmp_project / ".env")
        assert content is None  # should not cache
        assert file_cache.size == 0
        assert file_cache.stats.secret_skips == 1

    def test_env_local_excluded(self, file_cache: FileContentCache, tmp_project: Path):
        content = file_cache.put(tmp_project / ".env.local")
        assert content is None

    def test_git_dir_excluded(self, file_cache: FileContentCache, tmp_project: Path):
        content = file_cache.put(tmp_project / ".git" / "config")
        assert content is None

    def test_pycache_excluded(self, file_cache: FileContentCache, tmp_project: Path):
        content = file_cache.put(tmp_project / "src" / "__pycache__" / "main.cpython-312.pyc")
        assert content is None


class TestFileContentCacheStats:
    def test_hit_rate(self, file_cache: FileContentCache, tmp_project: Path):
        path = tmp_project / "src" / "main.py"
        file_cache.put(path)  # miss (put reads from disk)
        file_cache.get(path)  # hit
        file_cache.get(path)  # hit
        file_cache.get(tmp_project / "nope.py")  # miss

        stats = file_cache.stats
        # put doesn't count as miss; get misses: 1 (nope.py), hits: 2
        assert stats.hits == 2
        assert stats.misses >= 1

    def test_to_dict(self, file_cache: FileContentCache, tmp_project: Path):
        d = file_cache.to_dict()
        assert "entries" in d
        assert "stats" in d
        assert d["entries"] == 0


class TestFileContentCacheMetadata:
    def test_metadata_available(self, file_cache: FileContentCache, tmp_project: Path):
        path = tmp_project / "src" / "main.py"
        file_cache.put(path)
        meta = file_cache.get_metadata(path)
        assert meta is not None
        assert meta.size > 0
        assert meta.line_count >= 1

    def test_content_hash(self, file_cache: FileContentCache, tmp_project: Path):
        path = tmp_project / "src" / "main.py"
        file_cache.put(path)
        meta = file_cache.get_metadata(path)
        assert meta is not None
        assert len(meta.content_hash) == 32  # MD5 hex


class TestFileContentCacheCustomContent:
    def test_put_with_content(self, file_cache: FileContentCache, tmp_project: Path):
        path = tmp_project / "src" / "main.py"
        content = file_cache.put(path, content="custom content")
        assert content == "custom content"


# ── SearchCache Tests ────────────────────────────────────────────────────────


class TestSearchCacheBasic:
    def test_get_miss(self, search_cache: SearchCache):
        key = SearchCache.make_key("glob", pattern="*.py")
        result = search_cache.get(key)
        assert result is None

    def test_put_and_get(self, search_cache: SearchCache):
        key = SearchCache.make_key("glob", pattern="*.py")
        search_cache.put(key, ["file1.py", "file2.py"])
        result = search_cache.get(key)
        assert result == ["file1.py", "file2.py"]

    def test_key_deterministic(self):
        key1 = SearchCache.make_key("grep", pattern="auth", path="src/")
        key2 = SearchCache.make_key("grep", pattern="auth", path="src/")
        assert key1 == key2

    def test_key_different_for_different_params(self):
        key1 = SearchCache.make_key("grep", pattern="auth")
        key2 = SearchCache.make_key("grep", pattern="user")
        assert key1 != key2

    def test_key_param_order_independent(self):
        key1 = SearchCache.make_key("glob", pattern="*.py", path="src/")
        key2 = SearchCache.make_key("glob", path="src/", pattern="*.py")
        assert key1 == key2


class TestSearchCacheTTL:
    def test_entry_expires(self, search_cache: SearchCache):
        key = SearchCache.make_key("glob", pattern="*.py")
        search_cache.put(key, ["result"])
        assert search_cache.get(key) == ["result"]

        # Wait for TTL
        time.sleep(2.5)
        result = search_cache.get(key)
        assert result is None  # expired

    def test_invalidate_stale(self, search_cache: SearchCache):
        key = SearchCache.make_key("glob", pattern="*.py")
        search_cache.put(key, ["result"])
        time.sleep(2.5)
        removed = search_cache.invalidate_stale()
        assert removed == 1
        assert search_cache.size == 0


class TestSearchCacheEviction:
    def test_lru_eviction(self):
        cache = SearchCache(max_entries=3, ttl_seconds=60.0)
        for i in range(4):
            key = SearchCache.make_key("glob", pattern=f"*.{i}")
            cache.put(key, [f"file{i}"])
        assert cache.size == 3

    def test_invalidate_all(self, search_cache: SearchCache):
        for i in range(5):
            key = SearchCache.make_key("glob", pattern=f"*.{i}")
            search_cache.put(key, [f"file{i}"])
        count = search_cache.invalidate_all()
        assert count == 5
        assert search_cache.size == 0


class TestSearchCacheStats:
    def test_hit_rate(self, search_cache: SearchCache):
        key = SearchCache.make_key("glob", pattern="*.py")
        search_cache.put(key, ["result"])
        search_cache.get(key)  # hit
        search_cache.get(key)  # hit
        SearchCache.make_key("grep", pattern="nope")
        search_cache.get("nonexistent")  # miss

        assert search_cache.hit_rate > 0

    def test_to_dict(self, search_cache: SearchCache):
        d = search_cache.to_dict()
        assert "entries" in d
        assert "hit_rate" in d

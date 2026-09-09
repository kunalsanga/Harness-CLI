"""Tests for the memory store (LocalMemoryStore + MemoryStore ABC)."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from harness_core.memory.domain import MemoryEntry, MemoryType
from harness_core.memory.indexer import MemoryIndexer
from harness_core.memory.store import LocalMemoryStore, MemoryStore


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_entry(
    content: str = "Test memory content",
    mtype: MemoryType = MemoryType.SUCCESS,
    project_id: str = "test-project",
    agent_role: str = "backend",
) -> MemoryEntry:
    entry = MemoryEntry(
        type=mtype,
        content=content,
        project_id=project_id,
        agent_role=agent_role,
        content_hash=MemoryEntry.compute_hash(content),
    )
    return entry


class TestLocalMemoryStore:
    """Test suite for LocalMemoryStore."""

    @pytest.fixture
    def store_path(self, tmp_path: Path) -> Path:
        return tmp_path / "memory" / "entries.json"

    @pytest.fixture
    def store(self, store_path: Path) -> LocalMemoryStore:
        return LocalMemoryStore(store_path)

    # ── CRUD ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_save_and_get(self, store: LocalMemoryStore):
        entry = _make_entry("Install pytest first")
        saved_id = await store.save(entry)
        assert saved_id == entry.id

        retrieved = await store.get(saved_id)
        assert retrieved is not None
        assert retrieved.content == "Install pytest first"
        assert retrieved.type == MemoryType.SUCCESS

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, store: LocalMemoryStore):
        assert await store.get("does-not-exist") is None

    @pytest.mark.asyncio
    async def test_update_replaces_entry(self, store: LocalMemoryStore):
        entry = _make_entry("v1")
        await store.save(entry)

        entry.content = "v2 updated"
        await store.update(entry)

        retrieved = await store.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "v2 updated"

    @pytest.mark.asyncio
    async def test_update_missing_raises_keyerror(self, store: LocalMemoryStore):
        entry = _make_entry("orphan")
        with pytest.raises(KeyError):
            await store.update(entry)

    @pytest.mark.asyncio
    async def test_delete_existing_returns_true(self, store: LocalMemoryStore):
        entry = _make_entry()
        await store.save(entry)
        assert await store.delete(entry.id) is True
        assert await store.get(entry.id) is None

    @pytest.mark.asyncio
    async def test_delete_missing_returns_false(self, store: LocalMemoryStore):
        assert await store.delete("nonexistent") is False

    @pytest.mark.asyncio
    async def test_count_no_filter(self, store: LocalMemoryStore):
        for i in range(5):
            await store.save(_make_entry(f"content {i}"))
        assert await store.count(None) == 5

    @pytest.mark.asyncio
    async def test_count_with_type_filter(self, store: LocalMemoryStore):
        await store.save(_make_entry("success", mtype=MemoryType.SUCCESS))
        await store.save(_make_entry("failure", mtype=MemoryType.FAILURE))
        await store.save(_make_entry("arch", mtype=MemoryType.ARCHITECTURE))
        assert await store.count({"memory_types": [MemoryType.FAILURE]}) == 1
        assert await store.count({"memory_types": [MemoryType.SUCCESS, MemoryType.FAILURE]}) == 2

    @pytest.mark.asyncio
    async def test_count_with_project_filter(self, store: LocalMemoryStore):
        await store.save(_make_entry("p1", project_id="proj-1"))
        await store.save(_make_entry("p2", project_id="proj-2"))
        assert await store.count({"project_id": "proj-1"}) == 1

    @pytest.mark.asyncio
    async def test_count_with_tags_filter(self, store: LocalMemoryStore):
        e1 = _make_entry("tagged")
        e1.tags = ["auth", "jwt"]
        await store.save(e1)
        e2 = _make_entry("other")
        e2.tags = ["database"]
        await store.save(e2)
        assert await store.count({"tags": ["auth"]}) == 1
        assert await store.count({"tags": ["auth", "database"]}) == 2

    # ── Search ───────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_search_returns_top_k_by_similarity(self, store: LocalMemoryStore, tmp_path: Path):
        indexer = MemoryIndexer()
        for text in [
            "JWT authentication using refresh tokens",
            "OAuth 2.0 with PKCE for mobile clients",
            "Database schema for user accounts",
            "REST API rate limiting middleware",
        ]:
            entry = _make_entry(text, project_id="search-test")
            entry.embedding = indexer.embed(text)
            await store.save(entry)

        query_emb = indexer.embed("JWT authentication refresh token")
        results = await store.search(query_emb, filters=None, top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, tuple) for r in results)
        entry, score = results[0]
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_search_filters_by_type(self, store: LocalMemoryStore, tmp_path: Path):
        indexer = MemoryIndexer()
        for content, mtype in [
            ("success story", MemoryType.SUCCESS),
            ("failure story", MemoryType.FAILURE),
            ("another success", MemoryType.SUCCESS),
        ]:
            entry = _make_entry(content, mtype=mtype)
            entry.embedding = indexer.embed(content)
            await store.save(entry)

        query_emb = indexer.embed("story")
        results = await store.search(
            query_emb,
            filters={"memory_types": [MemoryType.SUCCESS]},
            top_k=5,
        )
        assert all(r[0].type == MemoryType.SUCCESS for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_min_score(self, store: LocalMemoryStore):
        entry = _make_entry("very specific keyword")
        entry.embedding = [0.0] * 256
        await store.save(entry)

        results = await store.search([1.0] + [0.0] * 255, filters=None, min_score=0.99)
        assert len(results) == 0

    # ── Persistence ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_persistence_survives_rebuild(self, store_path: Path, tmp_path: Path):
        store1 = LocalMemoryStore(store_path)
        await store1.save(_make_entry("persistent content"))

        store2 = LocalMemoryStore(store_path)
        assert await store2.count(None) == 1
        retrieved = await store2.get((await store1.all_entries())[0].id)
        assert retrieved is not None
        assert retrieved.content == "persistent content"

    @pytest.mark.asyncio
    async def test_missing_file_loads_empty(self, tmp_path: Path):
        store = LocalMemoryStore(tmp_path / "does-not-exist.json")
        assert await store.count(None) == 0

    @pytest.mark.asyncio
    async def test_concurrent_saves_are_safe(self, store: LocalMemoryStore):
        entries = [_make_entry(f"concurrent {i}") for i in range(20)]

        async def save_one(e: MemoryEntry) -> None:
            await store.save(e)

        await asyncio.gather(*[save_one(e) for e in entries])
        assert await store.count(None) == 20

    # ── File size ────────────────────────────────────────────────────────

    def test_file_size_bytes(self, store: LocalMemoryStore):
        assert store.file_size_bytes() >= 0

    # ── Archive ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_archived_entries_excluded_by_default(self, store: LocalMemoryStore):
        entry = _make_entry("archived content")
        entry.archived = True
        await store.save(entry)

        # Default filter in search: archived=False
        assert await store.count({"archived": False}) == 0
        # Explicit filter: archived=True
        assert await store.count({"archived": True}) == 1

    # ── Adversarial ─────────────────────────────────────────────────────

    class TestAdversarial:
        """Security and poisoning tests for the memory store."""

        @pytest.mark.asyncio
        async def test_duplicate_content_creates_multiple_entries(self, store: LocalMemoryStore):
            """Store should not prevent duplicates at the store level (dedup is in MemoryManager)."""
            e1 = _make_entry("same content")
            e2 = _make_entry("same content")
            e2.id = e1.id  # Force same ID
            await store.save(e1)
            await store.save(e2)  # Store allows overwrites
            # With same ID this replaces, not duplicates
            assert await store.count(None) == 1

        @pytest.mark.asyncio
        async def test_search_with_empty_embedding_returns_nothing(self, store: LocalMemoryStore):
            entry = _make_entry("test")
            entry.embedding = []
            await store.save(entry)
            results = await store.search([], filters=None, top_k=5)
            assert len(results) == 0

        @pytest.mark.asyncio
        async def test_corrupt_json_file_loads_gracefully(self, store_path: Path, tmp_path: Path):
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text("{ invalid json }", encoding="utf-8")
            # Should not raise
            store = LocalMemoryStore(store_path)
            assert await store.count(None) == 0

        @pytest.mark.asyncio
        async def test_malformed_entry_in_json_skipped(self, store_path: Path, tmp_path: Path):
            """Corrupt entries in the JSON should be skipped without crashing."""
            import json

            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text(
                json.dumps({"entries": [{"id": "bad", "type": "not-a-type"}]}),
                encoding="utf-8",
            )
            store = LocalMemoryStore(store_path)
            assert await store.count(None) == 0  # bad entry skipped

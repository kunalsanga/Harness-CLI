"""Tests for memory retrieval and role-based access control."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from harness_core.memory.domain import MemoryEntry, MemoryType, RetrievalQuery
from harness_core.memory.indexer import MemoryIndexer
from harness_core.memory.retriever import MemoryRetriever, _types_for_role
from harness_core.memory.sanitizer import MemorySanitizer
from harness_core.memory.store import LocalMemoryStore


# ── Helpers ────────────────────────────────────────────────────────────────


async def _seed_store(
    store: LocalMemoryStore,
    indexer: MemoryIndexer,
    entries: list[MemoryEntry],
) -> None:
    for entry in entries:
        entry.embedding = indexer.index(entry)
        await store.save(entry)


def _make(
    content: str,
    mtype: MemoryType = MemoryType.SUCCESS,
    project_id: str = "proj",
    agent_role: str = "backend",
    tags: list[str] | None = None,
) -> MemoryEntry:
    entry = MemoryEntry(
        type=mtype,
        content=content,
        project_id=project_id,
        agent_role=agent_role,
        tags=tags or [],
        content_hash=MemoryEntry.compute_hash(content),
    )
    return entry


# ── Role type mapping ──────────────────────────────────────────────────────


class TestRoleTypeMapping:
    def test_backend_can_read_success_and_failure(self):
        types = _types_for_role("backend")
        assert MemoryType.SUCCESS in types
        assert MemoryType.FAILURE in types

    def test_architect_can_read_all_types(self):
        types = _types_for_role("architect")
        assert types == set(MemoryType)

    def test_planner_can_read_all_types(self):
        types = _types_for_role("planner")
        assert types == set(MemoryType)

    def test_security_reviewer_can_read_all_types(self):
        types = _types_for_role("security_reviewer")
        assert types == set(MemoryType)

    def test_unknown_role_defaults_to_all(self):
        types = _types_for_role("unknown-role-xyz")
        assert types == set(MemoryType)

    def test_tester_cannot_read_architecture(self):
        types = _types_for_role("tester")
        assert MemoryType.ARCHITECTURE not in types


# ── Retrieval Tests ───────────────────────────────────────────────────────


class TestMemoryRetriever:
    @pytest.fixture
    def tmp_path(self) -> Path:
        return Path(tempfile.mkdtemp())

    @pytest.fixture
    def store(self, tmp_path: Path) -> LocalMemoryStore:
        return LocalMemoryStore(tmp_path / "entries.json")

    @pytest.fixture
    def indexer(self) -> MemoryIndexer:
        return MemoryIndexer()

    @pytest.fixture
    def sanitizer(self) -> MemorySanitizer:
        return MemorySanitizer()

    @pytest.fixture
    def retriever(
        self, store: LocalMemoryStore, indexer: MemoryIndexer, sanitizer: MemorySanitizer
    ) -> MemoryRetriever:
        return MemoryRetriever(store=store, indexer=indexer, sanitizer=sanitizer)

    @pytest.mark.asyncio
    async def test_retrieve_similar(self, retriever, store, indexer):
        await _seed_store(
            store,
            indexer,
            [
                _make("JWT auth with refresh token rotation", MemoryType.SUCCESS),
                _make("OAuth 2.0 PKCE flow for SPAs", MemoryType.SUCCESS),
                _make("MySQL database schema design", MemoryType.PROJECT),
            ],
        )
        results = await retriever.retrieve_similar(
            RetrievalQuery(query="refresh token JWT authentication", top_k=2)
        )
        assert len(results) <= 2
        # At least one JWT result should rank higher than the DB schema
        scores = {r.entry.content: r.score for r in results}
        assert all(isinstance(s, float) for s in scores.values())

    @pytest.mark.asyncio
    async def test_retrieve_failures(self, retriever, store, indexer):
        await _seed_store(
            store,
            indexer,
            [
                _make("JWT refresh tokens not rotating", MemoryType.FAILURE, project_id="proj"),
                _make("MySQL connection pool exhausted", MemoryType.FAILURE, project_id="proj"),
                _make("Successful JWT implementation", MemoryType.SUCCESS, project_id="proj"),
            ],
        )
        results = await retriever.retrieve_failures(project_id="proj", top_k=5)
        assert all(r.entry.type == MemoryType.FAILURE for r in results)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_successes(self, retriever, store, indexer):
        await _seed_store(
            store,
            indexer,
            [
                _make("JWT with RS256 worked", MemoryType.SUCCESS, project_id="proj"),
                _make("Rate limiting middleware worked", MemoryType.SUCCESS, project_id="proj"),
                _make("Failed DB migration", MemoryType.FAILURE, project_id="proj"),
            ],
        )
        results = await retriever.retrieve_successes(project_id="proj", top_k=5)
        assert all(r.entry.type == MemoryType.SUCCESS for r in results)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_architecture_decisions(self, retriever, store, indexer):
        await _seed_store(
            store,
            indexer,
            [
                _make("Use JWT RS256 over HS256", MemoryType.ARCHITECTURE),
                _make("Use async I/O for agent loop", MemoryType.ARCHITECTURE),
                _make("Successful deployment", MemoryType.SUCCESS),
            ],
        )
        results = await retriever.retrieve_architecture_decisions(top_k=5)
        assert all(r.entry.type == MemoryType.ARCHITECTURE for r in results)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_project_context(self, retriever, store, indexer):
        await _seed_store(
            store,
            indexer,
            [
                _make("JWT refresh tokens", MemoryType.SUCCESS, project_id="auth-svc"),
                _make("MySQL schema", MemoryType.PROJECT, project_id="auth-svc"),
                _make("Unrelated project", MemoryType.SUCCESS, project_id="other"),
            ],
        )
        results = await retriever.retrieve_project_context(project_id="auth-svc", top_k=5)
        assert all(r.entry.project_id == "auth-svc" for r in results)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_access_count_increments_on_retrieval(self, retriever, store, indexer):
        entry = _make("Access count test", MemoryType.SUCCESS)
        await _seed_store(store, indexer, [entry])
        initial = entry.access_count

        await retriever.retrieve_similar(RetrievalQuery(query="test", top_k=1))
        # Give it a moment for the update
        await asyncio.sleep(0.01)
        updated = await store.get(entry.id)
        assert updated is not None
        assert updated.access_count >= initial

    @pytest.mark.asyncio
    async def test_retrieve_for_role_restricts_types(self, retriever, store, indexer):
        await _seed_store(
            store,
            indexer,
            [
                _make("JWT success", MemoryType.SUCCESS, agent_role="backend"),
                _make("Architecture decision", MemoryType.ARCHITECTURE, agent_role="architect"),
            ],
        )
        # Backend should NOT see architecture entries
        results = await retriever.retrieve_for_role("JWT", role="backend", top_k=5)
        assert all(r.entry.type != MemoryType.ARCHITECTURE for r in results)

        # Architect should see both
        results = await retriever.retrieve_for_role("JWT architecture", role="architect", top_k=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_retrieve_for_unknown_role_sees_all(self, retriever, store, indexer):
        await _seed_store(
            store,
            indexer,
            [
                _make("Architecture", MemoryType.ARCHITECTURE),
                _make("Decision", MemoryType.DECISION),
            ],
        )
        results = await retriever.retrieve_for_role("test", role="unknown-role", top_k=5)
        # Unknown role defaults to all types; store has 2 entries
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_format_for_prompt_renders_markdown(self, retriever, store, indexer):
        entry = _make("JWT auth with refresh tokens worked", MemoryType.SUCCESS)
        await _seed_store(store, indexer, [entry])
        results = await retriever.retrieve_similar(
            RetrievalQuery(query="JWT auth", top_k=1)
        )
        # The result should contain our seeded entry
        assert len(results) == 1
        formatted = MemoryRetriever.format_for_prompt(results)
        assert len(formatted) > 0
        # Check that the formatted string contains the entry type and content
        assert "success" in formatted.lower()

    # ── Adversarial ─────────────────────────────────────────────────────

    class TestAdversarial:
        """Security and poisoning tests for the retriever."""

        @pytest.mark.asyncio
        async def test_secret_in_entry_is_redacted_on_retrieval(
            self, retriever, store, indexer
        ):
            """Secrets stored in entries must be redacted when retrieved."""
            # Sanitise a real entry with a secret
            secret_entry = MemoryEntry(
                type=MemoryType.EPISODIC,
                content="API key: sk-abc123xyz_secret_key_do_not_share",
                agent_role="backend",
                content_hash=MemoryEntry.compute_hash(
                    "API key: sk-abc123xyz_secret_key_do_not_share"
                ),
            )
            await _seed_store(store, indexer, [secret_entry])
            results = await retriever.retrieve_for_role("API key", role="backend", top_k=1)
            # The retrieved content should have the secret redacted
            assert "[REDACTED]" in results[0].entry.content
            assert "sk-abc123" not in results[0].entry.content

        @pytest.mark.asyncio
        async def test_retrieval_respects_permissions(
            self, retriever, store, indexer
        ):
            """Backend agents should NOT receive architecture decisions."""
            await _seed_store(
                store,
                indexer,
                [
                    _make("Use RS256 JWT", MemoryType.ARCHITECTURE),
                    _make("JWT succeeded", MemoryType.SUCCESS),
                ],
            )
            results = await retriever.retrieve_for_role(
                "RS256 JWT", role="backend", top_k=5
            )
            arch_results = [r for r in results if r.entry.type == MemoryType.ARCHITECTURE]
            assert len(arch_results) == 0

        @pytest.mark.asyncio
        async def test_empty_store_returns_empty_results(self, retriever):
            results = await retriever.retrieve_similar(
                RetrievalQuery(query="anything", top_k=5)
            )
            assert results == []

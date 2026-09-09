"""Phase 8F / 8G — retrieval-augmented planning and execution (prompt level).

These tests instantiate the real ``Planner`` and ``WorkerAgent`` and prove that
memory context is actually injected into the system/agent prompt, and that the
injection respects role permissions.  The manager-level retrieval helpers are
covered in ``test_memory_manager.py`` / ``test_memory_retrieval.py``; this file
closes the loop at the class boundaries that Phase 8F and 8G specify.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_core.agents.domain import AgentContract, AgentRole, WorkspaceScope
from harness_core.agents.registry import AgentProfile, AgentRegistry
from harness_core.agents.worker import WorkerAgent
from harness_core.memory.graph import KnowledgeGraph
from harness_core.memory.indexer import MemoryIndexer
from harness_core.memory.manager import MemoryManager
from harness_core.memory.retention import MemoryRetentionPolicy, RetentionConfig
from harness_core.memory.retriever import MemoryRetriever
from harness_core.memory.sanitizer import MemorySanitizer
from harness_core.memory.store import LocalMemoryStore
from harness_core.observability.events import EventBus
from harness_core.planning.planner import Planner
from harness_core.providers.base import CompletionRequest, CompletionResponse


# ── Helpers ────────────────────────────────────────────────────────────────


def _manager(tmp_path: Path) -> MemoryManager:
    base = tmp_path / ".harness" / "memory"
    store = LocalMemoryStore(base / "entries.json")
    graph = KnowledgeGraph(storage_path=base / "graph.json")
    indexer = MemoryIndexer()
    sanitizer = MemorySanitizer()
    retriever = MemoryRetriever(store=store, indexer=indexer, sanitizer=sanitizer)
    return MemoryManager(
        store=store,
        graph=graph,
        retriever=retriever,
        indexer=indexer,
        sanitizer=sanitizer,
        retention_policy=MemoryRetentionPolicy(config=RetentionConfig()),
        enabled=True,
    )


class _CaptureProvider:
    """Fake ModelProvider that records the system prompt it was given."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.captured_system: str = ""

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        for message in request.messages:
            if message.get("role") == "system":
                self.captured_system = message.get("content", "")
        return CompletionResponse(content=self.reply, model="fake")


_VALID_PLAN = (
    '{"summary": "single task plan", "tasks": ['
    '{"task_id": "task_1", "title": "Build", "objective": "Build JWT auth", '
    '"role": "coder", "dependencies": [], "success_criteria": ["works"], '
    '"workspace_scope": "project", "priority": 5, "model_policy": "auto", '
    '"expected_artifacts": [], "resources": []}]}'
)


def _coder_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentProfile(
            name="coder",
            role=AgentRole.CODER,
            system_instructions="",
            workspace_scope=WorkspaceScope.PROJECT,
        )
    )
    return registry


def _worker(
    role: AgentRole,
    objective: str,
    memory: MemoryManager | None,
    project_id: str = "",
    task_id: str = "w1",
) -> WorkerAgent:
    contract = AgentContract(
        agent_id=f"agent-{task_id}",
        role=role,
        task_id=task_id,
        objective=objective,
        workspace_scope=WorkspaceScope.PROJECT,
    )
    return WorkerAgent(
        contract=contract,
        provider=None,
        tools=[],
        event_bus=EventBus(),
        workspace_path="/tmp/ws",
        message_bus=None,
        memory=memory,
        project_id=project_id,
    )


# ── Phase 8F: Planner prompt augmentation ──────────────────────────────────


class TestPlannerAugmentation:
    @pytest.mark.asyncio
    async def test_planner_injects_prior_memory_context(self, tmp_path: Path):
        mm = _manager(tmp_path)
        await mm.record_failure(
            task_id="prev-auth-1",
            description="JWT auth without refresh token rotation",
            error="Refresh tokens were never rotated - security issue",
            agent_role="backend",
            project_id="auth-svc",
            tags=["auth", "jwt"],
        )
        await mm.record_success(
            task_id="prev-auth-2",
            description="JWT auth with refresh token rotation",
            outcome="Refresh token rotation works correctly",
            agent_role="backend",
            project_id="auth-svc",
            tags=["auth", "jwt"],
        )

        provider = _CaptureProvider(_VALID_PLAN)
        planner = Planner(
            provider=provider,  # type: ignore[arg-type]
            registry=_coder_registry(),
            memory=mm,
            project_id="auth-svc",
        )
        result = await planner.plan("Build another JWT authentication system")

        assert provider.captured_system, "Planner must send a system prompt"
        assert "# Prior Project Context" in provider.captured_system
        assert "refresh token rotation" in provider.captured_system
        assert "Past Failures" in provider.captured_system or "Prior" in provider.captured_system
        # Memory failures must never fail the planning call itself
        assert result is not None

    @pytest.mark.asyncio
    async def test_planner_without_memory_has_no_context_section(self, tmp_path: Path):
        provider = _CaptureProvider(_VALID_PLAN)
        planner = Planner(
            provider=provider,  # type: ignore[arg-type]
            registry=_coder_registry(),
            memory=None,
            project_id="auth-svc",
        )
        await planner.plan("Build JWT auth")
        assert "# Prior Project Context" not in provider.captured_system

    @pytest.mark.asyncio
    async def test_planner_memory_failure_does_not_break_planning(self, tmp_path: Path):
        """A broken memory subsystem must not take planning down with it."""

        class _BrokenMemory:
            enabled = True

            async def get_context_for_planner(self, user_request: str, project_id: str = ""):
                raise RuntimeError("memory backend is down")

        provider = _CaptureProvider(_VALID_PLAN)
        planner = Planner(
            provider=provider,  # type: ignore[arg-type]
            registry=_coder_registry(),
            memory=_BrokenMemory(),  # type: ignore[arg-type]
            project_id="auth-svc",
        )
        result = await planner.plan("Build JWT auth")
        # Planning still succeeds: the injected context is empty, not fatal
        assert "# Prior Project Context" not in provider.captured_system
        assert result.success is True


# ── Phase 8G: Worker prompt augmentation ──────────────────────────────────


class TestWorkerAugmentation:
    @pytest.mark.asyncio
    async def test_worker_prompt_includes_historical_context(self, tmp_path: Path):
        mm = _manager(tmp_path)
        await mm.record_failure(
            task_id="prev-1",
            description="Refresh token implementation",
            error="Refresh tokens were not rotated on each request",
            agent_role="backend",
            project_id="auth-svc",
        )
        worker = _worker(
            role=AgentRole.BACKEND,
            objective="Implement JWT authentication with refresh tokens",
            memory=mm,
            project_id="auth-svc",
        )
        prompt = await worker._build_prompt_async()  # noqa: SLF001
        assert "Historical Context" in prompt
        assert "not rotated" in prompt

    @pytest.mark.asyncio
    async def test_worker_without_memory_has_no_context(self):
        worker = _worker(
            role=AgentRole.BACKEND,
            objective="Implement JWT auth",
            memory=None,
        )
        prompt = await worker._build_prompt_async()  # noqa: SLF001
        assert "Historical Context" not in prompt

    @pytest.mark.asyncio
    async def test_worker_context_respects_role_permissions(self, tmp_path: Path):
        """Backend workers must not receive ARCHITECTURE decisions in-prompt."""
        mm = _manager(tmp_path)
        await mm.record_architecture(
            decision="Use RS256 asymmetric signing for JWT",
            rationale="Asymmetric keys are safer across services",
            agent_role="architect",
            project_id="auth-svc",
        )

        backend_worker = _worker(
            role=AgentRole.BACKEND,
            objective="Implement JWT authentication",
            memory=mm,
            project_id="auth-svc",
        )
        backend_prompt = await backend_worker._build_prompt_async()  # noqa: SLF001
        assert "RS256" not in backend_prompt

        architect_worker = _worker(
            role=AgentRole.ARCHITECT,
            objective="Decide JWT signing strategy",
            memory=mm,
            project_id="auth-svc",
        )
        architect_prompt = await architect_worker._build_prompt_async()  # noqa: SLF001
        assert "RS256" in architect_prompt

    @pytest.mark.asyncio
    async def test_worker_memory_failure_does_not_break_prompt(self, tmp_path: Path):
        class _BrokenMemory:
            enabled = True

            async def get_context_for_worker(self, objective: str, role: str, project_id: str = ""):
                raise RuntimeError("memory backend is down")

        worker = _worker(
            role=AgentRole.BACKEND,
            objective="Implement JWT auth",
            memory=_BrokenMemory(),  # type: ignore[arg-type]
        )
        prompt = await worker._build_prompt_async()  # noqa: SLF001
        # Agent prompt still builds without the context block
        assert "Implement JWT auth" in prompt
        assert "Historical Context" not in prompt

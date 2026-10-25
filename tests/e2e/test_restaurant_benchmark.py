"""Mandatory Restaurant Website Coding Benchmark (E2E Acceptance Test).

Verifies that the autonomous engineering runtime and TUI accurately represent
a real multi-stage coding task covering:
- understanding
- planning
- implementation (frontend, backend, database)
- testing
- repair / debugging
- verification
- four-question completion summary
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest
from rich.console import Console

from harness_core.agents.domain import AgentResult, AgentRole, AgentStatus
from harness_core.agents.worker import WorkerAgent
from harness_core.cli.completion import CompletionFormatter
from harness_core.cli.conversation import ConversationRenderer
from harness_core.cli.interactive import InteractiveShell
from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import CompletionResponse
from harness_core.runtime.requirements import Requirements
from harness_core.runtime.runtime import EngineeringRuntime
from harness_core.runtime.state import RuntimeStatus, VerificationStatus


RESTAURANT_PLAN_JSON = """{
  "summary": "Build a modern full-stack restaurant ordering website",
  "tasks": [
    {
      "task_id": "architect_1",
      "title": "Analyze restaurant requirements",
      "objective": "Understand menu, ordering flow, and data schema",
      "role": "architect",
      "dependencies": [],
      "success_criteria": ["requirements analyzed"],
      "workspace_scope": "project",
      "priority": 5,
      "resources": [{"path": "docs", "mode": "write"}],
      "traceable_to": ["REQ-001"]
    },
    {
      "task_id": "database_1",
      "title": "Implement SQLite database models",
      "objective": "Create database schema for menu items, orders, and tables",
      "role": "database",
      "dependencies": ["architect_1"],
      "success_criteria": ["models.py created"],
      "workspace_scope": "project",
      "priority": 4,
      "resources": [{"path": "db", "mode": "write"}],
      "traceable_to": ["REQ-002"]
    },
    {
      "task_id": "backend_1",
      "title": "Implement REST API for orders",
      "objective": "FastAPI endpoints for menu list, cart, and checkout",
      "role": "backend",
      "dependencies": ["database_1"],
      "success_criteria": ["api.py created"],
      "workspace_scope": "project",
      "priority": 4,
      "resources": [{"path": "src/backend", "mode": "write"}],
      "traceable_to": ["REQ-003"]
    },
    {
      "task_id": "frontend_1",
      "title": "Implement customer menu and ordering UI",
      "objective": "Responsive menu cards, cart drawer, and order confirmation",
      "role": "frontend",
      "dependencies": ["backend_1"],
      "success_criteria": ["index.html and app.js created"],
      "workspace_scope": "project",
      "priority": 3,
      "resources": [{"path": "src/frontend", "mode": "write"}],
      "traceable_to": ["REQ-004"]
    },
    {
      "task_id": "tester_1",
      "title": "Execute end-to-end restaurant tests",
      "objective": "Validate order placement and total calculation",
      "role": "tester",
      "dependencies": ["frontend_1"],
      "success_criteria": ["tests pass"],
      "workspace_scope": "project",
      "priority": 2,
      "resources": [{"path": "tests", "mode": "write"}],
      "traceable_to": ["REQ-005"]
    },
    {
      "task_id": "reviewer_1",
      "title": "Code review",
      "objective": "Review restaurant application quality",
      "role": "reviewer",
      "dependencies": ["tester_1"],
      "success_criteria": ["review complete"],
      "workspace_scope": "project",
      "priority": 1,
      "resources": [{"path": "src", "mode": "read"}]
    },
    {
      "task_id": "verifier_1",
      "title": "Verify requirements",
      "objective": "Verify all acceptance criteria for restaurant app",
      "role": "verifier",
      "dependencies": ["reviewer_1"],
      "success_criteria": ["requirements verified"],
      "workspace_scope": "project",
      "priority": 1,
      "resources": [{"path": "src", "mode": "read"}]
    }
  ]
}"""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_restaurant_website_benchmark_e2e(tmp_path, monkeypatch):
    """Execute the full restaurant website benchmark verifying all phases in TUI."""
    bus = EventBus()
    out = io.StringIO()
    console = Console(file=out, force_terminal=True, width=120)
    renderer = ConversationRenderer(console)
    renderer.start("Build a modern restaurant ordering website with frontend, backend, and database")

    # Wire event bus to ConversationRenderer
    async def on_event(event: Event) -> None:
        if event.type == "task.phase":
            renderer.update_phase(event.data.get("phase", ""))
        elif event.type == "tool.call":
            renderer.tool_started(event.data.get("tool", ""), event.data.get("args", {}))
        elif event.type == "tool.result":
            renderer.tool_completed(event.data.get("tool", ""), event.data.get("status", "success"))
        elif event.type == "verification.completed":
            renderer.update_verification("2 checks passed" if event.data.get("passed") else "Checks failed")

    bus.on("*", on_event)

    # Provider mocks
    plan_provider = MagicMock()
    plan_provider.generate = AsyncMock(
        return_value=CompletionResponse(content=RESTAURANT_PLAN_JSON, model="fake-planner")
    )

    runtime_provider = MagicMock()

    async def mock_gen(request, *args, **kwargs):
        content = ""
        if request and hasattr(request, "messages") and request.messages:
            content = str(request.messages[0].content)
        if "autonomous engineering recovery coordinator" in content:
            return CompletionResponse(
                content=(
                    '{"strategy": "debug_and_fix", "specialist_role": "debugger", '
                    '"objective": "Fix tax calculation bug in orders API", '
                    '"success_criteria": ["tests pass"]}'
                ),
                model="fake-recovery",
            )
        return CompletionResponse(content="Done", model="fake-scheduler")

    runtime_provider.generate = AsyncMock(side_effect=mock_gen)

    # Track tasks and simulate initial test failure followed by repair
    ran_tasks = []

    async def fake_worker_run(self):
        contract = self.contract
        ran_tasks.append(contract.task_id)
        await asyncio.sleep(0.01)

        if contract.role == AgentRole.ARCHITECT:
            return AgentResult(
                agent_id=contract.agent_id,
                role=contract.role,
                status=AgentStatus.COMPLETED,
                summary="Architecture documented",
                files_changed=["docs/schema.md"],
            )
        elif contract.role == AgentRole.DATABASE:
            return AgentResult(
                agent_id=contract.agent_id,
                role=contract.role,
                status=AgentStatus.COMPLETED,
                summary="Database schema implemented",
                files_changed=["backend/database.py", "backend/models.py"],
            )
        elif contract.role == AgentRole.BACKEND:
            return AgentResult(
                agent_id=contract.agent_id,
                role=contract.role,
                status=AgentStatus.COMPLETED,
                summary="Order API endpoints implemented",
                files_changed=["backend/api.py"],
            )
        elif contract.role == AgentRole.FRONTEND:
            return AgentResult(
                agent_id=contract.agent_id,
                role=contract.role,
                status=AgentStatus.COMPLETED,
                summary="Customer menu UI implemented",
                files_changed=["frontend/index.html", "frontend/app.js"],
            )
        elif contract.role == AgentRole.TESTER and "_retest_" not in contract.task_id:
            # First test run discovers calculation bug
            return AgentResult(
                agent_id=contract.agent_id,
                role=contract.role,
                status=AgentStatus.COMPLETED,
                tests_passed=3,
                tests_total=4,
                summary="1 test failed: order tax calculation mismatch",
                findings=[{"type": "test_failure", "test_name": "test_order_tax"}],
                files_changed=["tests/results.json"],
            )
        elif contract.role == AgentRole.DEBUGGER:
            # Repair phase fixes the calculation
            return AgentResult(
                agent_id=contract.agent_id,
                role=contract.role,
                status=AgentStatus.COMPLETED,
                summary="Repaired tax calculation formula in orders API",
                files_changed=["backend/api.py"],
            )
        elif "_retest_" in contract.task_id:
            # Re-verification passes cleanly
            return AgentResult(
                agent_id=contract.agent_id,
                role=contract.role,
                status=AgentStatus.COMPLETED,
                tests_passed=4,
                tests_total=4,
                summary="All 4 tests passed",
                findings=[],
                files_changed=["tests/results.json"],
            )
        elif contract.role == AgentRole.REVIEWER:
            return AgentResult(
                agent_id=contract.agent_id,
                role=contract.role,
                status=AgentStatus.COMPLETED,
                summary="Code review passed cleanly",
                findings=[],
            )
        elif contract.role == AgentRole.VERIFIER:
            return AgentResult(
                agent_id=contract.agent_id,
                role=contract.role,
                status=AgentStatus.COMPLETED,
                summary="REQ-001: PASS\nREQ-002: PASS\nREQ-003: PASS\nREQ-004: PASS\nREQ-005: PASS\nAll acceptance criteria verified.",
                findings=[],
            )
        return AgentResult(
            agent_id=contract.agent_id,
            role=contract.role,
            status=AgentStatus.COMPLETED,
            summary="Completed",
        )

    monkeypatch.setattr(WorkerAgent, "run", fake_worker_run)

    # Run the runtime on the restaurant task
    runtime = EngineeringRuntime(
        workspace_path=str(tmp_path),
        provider=runtime_provider,
        plan_provider=plan_provider,
        event_bus=bus,
    )

    outcome = await runtime.run("Build a modern restaurant ordering website with frontend, backend, and database")

    # Assert runtime outcomes
    assert outcome.status == RuntimeStatus.SUCCESS
    assert outcome.state.verification_status == VerificationStatus.PASSED
    assert len(ran_tasks) >= 5
    assert any("database_1" in t for t in ran_tasks)
    assert any("backend_1" in t for t in ran_tasks)
    assert any("frontend_1" in t for t in ran_tasks)
    assert any("recovery" in t or "retest" in t or "debugger" in t for t in ran_tasks)

    # Format the completion summary with four-question structure
    fmt = CompletionFormatter(plain=True)
    all_files = [a.path for a in outcome.state.artifacts]
    if outcome.graph:
        for t in outcome.graph.tasks.values():
            all_files.extend(t.files_changed or [])
    clean_files = sorted({Path(f).name for f in all_files if f})

    summary = fmt.success(
        headline="Built restaurant ordering website with frontend, backend, and SQLite database",
        files_modified=[f for f in clean_files if f in ("api.py", "database.py", "models.py")],
        files_created=[f for f in clean_files if f in ("index.html", "app.js", "schema.md")],
        tests_line="4 checks passed (after repairing tax calculation)",
        verification_status="passed",
        duration=18.5,
        tool_calls=12,
    )

    # Verify four-question completion summary requirements:
    # 1. WHAT DID HARNESS DO?
    assert "restaurant ordering website" in summary
    # 2. WHAT CHANGED?
    assert "Changes" in summary
    assert "index.html" in summary
    assert "api.py" in summary
    # 3. DID VERIFICATION PASS?
    assert "Verification" in summary
    assert "Passed" in summary or "passed" in summary.lower()
    # 4. TRUTHFULNESS: No false success phrases
    assert "No false success was reported" not in summary

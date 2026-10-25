"""Unit tests for task completion reliability, timeout failover, and honest plan reporting."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from harness_core.agent.todos import (
    apply_tool_result,
    reconcile_on_evidence,
    TodoItem,
    TodoStatus,
)
from harness_core.agent.types import TaskPlan, Task, TaskStatus, ToolCall, ToolResult, ToolResultStatus
from harness_core.routing.fallback import (
    FallbackEngine,
    FallbackConfig,
    RetryConfig,
    ERROR_CATEGORY_TIMEOUT,
    ERROR_CATEGORY_RATE_LIMIT,
)
from harness_core.providers.base import CompletionRequest, CompletionResponse, ModelProvider


class TimeoutMockProvider(ModelProvider):
    """Provider that simulates timeouts or delays."""

    def __init__(self, name: str = "timeout_provider", timeout_on: set[str] | None = None):
        self._name = name
        self.api_key = "test-key"
        self.timeout_on = timeout_on or set()
        self.call_count: dict[str, int] = {}

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        mid = request.model
        self.call_count[mid] = self.call_count.get(mid, 0) + 1
        if mid in self.timeout_on:
            raise TimeoutError(f"Request timed out for {mid}")
        return CompletionResponse(
            content=f"Response from {mid}",
            model=mid,
        )

    async def stream(self, request: CompletionRequest):
        yield ""

    async def list_models(self):
        return []

    async def health_check(self) -> bool:
        return True


class QuotaMockProvider(ModelProvider):
    """Provider that simulates account-wide quota exhaustion."""

    def __init__(self, name: str = "openrouter"):
        self._name = name
        self.api_key = "test-key"
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        self.call_count += 1
        raise RuntimeError("429 Rate limit exceeded: free model request limit reached for today.")

    async def stream(self, request: CompletionRequest):
        yield ""

    async def list_models(self):
        return []

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_timeout_fails_over_immediately_without_retrying_same_model():
    """When a model times out, fallback must NOT retry the same hung model.
    It must fail over immediately to the next candidate model in the chain."""
    config = FallbackConfig(
        max_fallback_models=2,
        retry=RetryConfig(max_retries=2, base_delay_seconds=0.01),
        model_attempt_timeout_seconds=5.0,
        total_timeout_seconds=10.0,
    )
    engine = FallbackEngine(fallback_config=config)

    # model-1 times out; model-2 succeeds
    provider = TimeoutMockProvider(timeout_on={"model-1"})
    model_chain = [("model-1", provider), ("model-2", provider)]

    request = CompletionRequest(messages=[{"role": "user", "content": "hello"}])
    result = await engine.execute(request, model_chain)

    assert result.succeeded is True
    assert result.model_used == "model-2"
    # model-1 should have been attempted exactly ONCE (no redundant retries on timeout)
    assert provider.call_count.get("model-1") == 1
    assert provider.call_count.get("model-2") == 1


@pytest.mark.asyncio
async def test_account_wide_quota_fails_promptly():
    """When an account-wide free quota limit (429) is hit, fallback must fail promptly."""
    config = FallbackConfig(
        max_fallback_models=3,
        retry=RetryConfig(max_retries=1, base_delay_seconds=0.01),
        total_timeout_seconds=10.0,
    )
    engine = FallbackEngine(fallback_config=config)

    provider = QuotaMockProvider()
    model_chain = [("free-1", provider), ("free-2", provider)]

    request = CompletionRequest(messages=[{"role": "user", "content": "code"}])
    result = await engine.execute(request, model_chain)

    assert result.succeeded is False
    assert result.error_category == ERROR_CATEGORY_RATE_LIMIT
    assert "Account-wide free tier or rate limit reached" in (result.final_error or "")
    # Should not keep trying more models on the same exhausted quota
    assert provider.call_count == 1


def test_edit_file_does_not_falsely_complete_implementation_todo():
    """A single file edit marks an implementation TODO IN_PROGRESS, never prematurely COMPLETED."""
    plan = TaskPlan()
    plan.add_spec("Add Enter keydown event listener to calculator", category="implement")
    item = plan.items[0]
    assert item.status == TodoStatus.PENDING

    call = ToolCall(tool_name="edit_file", arguments={"path": "script.js", "old_string": "a", "new_string": "b"})
    res = ToolResult(status=ToolResultStatus.SUCCESS, output="Edited script.js")

    affected = apply_tool_result(plan, call, res)
    assert affected is not None
    # Must be IN_PROGRESS, not COMPLETED
    assert affected.status == TodoStatus.IN_PROGRESS
    assert affected.evidence is not None
    assert affected.evidence.get("tool") == "edit_file"


def test_reconcile_on_task_failure_flags_unverified_todos():
    """On task failure, reconcile_on_evidence must revert or fail unverified implementation items."""
    plan = TaskPlan()
    plan.add_spec("Inspect script.js", category="inspect")
    plan.add_spec("Add keydown handler", category="implement")

    # Suppose inspect succeeded and was completed
    plan.complete_id(plan.items[0].id)
    # Suppose implementation was in progress or prematurely marked complete
    plan.items[1].status = TodoStatus.COMPLETED

    changed = reconcile_on_evidence(
        plan,
        did_inspect=True,
        did_implement=True,
        tests_passed=None,
        verified=None,
        did_commit=False,
        did_push=False,
        tests_run=0,
        task_failed=True,
    )

    # Item 1 (inspect) remains completed
    assert plan.items[0].status == TodoStatus.COMPLETED
    # Item 2 (implement) was unverified, so on task_failed=True it MUST NOT remain COMPLETED!
    assert plan.items[1].status == TodoStatus.FAILED
    assert plan.items[1].evidence.get("unverified") is True


def test_reconcile_on_success_requires_evidence():
    """On successful task, implementation completes if did_implement and verification did not fail."""
    plan = TaskPlan()
    plan.add_spec("Add keydown handler", category="implement")
    plan.items[0].status = TodoStatus.IN_PROGRESS

    changed = reconcile_on_evidence(
        plan,
        did_inspect=True,
        did_implement=True,
        tests_passed=True,
        verified=True,
        did_commit=False,
        did_push=False,
        tests_run=1,
        task_failed=False,
    )

    assert plan.items[0].status == TodoStatus.COMPLETED


def test_balanced_diff_preview():
    """Diff preview must show both added and removed lines when both exist."""
    from rich.console import Console
    from harness_core.cli.conversation import ConversationRenderer

    console = Console(record=True, width=80)
    conv = ConversationRenderer(console=console, plain=False)

    diff_text = """--- a/script.js
+++ b/script.js
@@ -1,4 +1,4 @@
-const old1 = 1;
-const old2 = 2;
+const new1 = 10;
+const new2 = 20;
"""
    preview = {"path": "script.js", "added": 2, "removed": 2, "diff": diff_text}
    group = conv._render_diff_preview(preview)
    assert group is not None
    # Render group to text
    with console.capture() as capture:
        console.print(group)
    output = capture.get()
    assert "-const old1" in output
    assert "+const new1" in output


def test_render_completion_reports_unrun_verification_on_failure():
    """When a task fails with modified files, render_completion must clearly report unrun tests and verification."""
    from rich.console import Console
    from harness_core.cli.conversation import ConversationRenderer

    console = Console(record=True, width=80)
    conv = ConversationRenderer(console=console, plain=False)

    with console.capture() as capture:
        conv.render_completion(
            elapsed=10.0,
            success=False,
            status="failed",
            files_changed=["script.js"],
        )
    output = capture.get()
    assert "Failed" in output
    assert "script.js" in output
    assert "Tests  None run" in output
    assert "Verification  Not completed" in output


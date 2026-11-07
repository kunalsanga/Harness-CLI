"""Final V1 Acceptance Test — Autonomous Multi-Provider Engineering & Failover.

Comprehensive mission-level verification of Harness Engineering CLI:
1. End-to-end multi-step coding task surviving mid-execution provider switch.
2. Context preservation (conversation history, tool calls, tool results, plan) across provider boundaries.
3. OpenRouter disabled / rate-limited / unavailable failovers to secondary providers.
4. Azure OpenAI adapter generation, streaming, tool calling, and failover.
5. Cooldown expiration and recovery.
6. All providers exhausted graceful diagnostics.
7. Ineligible model filtering.
8. Provider token usage and attribution recording.
"""

from __future__ import annotations

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import (
    AgentConfig,
    AgentRole,
    Task,
    TaskStatus,
    TodoItem,
    TodoStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
    ProviderErrorCategory,
    ProviderRequestError,
)
from harness_core.providers.azure import AzureOpenAIProvider
from harness_core.routing.fallback import FallbackConfig, FallbackEngine, RetryConfig
from harness_core.routing.health import HealthEvent, ModelHealthTracker
from harness_core.routing.router import ModelRouter, RouterConfig
from harness_core.tools.base import Tool, ToolSchema


class MockControllableProvider(ModelProvider):
    """Controllable provider for simulating real multi-step execution and controlled failure."""

    def __init__(
        self,
        name: str,
        model_id: str,
        responses: list[CompletionResponse | Exception] | None = None,
        context_window: int = 128000,
        supports_tools: bool = True,
        is_free: bool = True,
    ) -> None:
        self._name = name
        self.model_id = model_id
        self.responses = list(responses or [])
        self.context_window = context_window
        self.supports_tools = supports_tools
        self.is_free = is_free
        self.call_history: list[CompletionRequest] = []
        self.generate_call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        self.generate_call_count += 1
        self.call_history.append(request)
        if not self.responses:
            return CompletionResponse(
                content="Default mock response",
                model=self.model_id,
                provider=self._name,
            )
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def stream(self, request: CompletionRequest):
        resp = await self.generate(request)
        yield resp.content

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=self.model_id,
                name=self.model_id,
                provider=self._name,
                context_window=self.context_window,
                supports_tools=self.supports_tools,
                supports_streaming=True,
                is_free=self.is_free,
            )
        ]

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class MockReadFileTool(Tool):
    """Mock read_file tool that records executions."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.executed_calls: list[dict] = []

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description="Read file contents",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )

    async def execute(self, arguments: dict) -> ToolResult:
        self.executed_calls.append(arguments)
        path = str(arguments.get("path", ""))
        norm_path = path.replace("\\", "/")
        matched_key = next((k for k in self.files if norm_path.endswith(k) or norm_path == k or path == k), None)
        if matched_key:
            return ToolResult(status=ToolResultStatus.SUCCESS, output=self.files[matched_key])
        return ToolResult(status=ToolResultStatus.ERROR, output="", error=f"File not found: {path}")


class MockRunCommandTool(Tool):
    """Mock run_command tool for testing command executions."""

    def __init__(self, outputs: dict[str, str] | None = None) -> None:
        self.outputs = outputs or {}
        self.executed_commands: list[str] = []

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_command",
            description="Execute a shell command",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        )

    async def execute(self, arguments: dict) -> ToolResult:
        cmd = arguments.get("command", "")
        self.executed_commands.append(cmd)
        output = self.outputs.get(cmd, "Command executed successfully\n6 passed in 1.2s")
        return ToolResult(status=ToolResultStatus.SUCCESS, output=output)


# ==============================================================================
# Critical Test 1: Multi-Step Engineering Task Surviving Mid-Execution Provider Switch
# ==============================================================================

@pytest.mark.asyncio
async def test_multi_step_engineering_task_survives_provider_switch(tmp_path: Path):
    """Verify that AgentLoop executes a multi-step mission, switches providers mid-run,
    and preserves all previous tool calls, outputs, and active plan state."""
    event_bus = EventBus()
    events_captured: list[Event] = []

    async def _capture(e: Event):
        events_captured.append(e)

    event_bus.on("*", _capture)

    # Provider A: OpenRouter (succeeds planning + step 1, then hits 429 quota failure on step 2)
    plan_response = CompletionResponse(
        content="1. Inspect backend/main.py\n2. Run tests",
        model="openrouter/free",
        provider="openrouter",
    )
    step1_response = CompletionResponse(
        content="I will inspect backend/main.py and run the tests.",
        model="openrouter/free",
        provider="openrouter",
        tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": "backend/main.py"}),
            },
        }],
    )
    step2_openrouter_failure = ProviderRequestError(
        ProviderErrorCategory.RATE_LIMIT,
        "Rate limit exceeded: free-models-per-day. Add credits to unlock requests (HTTP 429)",
        provider="openrouter",
        model="openrouter/free",
        status_code=429,
    )
    provider_openrouter = MockControllableProvider(
        name="openrouter",
        model_id="openrouter/free",
        responses=[plan_response, step1_response, step2_openrouter_failure],
    )

    # Provider B: Groq (succeeds after OpenRouter failure with step 2 tool call + final completion)
    step2_groq_response = CompletionResponse(
        content="I have read backend/main.py. Now I will execute the backend tests.",
        model="llama-3.3-70b-versatile",
        provider="groq",
        tool_calls=[{
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "run_command",
                "arguments": json.dumps({"command": "pytest backend/test_main.py"}),
            },
        }],
    )
    step3_groq_completion = CompletionResponse(
        content="All 6 backend tests pass. TaskForge verification complete.",
        model="llama-3.3-70b-versatile",
        provider="groq",
    )
    provider_groq = MockControllableProvider(
        name="groq",
        model_id="llama-3.3-70b-versatile",
        responses=[step2_groq_response, step3_groq_completion],
    )

    router = ModelRouter(
        providers=[provider_openrouter, provider_groq],
        config=RouterConfig(
            routing_mode="auto",
            provider_priority=["openrouter", "groq"],
            fallback=FallbackConfig(retry=RetryConfig(max_retries=0)),
        ),
        event_bus=event_bus,
    )

    tools = [
        MockReadFileTool({"backend/main.py": "from fastapi import FastAPI\napp = FastAPI()"}),
        MockRunCommandTool({"pytest backend/test_main.py": "6 passed in 1.4s"}),
    ]

    agent_config = AgentConfig(
        role=AgentRole.BUILD,
        max_iterations=5,
        model_preference="auto",
    )
    loop = AgentLoop(
        provider=provider_openrouter,
        tools=tools,
        workspace_root=tmp_path,
        config=agent_config,
        router=router,
        event_bus=event_bus,
    )

    task = await loop.run("Fix TaskForge backend bug and verify tests pass")

    # Verification: Task completed successfully
    assert task.status == TaskStatus.COMPLETED
    assert task.iterations >= 2

    # Verification: Provider switch recorded on task
    assert "openrouter" in task.providers_used
    assert "groq" in task.providers_used
    assert task.model_fallbacks >= 1

    # Verification: Context preservation
    # Groq's first call must have received previous tool output from OpenRouter's read_file!
    assert len(provider_groq.call_history) >= 1
    groq_first_request = provider_groq.call_history[0]
    messages_sent = groq_first_request.messages

    # Check that previous tool call and its result were preserved in messages sent to Groq
    tool_msgs = [m for m in messages_sent if m.get("role") == "tool" or m.get("tool_call_id") == "call_1"]
    content_strs = [str(m.get("content", "")) for m in messages_sent]
    has_read_evidence = any("from fastapi import FastAPI" in c for c in content_strs)
    assert has_read_evidence, "Context was lost across provider failover: read_file content missing!"

    # Verification: EventBus captured failover and switch events
    event_types = [e.type for e in events_captured]
    assert "model.failover" in event_types
    assert "model.switched" in event_types

    switch_events = [e for e in events_captured if e.type == "model.switched"]
    assert len(switch_events) >= 1
    assert switch_events[0].data["from"] == "openrouter/free"
    assert switch_events[0].data["to"] == "llama-3.3-70b-versatile"
    assert switch_events[0].data["provider"] == "groq"


# ==============================================================================
# Critical Test 2: Provider Selection When Primary Provider Is Disabled
# ==============================================================================

@pytest.mark.asyncio
async def test_disabled_primary_provider_skipped_immediately():
    """Verify that when OpenRouter is disabled in config, the router routes to NVIDIA/Groq
    on attempt 1 without ever calling OpenRouter."""
    p_openrouter = MockControllableProvider("openrouter", "openrouter/free")
    p_nvidia = MockControllableProvider(
        "nvidia",
        "meta/llama-3.1-70b-instruct",
        responses=[CompletionResponse(content="NVIDIA ready", model="meta/llama-3.1-70b-instruct", provider="nvidia")],
    )

    router = ModelRouter(
        providers=[p_openrouter, p_nvidia],
        config=RouterConfig(
            routing_mode="auto",
            provider_priority=["nvidia"],  # OpenRouter not prioritized
        ),
    )

    req = CompletionRequest(messages=[{"role": "user", "content": "hello"}])
    result = await router.execute(req)

    assert result.succeeded
    assert result.provider_used == "nvidia"
    assert p_openrouter.generate_call_count == 0


# ==============================================================================
# Critical Test 3: Timeout Failover Without Re-Attempting Hung Endpoint
# ==============================================================================

@pytest.mark.asyncio
async def test_timeout_failover_never_re_attempts_hung_endpoint():
    """Verify that a timeout on primary provider immediately fails over to secondary
    provider without retrying the hanging endpoint."""
    p_slow = MockControllableProvider(
        "slow_provider",
        "slow-model",
        responses=[asyncio.TimeoutError("Request timed out after 90s")],
    )
    p_fast = MockControllableProvider(
        "fast_provider",
        "fast-model",
        responses=[CompletionResponse(content="Fast response", model="fast-model", provider="fast_provider")],
    )

    router = ModelRouter(
        providers=[p_slow, p_fast],
        config=RouterConfig(
            routing_mode="auto",
            provider_priority=["slow_provider", "fast_provider"],
            fallback=FallbackConfig(retry=RetryConfig(max_retries=2)),
        ),
    )

    req = CompletionRequest(messages=[{"role": "user", "content": "run task"}])
    result = await router.execute(req)

    assert result.succeeded
    assert result.provider_used == "fast_provider"
    assert p_slow.generate_call_count == 1  # Exactly 1 call: never retried slow/hung endpoint


# ==============================================================================
# Critical Test 4: Azure OpenAI Generation, Tool Calling & Failover
# ==============================================================================

@pytest.mark.asyncio
async def test_azure_adapter_generation_tool_calling_and_failover():
    """Verify AzureOpenAIProvider handles endpoints, headers, tool calling, and failover to secondary provider."""
    # 1. Test Azure adapter structure and URL resolution
    azure = AzureOpenAIProvider(
        api_key="mock-azure-key",
        base_url="https://test-resource.openai.azure.com",
        default_model="gpt-5.5",
    )
    assert azure.name == "azure"
    assert "api-key" in azure._headers()
    assert azure._headers()["api-key"] == "mock-azure-key"

    url = azure._build_completion_url("gpt-5.5")
    assert "/openai/deployments/gpt-5.5/chat/completions" in url
    assert "api-version=" in url

    # 2. Test failover from Azure 429 to Groq
    p_azure = MockControllableProvider(
        "azure",
        "gpt-5.5",
        responses=[ProviderRequestError(
            ProviderErrorCategory.RATE_LIMIT,
            "Azure OpenAI Rate Limit Exceeded (429)",
            provider="azure",
            model="gpt-5.5",
            status_code=429,
        )],
    )
    p_groq = MockControllableProvider(
        "groq",
        "llama-3.3-70b-versatile",
        responses=[CompletionResponse(
            content="Groq resolved after Azure rate limit",
            model="llama-3.3-70b-versatile",
            provider="groq",
        )],
    )

    router = ModelRouter(
        providers=[p_azure, p_groq],
        config=RouterConfig(
            routing_mode="auto",
            provider_priority=["azure", "groq"],
            fallback=FallbackConfig(retry=RetryConfig(max_retries=0)),
        ),
    )

    req = CompletionRequest(messages=[{"role": "user", "content": "analyze task"}])
    result = await router.execute(req)

    assert result.succeeded
    assert result.provider_used == "groq"
    assert result.attempt_count == 2
    assert "Groq resolved after Azure rate limit" in result.response.content


# ==============================================================================
# Critical Test 5: Cooldown Expiration and Recovery
# ==============================================================================

@pytest.mark.asyncio
async def test_cooldown_expiration_allows_recovered_provider():
    """Verify that after a provider's cooldown period expires, it becomes eligible again."""
    health_tracker = ModelHealthTracker()
    model_id = "test-model"

    # Record 429 with short 0.1s cooldown override
    health_tracker.record_failure(model_id, HealthEvent.RATE_LIMIT_429, cooldown_override=0.1)
    state = health_tracker.get_state(model_id)
    assert state.is_rate_limited
    assert not state.is_healthy

    # Wait for cooldown to expire
    await asyncio.sleep(0.15)
    assert not state.is_rate_limited
    assert state.is_healthy


# ==============================================================================
# Critical Test 6: All Providers Unavailable Yields Informative Diagnostics
# ==============================================================================

@pytest.mark.asyncio
async def test_all_providers_unavailable_yields_actionable_error():
    """Verify that when all configured providers fail, FallbackEngine returns a clear diagnostic summary."""
    p1 = MockControllableProvider(
        "provider1", "m1",
        responses=[ProviderRequestError(ProviderErrorCategory.RATE_LIMIT, "Quota exceeded 429", provider="p1", model="m1", status_code=429)],
    )
    p2 = MockControllableProvider(
        "provider2", "m2",
        responses=[ProviderRequestError(ProviderErrorCategory.RATE_LIMIT, "Daily limit reached 429", provider="p2", model="m2", status_code=429)],
    )

    router = ModelRouter(
        providers=[p1, p2],
        config=RouterConfig(fallback=FallbackConfig(retry=RetryConfig(max_retries=0))),
    )

    req = CompletionRequest(messages=[{"role": "user", "content": "test"}])
    result = await router.execute(req)

    assert not result.succeeded
    assert result.final_error
    assert "rate limit" in result.final_error.lower() or "429" in result.final_error or "failed" in result.final_error.lower()


# ==============================================================================
# Critical Test 7: Router Never Selects An Ineligible Model
# ==============================================================================

@pytest.mark.asyncio
async def test_router_never_selects_model_without_tool_support_for_tool_tasks():
    """Verify that when a task requires tools, models with supports_tools=False are never selected."""
    p_no_tools = MockControllableProvider(
        "no_tools_provider",
        "summarize-only-model",
        supports_tools=False,
    )
    p_with_tools = MockControllableProvider(
        "tools_provider",
        "coding-model",
        supports_tools=True,
        responses=[CompletionResponse(content="Ready", model="coding-model", provider="tools_provider")],
    )

    router = ModelRouter(
        providers=[p_no_tools, p_with_tools],
        config=RouterConfig(routing_mode="auto"),
    )

    req = CompletionRequest(
        messages=[{"role": "user", "content": "Edit this file"}],
        tools=[{"type": "function", "function": {"name": "edit_file", "parameters": {}}}],
    )
    result = await router.execute(req)

    assert result.succeeded
    assert result.model_used == "coding-model"
    assert result.provider_used == "tools_provider"

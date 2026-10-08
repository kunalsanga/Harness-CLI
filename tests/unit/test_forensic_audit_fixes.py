import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import Task, TaskStatus, ToolResult, ToolResultStatus
from harness_core.cli.interactive import InteractiveShell
from harness_core.cli.conversation import ConversationRenderer
from harness_core.agent.workflows import run_explain_workflow

@pytest.mark.asyncio
async def test_explain_workflow_does_not_return_placeholder():
    """Test that the explain workflow returns context pieces and loop does not use 'Project analyzed.' placeholder."""
    # Mock context engine to just return some dummy project info
    ctx = MagicMock()
    ctx.execute = AsyncMock(return_value=ToolResult(ToolResultStatus.SUCCESS, "README.md\nmain.py"))
    
    plan = MagicMock()
    
    with patch("harness_core.agent.workflows._run_tool", new=ctx.execute):
        wf_result = await run_explain_workflow(ctx, plan)
        
    assert wf_result.success
    assert "context_pieces" in wf_result.data
    
    # Test apply_workflow_result
    agent_loop = AgentLoop(config=MagicMock(), provider=MagicMock(), tools=[])
    task = Task(goal="explain this project")
    agent_loop._apply_workflow_result(task, wf_result)
    
    # The placeholder should NOT be set
    assert not task.result

@pytest.mark.asyncio
async def test_402_failure_does_not_become_paused():
    """Test that a provider exception doesn't become PAUSED just because tools ran."""
    provider = AsyncMock()
    provider.generate.side_effect = Exception("402 Payment Required")
    config = MagicMock()
    config.max_iterations = 10
    agent_loop = AgentLoop(config=config, provider=provider, tools=[])
    agent_loop._project_info = {}
    
    # Simulate some work being done (tools ran but no mutations)
    task = Task(goal="implement a calculator")
    agent_loop._active_task = task
    # We fake that a tool was called and returned something
    from harness_core.agent.types import ToolCall
    tc = ToolCall("read_file", {"path": "README.md"}, "call_1")
    tc.result = ToolResult(ToolResultStatus.SUCCESS, "foo")
    task.tool_calls.append(tc)
    
    # _modified_files is empty because it's read only
    agent_loop._modified_files = []
    
    # Run the loop
    with patch.object(agent_loop, "_build_messages", return_value=[]):
        with patch.object(agent_loop.budget, "check_all", return_value=(True, "")):
            # This calls the inner execution loop part directly since run() calls the whole workflow
            # We can mock the run loop to test the exception handler
            task = await agent_loop.run("implement a calculator")
            
    assert task.status == TaskStatus.FAILED
    assert "402 Payment Required" in task.error

@pytest.mark.asyncio
async def test_interactive_shell_duplicate_prompt_prevention():
    """Test that prompt is rendered exactly once: _execute_task calls start,
    and on_task_started does NOT call start again."""
    shell = InteractiveShell()
    shell.console = MagicMock()
    shell.plain = False
    
    # Stage 1: Set up provider mock so the canonical runtime path is exercised
    class _FakeProv:
        name = "fake"
        async def health_check(self): return True
        async def close(self): pass
        async def generate(self, req):
            from harness_core.providers.base import CompletionResponse
            return CompletionResponse(content="done", model="fake")
        async def list_models(self): return []
    shell._provider = _FakeProv()
    shell._router = MagicMock()
    shell._tools = []
    
    # Setup mock event bus
    from harness_core.observability.events import EventBus, Event
    shell._event_bus = EventBus()
    shell._setup_event_handlers()
    
    # We mock ConversationRenderer entirely
    with patch("harness_core.cli.interactive.ConversationRenderer") as MockConv:
        mock_conv_instance = MockConv.return_value
        
        # Stage 1: Mock EngineeringRuntime.execute_interactive instead of AgentLoop.run
        with patch("harness_core.runtime.runtime.EngineeringRuntime.execute_interactive", new_callable=AsyncMock) as mock_runtime:
            from harness_core.runtime.runtime import RuntimeOutcome
            from harness_core.runtime.state import RuntimeStatus, ProjectState
            state = ProjectState(project_id="test", workspace=".")
            state.status = RuntimeStatus.SUCCESS
            mock_runtime.return_value = RuntimeOutcome(state=state, status=RuntimeStatus.SUCCESS, duration_ms=100.0)
            
            await shell._execute_task("test goal")
            
            # _execute_task MUST call start (single authoritative call)
            mock_conv_instance.start.assert_called_once_with("test goal")
            
            # Now simulate the task.started event: it must NOT call start again
            shell._conv = mock_conv_instance
            mock_conv_instance.start.reset_mock()
            event = Event(type="task.started", source="agent_loop", data={"goal": "test goal"})
            for handler in shell._event_bus._handlers.get("task.started", []):
                await handler(event)
                
            # on_task_started must NOT call start — prompt was already rendered
        mock_conv_instance.start.assert_not_called()

def test_completion_formatter_distinguishes_statuses():
    """Test that CompletionFormatter correctly handles different TaskStatus."""
    from harness_core.cli.completion import CompletionFormatter
    fmt = CompletionFormatter(plain=True)
    
    # Success
    report = fmt.success(headline="Goal", files_modified=[], files_created=[], tests_line="", verification_status="", git_commit="", git_push="", duration=1, tool_calls=0, next_actions=[], agent_response="Explanation here")
    assert "Explanation here" in report
    assert "Task Completed" not in report # It uses the headline
    
    # Failure
    report_fail = fmt.failure(headline="Task Failed", what_happened="Provider error", files_modified=[], next_actions=[], agent_response="")
    assert "Task Failed" in report_fail
    assert "Provider error" in report_fail

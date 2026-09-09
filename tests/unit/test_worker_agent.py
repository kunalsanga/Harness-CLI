import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from harness_core.agents.domain import AgentContract, AgentRole, AgentStatus, MessageType
from harness_core.agents.worker import WorkerAgent
from harness_core.observability.events import EventBus

@pytest.fixture
def event_bus():
    return EventBus()

@pytest.fixture
def contract():
    return AgentContract(
        agent_id="test_coder_1",
        role=AgentRole.CODER,
        task_id="task_123",
        objective="Implement feature X",
        timeout_seconds=5.0
    )

@pytest.mark.asyncio
async def test_worker_agent_lifecycle(event_bus, contract):
    worker = WorkerAgent(
        contract=contract,
        provider=MagicMock(),
        tools=[],
        event_bus=event_bus,
        workspace_path="."
    )
    
    assert worker.status == AgentStatus.CREATED
    
    # Track emitted events
    emitted_events = []
    async def capture_event(event):
        emitted_events.append(event)
    
    event_bus.on("*", capture_event)
    
    # Mock AgentLoop
    with patch("harness_core.agent.loop.AgentLoop") as MockLoop:
        mock_loop_instance = MockLoop.return_value
        
        # Fake successful result
        mock_result = MagicMock()
        mock_result.status.value = "completed"
        mock_result.result = "Feature X implemented"
        mock_result.tool_calls = []
        mock_result.iterations = 1
        mock_result.model_used = "mock-model"
        mock_result.total_tokens = 100
        mock_result.error = None
        
        mock_loop_instance.run = AsyncMock(return_value=mock_result)
        
        result = await worker.run()
        
        assert result.status == AgentStatus.COMPLETED
        assert result.summary == "Feature X implemented"
        assert worker.status == AgentStatus.COMPLETED
        
        # Allow create_task callbacks to run
        await asyncio.sleep(0.01)

        # Verify events
        event_types = [e.type for e in emitted_events]
        assert "agent.status_changed" in event_types
        assert "agent.started" in event_types
        assert "agent.completed" in event_types

@pytest.mark.asyncio
async def test_worker_agent_timeout(event_bus, contract):
    contract.timeout_seconds = 0.1
    worker = WorkerAgent(
        contract=contract,
        provider=MagicMock(),
        tools=[],
        event_bus=event_bus,
        workspace_path="."
    )
    
    with patch("harness_core.agent.loop.AgentLoop") as MockLoop:
        mock_loop_instance = MockLoop.return_value
        
        async def slow_run(*args, **kwargs):
            await asyncio.sleep(0.5)
            return MagicMock()
            
        mock_loop_instance.run = slow_run
        
        result = await worker.run()
        
        assert result.status == AgentStatus.FAILED
        assert "timeout" in result.summary.lower()
        assert worker.status == AgentStatus.FAILED

@pytest.mark.asyncio
async def test_worker_agent_messaging(event_bus, contract):
    from harness_core.agents.message_bus import AgentMessageBus
    from harness_core.agents.domain import TaskGraph
    from harness_core.agents.registry import AgentRegistry
    
    task_graph = TaskGraph()
    # Add dummy tasks for validation
    from harness_core.agents.domain import SubTask, AgentRole
    task_graph.add_task(SubTask(task_id="task_123", role=AgentRole.CODER))
    task_graph.add_task(SubTask(task_id="task_456", role=AgentRole.REVIEWER))
    
    msg_bus = AgentMessageBus(event_bus, task_graph, AgentRegistry())
    
    worker = WorkerAgent(
        contract=contract,
        provider=MagicMock(),
        tools=[],
        event_bus=event_bus,
        workspace_path=".",
        message_bus=msg_bus
    )
    
    emitted_events = []
    async def capture_event(event):
        emitted_events.append(event)
        
    event_bus.on("agent_message_sent", capture_event)
    
    await worker.send_message(
        receiver_task_id="task_456",
        msg_type=MessageType.REVIEW_REQUEST,
        payload={"content": "Please review this."}
    )
    
    assert len(emitted_events) == 1
    msg_event = emitted_events[0]
    assert msg_event.type == "agent_message_sent"
    assert msg_event.data["sender_agent_id"] == "test_coder_1"
    assert msg_event.data["recipient_task_id"] == "task_456"
    assert msg_event.data["message_type"] == MessageType.REVIEW_REQUEST.value

@pytest.mark.asyncio
async def test_readonly_escalation(event_bus, contract):
    """Adversarial test: prove that denied tools like write_file and run_command
    are physically rejected by AgentLoop._execute_tool_checked, ensuring
    shell escalation is blocked at execution time.
    """
    from harness_core.agent.loop import AgentLoop
    from harness_core.agent.types import AgentConfig as LoopConfig
    from harness_core.tools.shell import RunCommandTool
    from harness_core.tools.filesystem import WriteFileTool, ReadFileTool
    from harness_core.agent.types import AgentConfig as LoopConfig, ToolCall
    from harness_core.agents.registry import AgentRegistry
    
    # 1. Instantiate the REVIEWER profile
    registry = AgentRegistry()
    reviewer_profile = registry.find_by_role(AgentRole.REVIEWER)[0]
    
    contract.role = AgentRole.REVIEWER
    contract.allowed_tools = reviewer_profile.allowed_tools
    contract.denied_tools = reviewer_profile.denied_tools
    
    # Give the worker all tools to simulate full environment
    all_tools = [ReadFileTool(), WriteFileTool(), RunCommandTool()]
    
    worker = WorkerAgent(
        contract=contract,
        provider=MagicMock(),
        tools=all_tools,
        event_bus=event_bus,
        workspace_path="."
    )
    
    # The worker filters tools internally
    filtered_tools = worker._filter_tools()
    
    # Simulate what AgentLoop will see
    loop = AgentLoop(
        provider=MagicMock(),
        tools=filtered_tools,
        workspace_root=".",
        config=LoopConfig()
    )
    
    # Attack 1: Direct write_file
    call_write = ToolCall(id="1", tool_name="write_file", arguments={"path": "x.py", "content": "malicious"})
    res_write = await loop._execute_tool_checked(call_write)
    assert res_write.execution_failed
    err_write = (res_write.error or res_write.output).lower()
    assert "unknown tool" in err_write or "not found" in err_write
    
    # Attack 2: run_command escalation
    call_run = ToolCall(id="2", tool_name="run_command", arguments={"command": "rm -rf /"})
    res_run = await loop._execute_tool_checked(call_run)
    assert res_run.execution_failed
    err_run = (res_run.error or res_run.output).lower()
    assert "unknown tool" in err_run or "not found" in err_run
    
    # Legitimate tool: read_file
    call_read = ToolCall(id="3", tool_name="read_file", arguments={})
    res_read = await loop._execute_tool_checked(call_read)
    # Since it's allowed, it should fail with missing arguments or execution error, NOT unknown tool
    err_read = (res_read.error or res_read.output).lower()
    assert "unknown tool" not in err_read

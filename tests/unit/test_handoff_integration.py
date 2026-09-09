import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from harness_core.observability.events import EventBus
from harness_core.agents.registry import AgentRegistry
from harness_core.agents.scheduler import Scheduler
from harness_core.agents.domain import TaskGraph, SubTask, AgentRole, AgentMessage, MessageType
from harness_core.providers.base import CompletionResponse

@pytest.mark.asyncio
async def test_end_to_end_handoff_flow():
    """
    Test Phase 6 E2E Integration: 
    ARCHITECT -> BACKEND -> TESTER -> VERIFIER
    """
    event_bus = EventBus()
    registry = AgentRegistry()
    from harness_core.agents.registry import AgentProfile
    registry.register(AgentProfile(name="arch", role=AgentRole.ARCHITECT, system_instructions=""))
    registry.register(AgentProfile(name="backend", role=AgentRole.BACKEND, system_instructions=""))
    registry.register(AgentProfile(name="tester", role=AgentRole.TESTER, system_instructions=""))
    registry.register(AgentProfile(name="verifier", role=AgentRole.VERIFIER, system_instructions=""))
    mock_provider = MagicMock()
    captured_prompts = {}
    async def mock_gen(*args, **kwargs):
        await asyncio.sleep(0.05)
        return CompletionResponse(content="Done", model="test")
    mock_provider.generate = AsyncMock(side_effect=mock_gen)

    scheduler = Scheduler(event_bus, registry, mock_provider, [], "/workspace", max_concurrency=4)

    # 1. Setup TaskGraph with our 4 tasks
    graph = TaskGraph()
    arch_task = SubTask(task_id="architect_1", role=AgentRole.ARCHITECT)
    backend_task = SubTask(task_id="backend_1", role=AgentRole.BACKEND, dependencies=["architect_1"])
    tester_task = SubTask(task_id="tester_1", role=AgentRole.TESTER, dependencies=["backend_1"])
    verifier_task = SubTask(task_id="verifier_1", role=AgentRole.VERIFIER, dependencies=["tester_1"])

    graph.add_task(arch_task)
    graph.add_task(backend_task)
    graph.add_task(tester_task)
    graph.add_task(verifier_task)

    events = []
    async def capture(event):
        events.append(event)
        
    event_bus.on("agent_handoff_created", capture)
    event_bus.on("agent_message_sent", capture)

    # In a real run, the worker's agent loop tool call would send the message.
    # Since we are mocking the provider, we need to inject a hook that sends the message
    # when the worker starts.
    async def on_task_started(event):
        try:
            task_id = event.data["task_id"]
            events.append(f"STARTED: {task_id}")
            worker = None
            for w in scheduler.workers.values():
                if w.contract.task_id == task_id:
                    worker = w
                    break
            
            if not worker:
                events.append(f"Worker not found for {task_id}")
                return

            if task_id == "architect_1":
                # Architect -> Backend
                await worker.send_message(
                    receiver_task_id="backend_1",
                    msg_type=MessageType.HANDOFF,
                    payload={"artifacts": ["docs/api-contract.md"], "contract_id": "ARCH-CONTRACT-7391"}
                )
            elif task_id == "backend_1":
                # Check if backend received the handoff
                inbox = scheduler.message_bus.get_messages_for("backend_1")
                assert len(inbox) == 1
                assert inbox[0].message_type == MessageType.HANDOFF
                assert inbox[0].payload["contract_id"] == "ARCH-CONTRACT-7391"
                
                worker.inbox = list(inbox)
                captured_prompts["backend_1"] = await worker._build_prompt_async()

                # Backend -> Tester
                await worker.send_message(
                    receiver_task_id="tester_1",
                    msg_type=MessageType.RESULT,
                    payload={"status": "COMPLETED", "summary": "Implemented API"}
                )
            elif task_id == "tester_1":
                # Tester -> Verifier
                await worker.send_message(
                    receiver_task_id="verifier_1",
                    msg_type=MessageType.VERIFICATION_REQUEST,
                    payload={"tests": ["test_api.py"], "summary": "Tests pass"}
                )
            elif task_id == "verifier_1":
                # Check if verifier got the request
                inbox = scheduler.message_bus.get_messages_for("verifier_1")
                assert len(inbox) == 1
                assert inbox[0].message_type == MessageType.VERIFICATION_REQUEST
        except Exception as e:
            events.append(e)

    event_bus.on("agent.started", on_task_started)

    # 2. Execute
    await scheduler.execute(graph)

    # 3. Assertions
    actual_events = [e for e in events if not isinstance(e, str)]
    # 3 Messages sent: HANDOFF, RESULT, VERIFICATION_REQUEST
    assert len(actual_events) == 3, f"Events: {events}"
    
    handoff_event = [e for e in actual_events if e.type == "agent_handoff_created"][0]
    assert handoff_event.data["sender_task_id"] == "architect_1"
    assert handoff_event.data["recipient_task_id"] == "backend_1"
    
    result_event = [e for e in actual_events if e.type == "agent_message_sent" and e.data["message_type"] == MessageType.RESULT.value][0]
    assert result_event.data["sender_task_id"] == "backend_1"
    assert result_event.data["recipient_task_id"] == "tester_1"

    verify_event = [e for e in actual_events if e.type == "agent_message_sent" and e.data["message_type"] == MessageType.VERIFICATION_REQUEST.value][0]
    assert verify_event.data["sender_task_id"] == "tester_1"
    assert verify_event.data["recipient_task_id"] == "verifier_1"
    
    # 4. Context Injection Verification
    backend_prompt = captured_prompts.get("backend_1", "")
    assert "ARCH-CONTRACT-7391" in backend_prompt, "Unique handoff payload must be injected into the recipient context."


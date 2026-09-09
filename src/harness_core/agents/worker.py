"""
Worker agent for M5 — a true autonomous multi-agent worker.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from harness_core.agents.message_bus import AgentMessageBus
from harness_core.observability.events import Event, EventBus

from .domain import AgentContract, AgentMessage, AgentResult, AgentStatus, MessageType


class WorkerAgent:
    """A standalone agent worker that executes an AgentContract."""

    def __init__(
        self,
        contract: AgentContract,
        provider: Any,
        tools: list,
        event_bus: EventBus,
        workspace_path: str,
        router: Any = None,
        message_bus: AgentMessageBus | None = None,
        memory: Any = None,
        project_id: str = "",
    ) -> None:
        self.contract = contract
        self.provider = provider
        self.tools = tools
        self.event_bus = event_bus
        self.message_bus = message_bus
        self.workspace_path = workspace_path
        self.router = router
        self.status = AgentStatus.CREATED
        self.inbox: list[AgentMessage] = []
        self._result: AgentResult | None = None
        # Phase 8G: RAG execution context
        self._memory = memory
        self._project_id = project_id

    async def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a structured event to the EventBus."""
        event = Event(
            type=event_type,
            source=self.contract.agent_id,
            data=data
        )
        await self.event_bus.emit(event)

    async def send_message(self, receiver_task_id: str, msg_type: MessageType, payload: dict[str, Any]) -> bool:
        """Send a message to another task via AgentMessageBus."""
        if not self.message_bus:
            return False

        msg = AgentMessage(
            sender_agent_id=self.contract.agent_id,
            sender_task_id=self.contract.task_id,
            recipient_task_id=receiver_task_id,
            message_type=msg_type,
            payload=payload,
        )
        return await self.message_bus.send(msg)

    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message into the inbox."""
        self.inbox.append(message)

    def set_status(self, new_status: AgentStatus) -> None:
        """Update status and emit event without blocking."""
        if self.status != new_status:
            old_status = self.status
            self.status = new_status
            asyncio.create_task(self.emit("agent.status_changed", {
                "agent_id": self.contract.agent_id,
                "role": self.contract.role.value,
                "old_status": old_status.value,
                "new_status": new_status.value
            }))

    async def run(self) -> AgentResult:
        """Start the agent execution lifecycle."""
        self.set_status(AgentStatus.RUNNING)
        await self.emit("agent.started", {"task_id": self.contract.task_id})

        start_time = time.time()

        # Load pending messages from message bus if available
        if self.message_bus:
            pending = self.message_bus.get_messages_for(self.contract.task_id)
            for p in pending:
                if p not in self.inbox:
                    self.inbox.append(p)

        # Phase 8G/9: Build prompt with optional RAG context (async for memory
        # retrieval) plus the runtime-assembled project context block.
        prompt = await self._build_prompt_async()
        filtered_tools = self._filter_tools()

        try:
            from harness_core.agent.loop import AgentLoop
            from harness_core.agent.types import AgentConfig as LoopAgentConfig
            from harness_core.agent.types import AgentRole as LoopRole

            # We reuse the robust existing AgentLoop, but scoped to this agent
            loop_config = LoopAgentConfig(
                role=LoopRole.BUILD,
                max_iterations=30,
                max_tool_calls=100,
                routing_mode=self.contract.model_policy,
            )

            agent_loop = AgentLoop(
                provider=self.provider,
                tools=filtered_tools,
                workspace_root=Path(self.workspace_path),
                config=loop_config,
                event_bus=self.event_bus,
                router=self.router,
                agent_id=self.contract.agent_id,
                task_id=self.contract.task_id,
            )

            # Execution with contract-defined timeout
            task_result = await asyncio.wait_for(
                agent_loop.run(prompt),
                timeout=self.contract.timeout_seconds,
            )

            status = AgentStatus.COMPLETED if task_result.status.value == "completed" else AgentStatus.FAILED
            model_id = task_result.models_used[0] if getattr(task_result, "models_used", None) else "unknown"
            self._result = AgentResult(
                agent_id=self.contract.agent_id,
                role=self.contract.role,
                status=status,
                summary=task_result.result or f"Task {self.contract.task_id} completed",
                tool_calls=len(task_result.tool_calls),
                iterations=task_result.iterations,
                model_id=model_id,
                tokens_used=getattr(task_result, "total_tokens", 0),
                duration_ms=(time.time() - start_time) * 1000
            )

            if task_result.error:
                self._result.errors.append(task_result.error)

            self.set_status(status)
            await self.emit("agent.completed" if status == AgentStatus.COMPLETED else "agent.failed", {
                "agent_id": self.contract.agent_id,
                "summary": self._result.summary
            })

        except TimeoutError:
            self.set_status(AgentStatus.FAILED)
            self._result = AgentResult(
                agent_id=self.contract.agent_id,
                role=self.contract.role,
                status=AgentStatus.FAILED,
                summary="Agent timeout",
                errors=["Timeout: agent exceeded time limit"],
                duration_ms=(time.time() - start_time) * 1000
            )
            await self.emit("agent.failed", {"error": "Timeout"})
        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            self._result = AgentResult(
                agent_id=self.contract.agent_id,
                role=self.contract.role,
                status=AgentStatus.FAILED,
                summary="Agent error",
                errors=[f"Error: {str(e)}"],
                duration_ms=(time.time() - start_time) * 1000
            )
            await self.emit("agent.failed", {"error": str(e)})

        return self._result

    async def _build_prompt_async(self) -> str:
        parts = [
            f"You are a specialized agent: {self.contract.role.value}",
            f"Objective: {self.contract.objective}",
        ]

        if self.contract.system_instructions:
            parts.append(f"System Instructions:\n{self.contract.system_instructions}")

        parts.append(f"Workspace Scope: {self.contract.workspace_scope.value.upper()}")

        if self.contract.inputs:
            parts.append(f"Inputs available: {', '.join(self.contract.inputs)}")

        if self.contract.success_criteria:
            parts.append("Success Criteria:")
            for sc in self.contract.success_criteria:
                parts.append(f"- {sc}")

        if self.contract.output_requirements:
            parts.append("Output Requirements:")
            for req in self.contract.output_requirements:
                parts.append(f"- {req}")

        # Phase 8G: Inject historical context from prior executions (RAG)
        if self._memory and getattr(self._memory, "enabled", False):
            try:
                ctx = await self._memory.get_context_for_worker(
                    objective=self.contract.objective,
                    role=self.contract.role.value,
                    project_id=self._project_id,
                )
                if ctx:
                    parts.append(f"\n## Historical Context\n{ctx}")
            except Exception:
                # Memory retrieval failures must not break agent execution
                pass

        # Phase 9: Runtime-provided project context (requirements, traces,
        # artifact references, handoffs). Assembled by EngineeringRuntime and
        # attached to the task before the scheduler executes it.
        runtime_ctx = getattr(self.contract, "runtime_context", "") or getattr(self, "runtime_context", "")
        if runtime_ctx:
            parts.append(f"\n{runtime_ctx}")

        if self.inbox:
            parts.append("\nRelevant Handoffs/Context:")
            for msg in self.inbox:
                parts.append(f"--- Message from task '{msg.sender_task_id}' (Type: {msg.message_type.value}) ---")
                import json
                parts.append(json.dumps(msg.payload, indent=2))
                parts.append("------------------------------------------")

        return "\n".join(parts)

    def _filter_tools(self) -> list:
        filtered = self.tools

        # Apply allowlist if specified
        if self.contract.allowed_tools:
            filtered = [t for t in filtered if t.schema.name in self.contract.allowed_tools]

        # Apply denylist if specified
        if self.contract.denied_tools:
            filtered = [t for t in filtered if t.schema.name not in self.contract.denied_tools]

        return filtered

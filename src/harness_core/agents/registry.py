"""
Agent registry for M5 — register, lookup, and configure specialized agents.

Each agent declares its role, capabilities, allowed tools, and resource limits.
Models are selected separately by ModelRouter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .domain import AgentRole


@dataclass
class AgentConfig:
    """Configuration for a specialized agent."""

    name: str = ""
    role: AgentRole = AgentRole.CODER
    display_name: str = ""
    description: str = ""

    # Capabilities
    capabilities: list[str] = field(default_factory=list)
    preferred_task_types: list[str] = field(default_factory=list)

    # Tool access
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)

    # System instructions (role-specific prompt)
    system_instructions: str = ""

    # Resource limits
    max_iterations: int = 30
    max_tool_calls: int = 100
    max_tokens: int = 32000

    # Model preferences (used by orchestrator for routing)
    prefer_fast_model: bool = False
    prefer_strong_model: bool = False
    prefer_cheap_model: bool = False

    # Metadata
    enabled: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "display_name": self.display_name,
            "description": self.description,
            "capabilities": self.capabilities,
            "preferred_task_types": self.preferred_task_types,
            "allowed_tools": self.allowed_tools,
            "system_instructions": self.system_instructions[:100] + "..." if len(self.system_instructions) > 100 else self.system_instructions,
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "enabled": self.enabled,
        }


class AgentRegistry:
    """Registry of available specialized agents.

    Thread-safe. Agents are registered at startup.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentConfig] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the default agent configurations."""

        self.register(AgentConfig(
            name="planner",
            role=AgentRole.PLANNER,
            display_name="Planner",
            description="Decomposes complex tasks into structured subtask graphs",
            capabilities=["task_decomposition", "dependency_analysis", "prioritization"],
            preferred_task_types=["implementation", "refactoring", "documentation"],
            allowed_tools=["read_file", "list_files", "grep", "glob"],
            system_instructions=(
                "You are a task planner. Analyze the user's request and decompose it "
                "into structured subtasks with clear dependencies. Each subtask should "
                "have a specific role (researcher, coder, tester, reviewer) and clear "
                "acceptance criteria. Think step by step about dependencies."
            ),
            prefer_strong_model=True,
        ))

        self.register(AgentConfig(
            name="researcher",
            role=AgentRole.RESEARCHER,
            display_name="Researcher",
            description="Investigates codebase, finds relevant files, and gathers context",
            capabilities=["codebase_search", "file_analysis", "dependency_tracking"],
            preferred_task_types=["research", "repository_analysis"],
            allowed_tools=["read_file", "list_files", "grep", "glob", "run_command"],
            system_instructions=(
                "You are a code researcher. Investigate the codebase to find relevant "
                "files, understand architecture, and gather context. Be thorough but "
                "efficient. Report your findings with file paths and key observations."
            ),
            prefer_fast_model=True,
        ))

        self.register(AgentConfig(
            name="analyzer",
            role=AgentRole.ANALYZER,
            display_name="Analyzer",
            description="Analyzes code quality, identifies issues, and proposes improvements",
            capabilities=["code_analysis", "issue_detection", "architecture_review"],
            preferred_task_types=["analysis", "review", "debugging"],
            allowed_tools=["read_file", "list_files", "grep", "glob", "run_command"],
            system_instructions=(
                "You are a code analyzer. Examine code for bugs, security issues, "
                "performance problems, and architectural concerns. Provide specific, "
                "actionable findings with file locations and severity."
            ),
            prefer_strong_model=True,
        ))

        self.register(AgentConfig(
            name="coder",
            role=AgentRole.CODER,
            display_name="Coder",
            description="Implements code changes, fixes bugs, and writes new features",
            capabilities=["code_writing", "bug_fixing", "feature_implementation", "refactoring"],
            preferred_task_types=["implementation", "bug_fix", "refactoring"],
            allowed_tools=["read_file", "write_file", "edit_file", "list_files", "grep", "glob", "run_command"],
            system_instructions=(
                "You are a software engineer. Implement code changes as specified. "
                "Write clean, well-structured code that follows project conventions. "
                "Make minimal changes to achieve the goal. Verify your work."
            ),
            prefer_strong_model=True,
        ))

        self.register(AgentConfig(
            name="tester",
            role=AgentRole.TESTER,
            display_name="Tester",
            description="Runs tests, diagnoses failures, and validates correctness",
            capabilities=["test_execution", "failure_diagnosis", "coverage_analysis"],
            preferred_task_types=["testing", "verification"],
            allowed_tools=["read_file", "run_command", "grep", "list_files"],
            system_instructions=(
                "You are a test engineer. Run tests, analyze failures, and validate "
                "that changes work correctly. Report specific failures with file paths "
                "and error messages. Suggest fixes for failing tests."
            ),
            prefer_cheap_model=True,
        ))

        self.register(AgentConfig(
            name="reviewer",
            role=AgentRole.REVIEWER,
            display_name="Reviewer",
            description="Reviews code changes for quality, security, and correctness",
            capabilities=["code_review", "security_review", "architecture_review"],
            preferred_task_types=["review", "security"],
            allowed_tools=["read_file", "list_files", "grep", "glob", "run_command"],
            system_instructions=(
                "You are a code reviewer. Inspect changes for correctness, security, "
                "performance, and adherence to project conventions. Provide structured "
                "findings with severity levels and specific recommendations. "
                "End with APPROVED or CHANGES_REQUESTED."
            ),
            prefer_strong_model=True,
        ))

        self.register(AgentConfig(
            name="debugger",
            role=AgentRole.DEBUGGER,
            display_name="Debugger",
            description="Diagnoses failures and coordinates repair cycles",
            capabilities=["failure_analysis", "root_cause_detection", "repair_coordination"],
            preferred_task_types=["debugging", "failure_recovery"],
            allowed_tools=["read_file", "write_file", "edit_file", "list_files", "grep", "glob", "run_command"],
            system_instructions=(
                "You are a debugger. Analyze test failures, error messages, and unexpected "
                "behavior. Identify root causes and implement fixes. Track what you've "
                "tried and what worked. Be methodical."
            ),
            prefer_strong_model=True,
        ))

    def register(self, config: AgentConfig) -> None:
        """Register an agent configuration."""
        self._agents[config.name] = config

    def unregister(self, name: str) -> bool:
        """Remove an agent configuration."""
        if name in self._agents:
            del self._agents[name]
            return True
        return False

    def get(self, name: str) -> AgentConfig | None:
        """Get an agent configuration by name."""
        return self._agents.get(name)

    def list_all(self) -> list[AgentConfig]:
        """List all registered agents."""
        return list(self._agents.values())

    def list_enabled(self) -> list[AgentConfig]:
        """List all enabled agents."""
        return [a for a in self._agents.values() if a.enabled]

    def find_by_role(self, role: AgentRole) -> list[AgentConfig]:
        """Find agents by role."""
        return [a for a in self._agents.values() if a.role == role and a.enabled]

    def find_for_task(self, task_type: str) -> list[AgentConfig]:
        """Find agents suitable for a task type."""
        suitable = []
        for agent in self._agents.values():
            if not agent.enabled:
                continue
            if task_type in agent.preferred_task_types:
                suitable.append(agent)
            elif not agent.preferred_task_types:
                suitable.append(agent)  # No preference = available for all
        return suitable

    def get_default_for_role(self, role: AgentRole) -> AgentConfig | None:
        """Get the default (first enabled) agent for a role."""
        agents = self.find_by_role(role)
        return agents[0] if agents else None

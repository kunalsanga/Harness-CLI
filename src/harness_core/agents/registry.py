"""
Agent registry and profile definitions for M5 capabilities.

Defines the AgentProfile (previously AgentConfig) system which enforces 
specialist capability profiles, allowed tools, model policies, and constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .domain import AgentRole, WorkspaceScope

logger = logging.getLogger(__name__)


@dataclass
class AgentProfile:
    """Specialist agent capability profile."""

    name: str
    role: AgentRole
    display_name: str = ""
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    preferred_task_types: list[str] = field(default_factory=list)
    
    # Tool constraints
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    
    # Workspace & Model
    workspace_scope: WorkspaceScope = WorkspaceScope.PROJECT
    model_policy: str = "auto"
    
    # Execution parameters
    max_concurrency: int = 1
    timeout_seconds: float = 300.0
    budget_cost: float = 1.0
    
    # Instructions
    system_instructions: str = ""
    output_requirements: list[str] = field(default_factory=list)
    
    enabled: bool = True

    # Legacy attributes mapped to model_policy for single-agent backwards compatibility
    prefer_strong_model: bool = False
    prefer_fast_model: bool = False
    prefer_cheap_model: bool = False

    def __post_init__(self):
        # Translate legacy preference to model policy if explicitly set
        if self.prefer_strong_model and self.model_policy == "auto":
            self.model_policy = "reasoning_high"
        elif self.prefer_fast_model and self.model_policy == "auto":
            self.model_policy = "fast"
        elif self.prefer_cheap_model and self.model_policy == "auto":
            self.model_policy = "fast"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "display_name": self.display_name,
            "description": self.description,
            "capabilities": self.capabilities,
            "preferred_task_types": self.preferred_task_types,
            "allowed_tools": self.allowed_tools,
            "denied_tools": self.denied_tools,
            "workspace_scope": self.workspace_scope.value,
            "model_policy": self.model_policy,
            "enabled": self.enabled,
        }

# Maintain legacy alias to ensure tests and external plugins do not break
AgentConfig = AgentProfile


class AgentRegistry:
    """Registry of available specialized agents.

    Thread-safe. Agents are registered at startup.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentProfile] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the default agent configurations."""

        # 1. PLANNER
        self.register(AgentProfile(
            name="planner",
            role=AgentRole.PLANNER,
            display_name="Planner",
            description="Decomposes complex tasks into structured subtask graphs",
            capabilities=["task_decomposition", "dependency_analysis", "prioritization"],
            preferred_task_types=["implementation", "refactoring", "documentation"],
            allowed_tools=["read_file", "list_files", "grep", "glob"],
            workspace_scope=WorkspaceScope.READ_ONLY,
            model_policy="reasoning_high",
            system_instructions=(
                "You are a task planner. Analyze the user's request and decompose it "
                "into structured subtasks with clear dependencies. Each subtask should "
                "have a specific role and clear acceptance criteria."
            )
        ))

        # 2. ARCHITECT
        self.register(AgentProfile(
            name="architect",
            role=AgentRole.ARCHITECT,
            display_name="Architect",
            description="Designs system architecture, components, and contracts",
            capabilities=["system_design", "api_design", "data_modeling", "technology_selection"],
            preferred_task_types=["architecture", "design", "planning"],
            allowed_tools=["read_file", "write_file", "list_files", "grep", "glob"],
            workspace_scope=WorkspaceScope.PROJECT,
            model_policy="reasoning_high",
            system_instructions=(
                "You are a senior software architect. Design robust, scalable architectures, "
                "API contracts, and database schemas. Create architecture design records (ADRs)."
            )
        ))

        # 3. RESEARCHER
        self.register(AgentProfile(
            name="researcher",
            role=AgentRole.RESEARCHER,
            display_name="Researcher",
            description="Investigates codebase, finds relevant files, and gathers context",
            capabilities=["codebase_search", "file_analysis", "dependency_tracking"],
            preferred_task_types=["research", "repository_analysis"],
            allowed_tools=["read_file", "list_files", "grep", "glob", "run_command"],
            workspace_scope=WorkspaceScope.READ_ONLY,
            model_policy="fast",
            system_instructions=(
                "You are a code researcher. Investigate the codebase to find relevant "
                "files, understand architecture, and gather context. Report file paths."
            )
        ))

        # 4. UI_DESIGNER
        self.register(AgentProfile(
            name="ui_designer",
            role=AgentRole.UI_DESIGNER,
            display_name="UI Designer",
            description="Designs beautiful, accessible user interfaces and design systems",
            capabilities=["ux_design", "ui_design", "accessibility", "design_systems"],
            preferred_task_types=["design", "frontend"],
            allowed_tools=["read_file", "write_file", "edit_file", "run_command"],
            workspace_scope=WorkspaceScope.FRONTEND,
            model_policy="coding",
            system_instructions=(
                "You are a UI/UX designer. Design components, layout, and visual styles "
                "ensuring consistency and accessibility. Focus on HTML/CSS structure."
            )
        ))

        # 5. FRONTEND
        self.register(AgentProfile(
            name="frontend",
            role=AgentRole.FRONTEND,
            display_name="Frontend Engineer",
            description="Builds client-side user interfaces and state management",
            capabilities=["frontend", "component_design", "styling", "accessibility"],
            preferred_task_types=["frontend", "implementation"],
            allowed_tools=["read_file", "write_file", "edit_file", "list_files", "grep", "glob", "run_command"],
            workspace_scope=WorkspaceScope.FRONTEND,
            model_policy="coding",
            system_instructions=(
                "You are a frontend engineer. Implement responsive user interfaces, "
                "components, and client-side logic. Follow provided design contracts."
            )
        ))

        # 6. BACKEND
        self.register(AgentProfile(
            name="backend",
            role=AgentRole.BACKEND,
            display_name="Backend Engineer",
            description="Builds server APIs, business logic, and backend services",
            capabilities=["api", "business_logic", "server", "testing"],
            preferred_task_types=["backend", "implementation"],
            allowed_tools=["read_file", "write_file", "edit_file", "list_files", "grep", "glob", "run_command"],
            workspace_scope=WorkspaceScope.BACKEND,
            model_policy="reasoning_high",
            system_instructions=(
                "You are a backend engineer. Implement APIs, services, and business logic. "
                "Ensure efficiency, correctness, and robust error handling."
            )
        ))

        # 7. DATABASE
        self.register(AgentProfile(
            name="database",
            role=AgentRole.DATABASE,
            display_name="Database Engineer",
            description="Manages schemas, migrations, and database performance",
            capabilities=["schema_design", "sql", "migrations", "performance"],
            preferred_task_types=["database", "migration"],
            allowed_tools=["read_file", "write_file", "edit_file", "list_files", "run_command"],
            workspace_scope=WorkspaceScope.DATABASE,
            model_policy="coding",
            system_instructions=(
                "You are a database engineer. Write and review SQL schemas, migrations, "
                "and optimized queries."
            )
        ))

        # 8. INTEGRATION
        self.register(AgentProfile(
            name="integration",
            role=AgentRole.INTEGRATION,
            display_name="Integration Engineer",
            description="Connects frontend, backend, and external services",
            capabilities=["api_integration", "service_stitching", "orchestration"],
            preferred_task_types=["integration", "fullstack"],
            allowed_tools=["read_file", "write_file", "edit_file", "list_files", "grep", "glob", "run_command"],
            workspace_scope=WorkspaceScope.PROJECT,
            model_policy="reasoning_high",
            system_instructions=(
                "You are an integration engineer. Wire together APIs, backend services, "
                "and frontend clients ensuring seamless end-to-end functionality."
            )
        ))

        # 9. TESTER
        self.register(AgentProfile(
            name="tester",
            role=AgentRole.TESTER,
            display_name="Tester",
            description="Writes tests, runs suites, and validates correctness",
            capabilities=["test_execution", "failure_diagnosis", "test_authoring"],
            preferred_task_types=["testing", "verification"],
            allowed_tools=["read_file", "write_file", "run_command", "grep", "list_files"],
            workspace_scope=WorkspaceScope.TESTS,
            model_policy="fast",
            system_instructions=(
                "You are a test engineer. Run tests, write new unit and integration tests, "
                "and validate that changes work correctly. Report specific failures."
            )
        ))

        # 10. DEBUGGER
        self.register(AgentProfile(
            name="debugger",
            role=AgentRole.DEBUGGER,
            display_name="Debugger",
            description="Diagnoses complex failures and coordinates repair cycles",
            capabilities=["failure_analysis", "root_cause_detection", "repair_coordination"],
            preferred_task_types=["debugging", "failure_recovery"],
            allowed_tools=["read_file", "write_file", "edit_file", "list_files", "grep", "glob", "run_command"],
            workspace_scope=WorkspaceScope.PROJECT,
            model_policy="reasoning_high",
            system_instructions=(
                "You are a debugger. Analyze test failures, error messages, and unexpected "
                "behavior. Identify root causes and implement fixes methodically."
            )
        ))

        # 11. REVIEWER
        self.register(AgentProfile(
            name="reviewer",
            role=AgentRole.REVIEWER,
            display_name="Reviewer",
            description="Reviews code changes for quality, correctness, and architecture",
            capabilities=["code_review", "architecture_review", "best_practices"],
            preferred_task_types=["review", "quality"],
            allowed_tools=["read_file", "list_files", "grep", "glob"],
            denied_tools=["write_file", "edit_file", "delete_file", "git_commit", "run_command"],
            workspace_scope=WorkspaceScope.READ_ONLY,
            model_policy="reasoning_high",
            system_instructions=(
                "You are a code reviewer. Inspect changes for correctness, security, "
                "performance, and adherence to project conventions. You do not modify code."
                "End with APPROVED or CHANGES_REQUESTED."
            )
        ))

        # 12. SECURITY_REVIEWER
        self.register(AgentProfile(
            name="security_reviewer",
            role=AgentRole.SECURITY_REVIEWER,
            display_name="Security Reviewer",
            description="Audits codebase for security vulnerabilities",
            capabilities=["security_audit", "vulnerability_detection", "threat_modeling"],
            preferred_task_types=["security", "audit"],
            allowed_tools=["read_file", "list_files", "grep"],
            denied_tools=["write_file", "edit_file", "delete_file", "git_commit", "run_command"],
            workspace_scope=WorkspaceScope.READ_ONLY,
            model_policy="reasoning_high",
            system_instructions=(
                "You are a security reviewer. Audit the codebase for vulnerabilities, "
                "improper access controls, injection risks, and bad practices. Do not modify code."
            )
        ))

        # 13. VERIFIER
        self.register(AgentProfile(
            name="verifier",
            role=AgentRole.VERIFIER,
            display_name="Verifier",
            description="Checks output formats, conventions, and basic invariants",
            capabilities=["linting", "formatting", "invariant_checks"],
            preferred_task_types=["verification", "lint"],
            allowed_tools=["read_file", "grep"],
            denied_tools=["write_file", "edit_file", "delete_file", "git_commit", "run_command"],
            workspace_scope=WorkspaceScope.READ_ONLY,
            model_policy="deterministic",
            system_instructions=(
                "You are a verifier. Ensure code meets formatting constraints, passes "
                "linters, and adheres to structural invariants."
            )
        ))

        # 14. GIT_RELEASE
        self.register(AgentProfile(
            name="git_release",
            role=AgentRole.GIT_RELEASE,
            display_name="Release Engineer",
            description="Manages commits, changelogs, tags, and deployment scripts",
            capabilities=["git_management", "changelog", "release"],
            preferred_task_types=["release", "versioning"],
            allowed_tools=["read_file", "write_file", "run_command"],
            workspace_scope=WorkspaceScope.PROJECT,
            model_policy="coding",
            system_instructions=(
                "You are a release engineer. Review changes, construct descriptive commit "
                "messages, generate changelogs, and prepare release scripts."
            )
        ))

        # Legacy Coder / Analyzer mapped to broad defaults
        self.register(AgentProfile(
            name="coder",
            role=AgentRole.CODER,
            display_name="Coder",
            description="Implements code changes, fixes bugs, and writes new features",
            capabilities=["code_writing", "bug_fixing", "feature_implementation", "refactoring"],
            preferred_task_types=["implementation", "bug_fix", "refactoring"],
            allowed_tools=["read_file", "write_file", "edit_file", "list_files", "grep", "glob", "run_command"],
            workspace_scope=WorkspaceScope.PROJECT,
            model_policy="coding",
            system_instructions=(
                "You are a software engineer. Implement code changes as specified. "
                "Write clean, well-structured code that follows project conventions."
            )
        ))
        
        self.register(AgentProfile(
            name="analyzer",
            role=AgentRole.ANALYZER,
            display_name="Analyzer",
            description="Analyzes code quality, identifies issues, and proposes improvements",
            capabilities=["code_analysis", "issue_detection", "architecture_review"],
            preferred_task_types=["analysis", "review", "debugging"],
            allowed_tools=["read_file", "list_files", "grep", "glob", "run_command"],
            workspace_scope=WorkspaceScope.READ_ONLY,
            model_policy="reasoning_high",
            system_instructions=(
                "You are a code analyzer. Examine code for bugs, security issues, "
                "performance problems, and architectural concerns."
            )
        ))

    def register(self, profile: AgentProfile) -> None:
        """Register an agent profile."""
        self._agents[profile.name] = profile

    def unregister(self, name: str) -> bool:
        """Remove an agent profile."""
        if name in self._agents:
            del self._agents[name]
            return True
        return False

    def get(self, name: str) -> AgentProfile | None:
        """Get an agent profile by name."""
        return self._agents.get(name)

    def list_all(self) -> list[AgentProfile]:
        """List all registered agents."""
        return list(self._agents.values())

    def list_enabled(self) -> list[AgentProfile]:
        """List all enabled agents."""
        return [a for a in self._agents.values() if a.enabled]

    def find_by_role(self, role: AgentRole) -> list[AgentProfile]:
        """Find agents by role."""
        return [a for a in self._agents.values() if a.role == role and a.enabled]

    def find_for_task(self, task_type: str) -> list[AgentProfile]:
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

    def get_default_for_role(self, role: AgentRole) -> AgentProfile | None:
        """Get the default (first enabled) agent for a role."""
        agents = self.find_by_role(role)
        return agents[0] if agents else None

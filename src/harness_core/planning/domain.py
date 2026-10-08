"""Domain objects for the autonomous planner."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any


class TaskIntent(enum.Enum):
    """Classifies the user's request intent for execution strategy selection.

    Intent determines the execution path -- not every request needs a
    full planner -> TaskGraph -> multi-agent pipeline.
    """

    EXPLAIN = "explain"              # explain, describe, summarize (read-only)
    ANALYZE = "analyze"              # inspect, audit, review (read-only)
    RESEARCH = "research"            # investigate, explore, discover (read-only)
    IMPLEMENT = "implement"          # build new functionality
    MODIFY = "modify"                # change existing code
    DEBUG = "debug"                  # diagnose and fix defects
    TEST = "test"                    # run or write tests
    REFACTOR = "refactor"            # restructure without behavior change
    REVIEW = "review"                # code review (read-only unless fixes requested)
    GIT = "git"                      # commit, push, branch
    DEPLOY = "deploy"                # deploy, release
    MIXED = "mixed"                  # combination of intents

    # Backward compatibility aliases
    READ_ONLY = "read_only"          # deprecated alias for EXPLAIN

    @property
    def is_read_only(self) -> bool:
        """True when this intent never modifies the repository."""
        return self in (
            TaskIntent.EXPLAIN,
            TaskIntent.ANALYZE,
            TaskIntent.RESEARCH,
            TaskIntent.REVIEW,
        )


class TaskComplexity(enum.Enum):
    """Estimated complexity determines execution strategy."""

    TRIVIAL = "trivial"     # single tool call or simple answer
    SIMPLE = "simple"       # understand -> execute -> verify -> answer
    MEDIUM = "medium"       # plan -> execute -> verify -> repair -> complete
    COMPLEX = "complex"     # planner -> TaskGraph -> workers -> integration -> verify


@dataclass
class TaskContract:
    """Structured contract defining the full scope of a user request.

    Every user request becomes a TaskContract before execution begins.
    The contract captures intent, scope, constraints, and completion
    criteria so the runtime can choose the right execution strategy
    and verify completion honestly.

    This is the SINGLE SOURCE OF TRUTH for what the task requires.
    The TaskGraph, TODO generator, AgentLoop, and CompletionEvaluator
    all consult this contract.
    """

    objective: str = ""
    intent: TaskIntent = TaskIntent.IMPLEMENT
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    workspace: str = ""
    scope: str = "project"                  # project, frontend, backend, etc.
    allowed_operations: list[str] = field(default_factory=list)
    forbidden_operations: list[str] = field(default_factory=list)
    expected_outcome: str = ""
    requirements: list[str] = field(default_factory=list)
    verification_requirements: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    execution_mode: str = "autonomous"       # autonomous, interactive, dry_run
    risk_level: str = "low"                  # low, medium, high

    @property
    def is_read_only(self) -> bool:
        """True when this intent never modifies the repository."""
        return self.intent.is_read_only

    @property
    def needs_verification(self) -> bool:
        """True when the runtime must run the VerificationEngine.

        Read-only tasks (EXPLAIN, ANALYZE, RESEARCH, REVIEW) do not
        need verification because they produce no file changes.
        """
        return not self.is_read_only

    @property
    def needs_git(self) -> bool:
        """True when the task requires Git operations."""
        return self.intent == TaskIntent.GIT

    @property
    def needs_planning(self) -> bool:
        """True when the task should go through model-driven planning.

        Read-only and trivial tasks can skip the planner and use a
        deterministic workflow instead.
        """
        return self.complexity in (
            TaskComplexity.MEDIUM,
            TaskComplexity.COMPLEX,
        ) and not self.is_read_only

    def should_skip_verification(self) -> bool:
        """True when verification should be entirely skipped."""
        return self.is_read_only

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "intent": self.intent.value,
            "complexity": self.complexity.value,
            "workspace": self.workspace,
            "scope": self.scope,
            "allowed_operations": self.allowed_operations,
            "forbidden_operations": self.forbidden_operations,
            "expected_outcome": self.expected_outcome,
            "requirements": self.requirements,
            "verification_requirements": self.verification_requirements,
            "completion_criteria": self.completion_criteria,
            "execution_mode": self.execution_mode,
            "risk_level": self.risk_level,
            "is_read_only": self.is_read_only,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskContract:
        intent_str = data.get("intent", "implement")
        try:
            intent = TaskIntent(intent_str)
        except ValueError:
            intent = TaskIntent.IMPLEMENT
        complexity_str = data.get("complexity", "medium")
        try:
            complexity = TaskComplexity(complexity_str)
        except ValueError:
            complexity = TaskComplexity.MEDIUM
        return cls(
            objective=data.get("objective", ""),
            intent=intent,
            complexity=complexity,
            workspace=data.get("workspace", ""),
            scope=data.get("scope", "project"),
            allowed_operations=data.get("allowed_operations", []),
            forbidden_operations=data.get("forbidden_operations", []),
            expected_outcome=data.get("expected_outcome", ""),
            requirements=data.get("requirements", []),
            verification_requirements=data.get("verification_requirements", []),
            completion_criteria=data.get("completion_criteria", []),
            execution_mode=data.get("execution_mode", "autonomous"),
            risk_level=data.get("risk_level", "low"),
        )


@dataclass
class PlannedTask:
    """A task generated by the planner before becoming a true SubTask."""

    task_id: str
    title: str
    objective: str
    role: str
    dependencies: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    workspace_scope: str = "project"
    priority: int = 5
    model_policy: str = "auto"
    expected_artifacts: list[str] = field(default_factory=list)
    resources: list[dict[str, str]] = field(default_factory=list)
    # Phase 9: requirements this task implements (proposed by the planner as
    # strings like "REQ-001"; the runtime validates & records the mapping).
    traceable_to: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlannedTask:
        return cls(
            task_id=data.get("task_id", ""),
            title=data.get("title", ""),
            objective=data.get("objective", ""),
            role=data.get("role", ""),
            dependencies=data.get("dependencies", []),
            success_criteria=data.get("success_criteria", []),
            workspace_scope=data.get("workspace_scope", "project"),
            priority=data.get("priority", 5),
            model_policy=data.get("model_policy", "auto"),
            expected_artifacts=data.get("expected_artifacts", []),
            resources=data.get("resources", []),
            traceable_to=list(data.get("traceable_to", [])),
        )


@dataclass
class Plan:
    """A collection of planned tasks representing an execution graph."""

    tasks: list[PlannedTask] = field(default_factory=list)
    summary: str = ""
    contract: TaskContract | None = None  # attached by classify_request

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        return cls(
            tasks=[PlannedTask.from_dict(t) for t in data.get("tasks", [])],
            summary=data.get("summary", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> Plan:
        """Parse JSON into a Plan."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class PlanningConstraint:
    """Constraints passed to the planner."""

    available_roles: list[str] = field(default_factory=list)
    available_tools: list[str] = field(default_factory=list)


@dataclass
class PlanningResult:
    """Result of a planning operation."""

    plan: Plan | None = None
    success: bool = False
    errors: list[str] = field(default_factory=list)
    raw_response: str = ""
    contract: TaskContract | None = None  # intent classification result


# ── Intent classification ─────────────────────────────────────────────

# Keyword patterns for intent detection (order matters: most specific first).
_INTENT_PATTERNS: dict[TaskIntent, list[str]] = {
    TaskIntent.GIT: [
        "commit", "push", "pull request", "merge", "branch",
        "revert", "stash", "git",
    ],
    TaskIntent.TEST: [
        "run the test", "run tests", "test suite", "write tests", "add tests",
        "test coverage", "unit test", "integration test",
    ],
    TaskIntent.DEBUG: [
        "fix the", "fix bug", "debug", "error in", "failing",
        "broken", "doesn't work", "not working", "crash",
        "exception", "traceback",
    ],
    TaskIntent.EXPLAIN: [
        "explain", "describe", "what is", "what does", "how does",
        "overview", "summarize", "tell me about",
    ],
    TaskIntent.RESEARCH: [
        "investigate", "explore", "discover", "find out",
        "research", "look into", "analyze this",
    ],
    TaskIntent.ANALYZE: [
        "analyze", "audit", "assess", "evaluate",
        "security review", "code review", "performance analysis",
    ],
    TaskIntent.REVIEW: [
        "review", "inspect", "show me", "list", "read",
    ],
    TaskIntent.REFACTOR: [
        "refactor", "rename", "move", "restructure",
        "clean up", "simplify", "extract",
    ],
    TaskIntent.MODIFY: [
        "update", "change", "modify", "adjust", "tweak",
        "improve", "enhance", "tune",
    ],
    TaskIntent.IMPLEMENT: [
        "build", "create", "implement", "add", "new",
        "make", "develop", "write", "construct",
    ],
    TaskIntent.DEPLOY: [
        "deploy", "release", "ship", "publish",
        "rollout", "launch",
    ],
}

# Complexity indicators
_SIMPLE_INDICATORS = [
    "explain", "what is", "describe", "show me", "list",
]
_MEDIUM_INDICATORS = [
    "fix", "update", "refactor", "add", "change",
]
_COMPLEX_INDICATORS = [
    "build a", "create a", "implement", "full stack",
    "end to end", "complete", "production", "dashboard",
    "website", "application", "api", "database",
]


def classify_request(request: str) -> TaskContract:
    """Classify a user request into a structured TaskContract.

    Uses keyword matching for fast, deterministic intent detection.
    The model may override this classification during planning, but
    the initial contract gives the runtime the right execution strategy
    from the start.
    """
    low = request.lower().strip()

    # Intent detection: score each intent by keyword matches
    scores: dict[TaskIntent, int] = {intent: 0 for intent in TaskIntent}
    for intent, keywords in _INTENT_PATTERNS.items():
        for kw in keywords:
            if kw in low:
                scores[intent] += 1

    # Pick the highest-scoring intent; default to IMPLEMENT
    best_intent = TaskIntent.IMPLEMENT
    best_score = 0
    for intent, score in scores.items():
        if score > best_score:
            best_score = score
            best_intent = intent

    # If multiple intents score equally, it's MIXED
    top_intents = [i for i, s in scores.items() if s == best_score and s > 0]
    if len(top_intents) > 1:
        # If one is IMPLEMENT and another is specific, prefer the specific one
        non_implement = [i for i in top_intents if i not in (
            TaskIntent.IMPLEMENT, TaskIntent.MIXED,
        )]
        if non_implement:
            best_intent = non_implement[0]
        else:
            best_intent = TaskIntent.MIXED

    # Complexity estimation
    complexity = TaskComplexity.MEDIUM  # default
    if any(ind in low for ind in _SIMPLE_INDICATORS) and best_score <= 1:
        complexity = TaskComplexity.SIMPLE
    elif any(ind in low for ind in _COMPLEX_INDICATORS):
        complexity = TaskComplexity.COMPLEX
    elif any(ind in low for ind in _MEDIUM_INDICATORS):
        complexity = TaskComplexity.MEDIUM

    # Trivial: very short read-only requests
    if best_intent == TaskIntent.READ_ONLY and len(request.split()) <= 8:
        complexity = TaskComplexity.TRIVIAL

    # Risk assessment
    risk = "low"
    if best_intent in (TaskIntent.IMPLEMENT, TaskIntent.MODIFY):
        risk = "medium"
    if "database" in low or "production" in low or "deploy" in low:
        risk = "high"

    # Scope detection
    scope = "project"
    if any(w in low for w in ("frontend", "ui", "css", "html", "react", "vue")):
        scope = "frontend"
    elif any(w in low for w in ("backend", "api", "server", "endpoint")):
        scope = "backend"
    elif any(w in low for w in ("database", "db", "schema", "migration")):
        scope = "database"
    elif any(w in low for w in ("test", "spec", "coverage")):
        scope = "tests"

    # Compute allowed/forbidden operations based on intent
    allowed, forbidden = _operations_for_intent(best_intent)

    return TaskContract(
        objective=request,
        intent=best_intent,
        complexity=complexity,
        scope=scope,
        allowed_operations=allowed,
        forbidden_operations=forbidden,
        expected_outcome=_expected_outcome(best_intent),
        risk_level=risk,
    )


def _operations_for_intent(intent: TaskIntent) -> tuple[list[str], list[str]]:
    """Return (allowed, forbidden) operations for the given intent."""
    if intent.is_read_only:
        return [
            "read_file", "list_files", "glob", "grep",
            "git_status", "git_diff", "git_log", "git_remote",
            "git_identity",
        ], [
            "write_file", "edit_file", "run_command",
            "git_add", "git_commit", "git_push",
        ]
    if intent == TaskIntent.GIT:
        return [
            "read_file", "list_files", "glob", "grep",
            "git_status", "git_diff", "git_log", "git_remote",
            "git_identity", "git_add", "git_commit", "git_push",
            "run_command",
        ], []
    if intent == TaskIntent.TEST:
        return [
            "read_file", "list_files", "glob", "grep",
            "run_command",
        ], [
            "write_file", "edit_file",
        ]
    # IMPLEMENT, MODIFY, DEBUG, REFACTOR, DEPLOY, MIXED — full access
    return [
        "read_file", "list_files", "glob", "grep",
        "write_file", "edit_file", "run_command",
        "git_status", "git_diff", "git_log", "git_remote",
        "git_identity", "git_add", "git_commit",
    ], []


def _expected_outcome(intent: TaskIntent) -> str:
    """Map intent to a human-readable expected outcome."""
    return {
        TaskIntent.EXPLAIN: "Explanation generated from repository inspection",
        TaskIntent.READ_ONLY: "Explanation generated from repository inspection",
        TaskIntent.ANALYZE: "Analysis complete with findings and recommendations",
        TaskIntent.RESEARCH: "Research complete with findings",
        TaskIntent.IMPLEMENT: "Requirements implemented and verification passed",
        TaskIntent.MODIFY: "Code modified and existing tests still pass",
        TaskIntent.DEBUG: "Defect fixed and verification passed",
        TaskIntent.TEST: "Requested tests executed and results reported",
        TaskIntent.REFACTOR: "Code restructured without behavior change",
        TaskIntent.REVIEW: "Review complete with findings",
        TaskIntent.GIT: "Requested Git operation completed successfully",
        TaskIntent.DEPLOY: "Deployment completed",
        TaskIntent.MIXED: "All sub-tasks completed and verified",
    }.get(intent, "Task completed")

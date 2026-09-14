"""The core agent loop — orchestrates the engineering workflow."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness_core.agent.completion import can_complete_task, completion_blockers
from harness_core.runtime.steering import SteeringBuffer
from harness_core.agent.todos import (
    apply_tool_result,
    apply_tool_started,
    fallback_todo_plan,
    reconcile_on_evidence,
    sanitize_todo_titles,
    select_todo,
)
from harness_core.agent.types import (
    AgentConfig,
    Task,
    TaskStatus,
    TodoItem,
    TodoStatus,
    ToolCall,
    ToolResult,
    ToolResults,
    ToolResultStatus,
)
from harness_core.agent.steering import SteeringBuffer, steering_event_data
from harness_core.agent.ask_user import (
    AskUserManager,
    QuestionStatus,
    question_answered_payload,
)
from harness_core.agent.micro import default_micro_registry
from harness_core.context.compaction import AgentMessage, ContextCompactor
from harness_core.context.engine import ContextEngine
from harness_core.context.models import ContextRequest
from harness_core.context.pipeline import ContextPipeline
from harness_core.context.pack import estimate_tokens
from harness_core.context.reuse import ContextReuseManager
from harness_core.observability import semantic
from harness_core.observability.run_metrics import RunMetrics
from harness_core.observability.events import Event, EventBus
from harness_core.permissions.manager import PermissionManager
from harness_core.providers.base import CompletionRequest, ModelProvider
from harness_core.routing.budgets import BudgetManager
from harness_core.routing.router import ModelRouter
from harness_core.tools.base import Tool
from harness_core.tools.diagnosis import (
    classify_command_failure,
    normalize_shell_command,
    parse_test_counts,
)
from harness_core.tools.paths import cwd_in_workspace, resolve_in_workspace
from harness_core.verification.engine import VerificationEngine
from harness_core.verification.integrity import check_test_integrity

if TYPE_CHECKING:
    from harness_core.routing.task_aware import TaskAwareRouter


# ── Execution governor limits ────────────────────────────────────────────
# Repeated identical failures: diagnose AND block blind retries at 2.
REPEAT_DIAGNOSIS_THRESHOLD = 2
REPEAT_BLOCK_THRESHOLD = 2
# Stagnation: warn at 2 no-progress iterations, stop at 3.
STAGNATION_WARN_AT = 2
STAGNATION_STOP_AT = 3
# How many times we may re-prompt a model that produced zero tool calls.
MAX_NO_TOOL_NUDGES = 2
# Read-only tool calls without any write/edit: after this many, force text.
_MAX_READONLY_TOOL_CALLS = 25
# Tool names that are read-only (never modify the workspace).
_READONLY_TOOLS = frozenset({
    "read_file", "list_files", "glob", "grep",
    "git_status", "git_diff", "git_log", "git_remote", "git_identity",
})

# Phase 10: typed failure messages. Keys match FailureReason values.
_FAILURE_MESSAGES: dict[str, str] = {
    "model_unavailable": "⚠ Model temporarily unavailable",
    "model_rate_limited": "⚠ All free models rate limited (429)",
    "provider_auth_failure": "✗ Provider authentication failed",
    "payment_required": "✗ Model requires payment (402)",
    "tool_failure": "✗ Tool operation failed",
    "test_failure": "✗ Tests failed",
    "verification_failure": "✗ Verification failed",
    "user_cancelled": "⚠ Task cancelled",
    "task_stagnation": "✗ Task stagnation — no progress",
    "completion_invariant": "✗ Completion invariant violated",
    "permission_denied": "⚠ Permission denied",
    "budget_exceeded": "⚠ Budget exceeded",
    "unknown": "✗ Unknown error",
}

# Phase 11: pause-specific messaging
_PAUSED_MESSAGES: dict[str, str] = {
    "model_unavailable": "⚠ Model temporarily unavailable — task paused, state preserved",
    "model_rate_limited": "⚠ Free models temporarily unavailable (429) — task paused, state preserved",
    "provider_auth_failure": "⚠ Provider authentication failed — task paused, state preserved",
    "payment_required": "⚠ All viable models require payment — task paused, state preserved",
}


def _classify_model_failure(err: str) -> str:
    """Map a model error string to a typed FailureReason value."""
    low = (err or "").lower()
    if "429" in low or "rate limit" in low or "too many requests" in low:
        return "model_rate_limited"
    if "402" in low or "payment required" in low:
        return "payment_required"
    if "401" in low or "unauthorized" in low:
        return "provider_auth_failure"
    if "403" in low or "forbidden" in low:
        return "provider_auth_failure"
    if "all" in low and "failed" in low:
        return "model_unavailable"
    if "network" in low or "connection" in low or "timeout" in low:
        return "model_unavailable"
    return "model_unavailable"


# How many iterations may be spent in diagnosis mode before stopping.
MAX_DIAGNOSIS_ITERATIONS = 5

# ── Context window management ────────────────────────────────────────────
# Three independent mechanisms, weakest to strongest:
#
#   1. HISTORY_TOKEN_BUDGET decides *when* older tool calls stop being sent
#      verbatim and collapse into a single TASK STATE summary.
#   2. MAX_TOOL_RESULT_TOKENS caps any *single* tool result, so one enormous
#      file read or test dump cannot dominate the prompt on its own.
#   3. AgentConfig.context_token_budget is a hard ceiling on the whole
#      assembled message list. _build_messages drops history until the
#      request fits, so exceeding the model's window is not merely unlikely.
#
# Mechanisms 1 and 2 keep prompts small in the common case; mechanism 3 is
# what makes an over-limit request impossible.
#
# History compaction: keep recent tool calls verbatim, summarize older.
HISTORY_CHAR_BUDGET = 120_000
# The trigger is expressed in tokens; ~4 characters per token.
HISTORY_TOKEN_BUDGET = HISTORY_CHAR_BUDGET // 4
KEEP_RECENT_CALLS = 10
# No single tool result may occupy more than this share of the prompt.
MAX_TOOL_RESULT_TOKENS = 8_000
# Used when no AgentConfig budget is available.
DEFAULT_CONTEXT_TOKEN_BUDGET = 120_000
# Space held back for the TASK STATE summary whenever history is dropped, so
# adding the summary cannot itself push the request over the ceiling.
_SUMMARY_TOKEN_RESERVE = 1_500
# A tool result is never shrunk below this: less than this and the model gets
# no usable signal, at which point dropping the call outright would be better.
_MIN_TOOL_RESULT_TOKENS = 256
# Marker inserted where the middle of an oversized tool result was removed.
_TRUNCATION_NOTICE = (
    "\n\n... [{removed:,} characters omitted by the harness to protect the "
    "context window; re-read a narrower range if you need the middle] ...\n\n"
)

_TEST_COMMAND_MARKERS = (
    "pytest", "npm test", "yarn test", "pnpm test", "bun test",
    "cargo test", "go test", "jest", "vitest", "mocha", "phpunit",
    "dotnet test", "gradle test", "mvn test",
)


def truncate_to_tokens(text: str, max_tokens: int = MAX_TOOL_RESULT_TOKENS) -> str:
    """Clamp `text` to roughly `max_tokens` tokens, keeping head and tail.

    Tool output is most useful at its edges: the head carries the opening of a
    file or the start of a command, the tail carries the failing assertion or
    the summary line. The middle is dropped and replaced with an explicit
    notice so the model knows the content is partial rather than complete.
    """
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    # The notice itself costs tokens, so it comes out of the budget rather
    # than being added on top — otherwise the "capped" result overshoots.
    notice_len = len(_TRUNCATION_NOTICE.format(removed=len(text)))
    body_chars = max(max_chars - notice_len, 8)
    # Favour the head — for source files the beginning carries imports and
    # signatures — but always keep enough of the tail to see a final error.
    head_chars = (body_chars * 7) // 10
    tail_chars = body_chars - head_chars
    removed = len(text) - head_chars - tail_chars
    return text[:head_chars] + _TRUNCATION_NOTICE.format(removed=removed) + text[-tail_chars:]


def message_tokens(message: dict[str, Any]) -> int:
    """Estimate the prompt tokens contributed by one chat message.

    Counts the textual content plus any serialized tool_calls payload, since
    a message with `content: None` still costs tokens for the function name
    and arguments it carries.
    """
    total = estimate_tokens(message.get("content") or "")
    tool_calls = message.get("tool_calls")
    if tool_calls:
        total += estimate_tokens(json.dumps(tool_calls))
    # Small fixed overhead per message for role and framing tokens.
    return total + 4


def messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate the total prompt tokens for an assembled message list."""
    return sum(message_tokens(m) for m in messages)


class AgentLoop:
    """The core agent loop that drives the engineering workflow."""

    def __init__(
        self,
        provider: ModelProvider,
        tools: list[Tool],
        workspace_root: Path | None = None,
        config: AgentConfig | None = None,
        event_bus: EventBus | None = None,
        router: ModelRouter | None = None,
        task_aware: TaskAwareRouter | None = None,
        agent_id: str = "",
        task_id: str = "",
        run_id: str = "",
        steering_buffer: "SteeringBuffer | None" = None,
        context_pipeline: ContextPipeline | None = None,
    ) -> None:
        self.provider = provider
        self.tools = {t.schema.name: t for t in tools}
        self.workspace_root = workspace_root or Path.cwd()
        self.config = config or AgentConfig()
        self.event_bus = event_bus or EventBus()
        self.router = router
        self.task_aware = task_aware
        # Phase 10: identity stamped on emitted events so live dashboards can
        # attribute tool/test activity to the correct agent without guessing.
        self.agent_id = agent_id
        self.task_id = task_id
        # Harness 2.0 Phase 1: run identity + steering + context intelligence.
        # All optional with safe defaults so existing call sites are unaffected.
        import uuid as _uuid
        self.run_id = run_id or _uuid.uuid4().hex[:12]
        self.steering = steering_buffer or SteeringBuffer()
        self.ask_user = AskUserManager()
        self.micro_workers = default_micro_registry()
        self.run_metrics = RunMetrics(run_id=self.run_id)
        self._cancelled = False
        self.budget = BudgetManager() if router is None else router.budget
        self.context_engine = ContextEngine(self.workspace_root)
        self.context_reuse = ContextReuseManager()
        # Context intelligence pipeline reuses the SAME ContextReuseManager
        # instance (Part 6) — one snapshot cache, no duplicate systems.
        self._context_pipeline = context_pipeline or ContextPipeline(
            self.workspace_root, reuse=self.context_reuse
        )
        self._last_context_selection: Any = None
        self.permission_manager = PermissionManager(
            self.workspace_root,
            autonomous_mode=self.config.autonomous_mode,
        )
        self.verification_engine = VerificationEngine(self.workspace_root)
        self._current_phase: str = ""
        self._recent_denials: list[tuple[str, str]] = []  # (tool_name, args_key)
        self._consecutive_denials: int = 0
        self._recent_failures: list[tuple[str, str, int]] = []  # (tool_name, args_key, exit_code)
        self._consecutive_failures: int = 0
        self._last_command_hash: str | None = None
        # Execution governor state
        self._project_info: dict[str, Any] | None = None
        self._failure_counts: dict[str, int] = {}  # action_key -> consecutive failures
        self._seen_actions: set[str] = set()  # action keys attempted (progress detection)
        self._stagnation_counter: int = 0
        self._diagnosis_active: bool = False
        self._diagnosis_iterations: int = 0
        self._no_tool_nudges: int = 0
        self._readonly_tool_calls: int = 0
        self._corrections: list[str] = []  # injected guidance messages
        self._modified_files: list[str] = []  # files written/edited (raw paths)
        self._had_test_failure: bool = False
        self._last_model_used: str = ""  # for model rotation on no-tool responses
        self._active_task: Task | None = None  # currently running task (cancellation reporting)
        self._completed_operations: set[str] = set()  # successful operation keys (Phase 8)
        self._pending_workflow_events: list = []  # deferred events from workflows

        # Steering: single unified buffer (Harness 2.0 Part 7). The legacy
        # event-driven entry point (steering.received) feeds the same buffer
        # as the programmatic submit_steering_message API.
        if self.event_bus:
            self.event_bus.on("steering.received", self._on_steering)

    async def _on_steering(self, event: Event) -> None:
        """Handle steering messages injected by the user during execution.

        Event-driven bridge into the unified SteeringBuffer: the buffer is
        the only steering state; this keeps the existing event API working
        while the programmatic API and the loop drain from one queue.
        """
        message = event.data.get("message")
        if message:
            await self._enqueue_steering(str(message))

    def _workspace_snapshot_text(self) -> str:
        """Compact workspace snapshot for model context.

        Built from project discovery so the model never has to guess
        whether a workspace exists or probe the environment endlessly.
        """
        info = self._project_info
        if not info:
            return ""
        lines: list[str] = [f"Workspace root: {info.get('root', '')}"]
        files = info.get("files", [])
        if files:
            lines.append("Files: " + ", ".join(files[:20]))
        if info.get("languages"):
            lines.append("Languages: " + ", ".join(info["languages"]))
        if info.get("package_manager"):
            lines.append(f"Package manager: {info['package_manager']}")
        if info.get("has_tests"):
            lines.append("Test suite: detected")
        if info.get("has_git"):
            lines.append("Git repository: yes")
        if info.get("readme"):
            lines.append(f"README: {info['readme']}")
        return "\n".join(lines)

    def _workspace_has_files(self) -> bool:
        return bool(self._project_info and self._project_info.get("files"))

    def _cached_file_read(self, path: str) -> str | None:
        """Read a file with context-reuse awareness.

        Returns the file content if the file hasn't changed since the last
        read, or None if a fresh read is needed.  When a fresh read is
        performed the snapshot is recorded for future checks.

        This prevents the model from receiving identical file content on
        repeated reads — the ContextReuseManager tracks which files have
        been read and whether they have changed (by hash/mtime).
        """
        try:
            p = Path(path)
            if not p.exists():
                return None
            stat = p.stat()
            snap = self.context_reuse.snapshot(path)
            if snap is not None and not self.context_reuse.is_unchanged(
                path, size=stat.st_size, mtime_ns=int(stat.st_mtime * 1e9)
            ):
                # File changed — re-read and update snapshot
                content = p.read_text(encoding="utf-8", errors="replace")
                self.context_reuse.record_read(
                    path, content=content, size=stat.st_size,
                    mtime_ns=int(stat.st_mtime * 1e9),
                )
                return content
            if snap is None:
                # First read — record it
                content = p.read_text(encoding="utf-8", errors="replace")
                self.context_reuse.record_read(
                    path, content=content, size=stat.st_size,
                    mtime_ns=int(stat.st_mtime * 1e9),
                )
                return content
            # File unchanged — return None to signal "no fresh read needed"
            return None
        except Exception:
            return None

    def invalidate_file_cache(self, path: str) -> None:
        """Invalidate the cached snapshot for a file after a write/edit."""
        self.context_reuse.invalidate(path)


    def _system_prompt(self) -> str:
        """Build the workspace-aware system prompt."""
        base = """You are an autonomous software engineering agent operating INSIDE a real workspace.

Your goal is to complete engineering tasks reliably. You must:
1. Understand the task
2. Plan your approach
3. Execute tools to inspect and modify code
4. Verify your changes work
5. Report results with evidence

You have filesystem and execution tools. Use them to read, edit, write, search, and run commands.

WORKSPACE RULES:
- The workspace and its files are REAL and accessible through your tools.
- NEVER claim you lack visibility into the project. If you need information, use a tool.
- NEVER ask the user for information you can discover with tools (project type,
  tech stack, file names, test commands). Discover it yourself.
- Inspect relevant files BEFORE modifying them.
- Prefer acting over explaining. Do not answer a coding task with prose only.

CRITICAL: You MUST produce a text response (your final answer) after gathering
enough information. Do NOT keep calling tools indefinitely. Once you have read
the relevant files and have enough context, STOP calling tools and write your
response. For explanation requests, read a few key files then explain. For
coding requests, implement the change then verify it. Never read the same
file more than twice."""

        snapshot = self._workspace_snapshot_text()
        if snapshot:
            base += f"\n\nCURRENT WORKSPACE (already discovered, do not re-probe):\n{snapshot}"

        return base + """

CRITICAL RULE: NEVER claim a task is complete when a required tool call failed.
The runtime execution results (exit codes, stderr) are the source of truth.
If a command exits with a non-zero exit code, the task is NOT complete.
Your text response cannot override actual tool failures.

If a command fails:
- Read the error output carefully
- Diagnose the root cause BEFORE retrying
- Fix the implementation code
- Run the command again
- Only claim success when the command passes
- NEVER run the exact same failing command more than twice; diagnose instead.
- NEVER weaken, delete, or skip tests just to make them pass. Fix the implementation.

IMPORTANT: If a tool call returns "permission denied", do NOT retry the same command.
Instead:
- Try a different approach that does not require the blocked operation
- If verification is blocked, skip verification and report what was completed
- Never retry a denied command more than once
- Accept the permission constraint and work within it

CRITICAL GIT RULE: NEVER invent Git identity (user.name, user.email).
- If git_identity check fails, report that identity is missing and stop.
- Do NOT run: git config user.name "Some Name"
- Do NOT run: git config user.email "some@email.com"
- Configure your OWN identity manually if needed.
- The agent must NEVER write fake placeholder identities into repositories.

Always verify your work before claiming success. Do not claim success without evidence.
However, if verification itself is blocked by permissions, report that clearly.


When you are done, summarize what you did and provide evidence of success."""

    @staticmethod
    def _action_key(tool_name: str, arguments: dict[str, Any]) -> str:
        """Normalized identity for a tool call (used for repetition detection).

        For run_command, the command is normalized (cd / shell wrappers
        stripped) so retrying the same underlying command through different
        shell formats counts as the same action — blocking shell-guessing loops.
        """
        args = arguments
        if tool_name == "run_command" and isinstance(arguments.get("command"), str):
            args = dict(arguments)
            args["command"] = normalize_shell_command(arguments["command"])
        return f"{tool_name}:{json.dumps(args, sort_keys=True)}"

    @staticmethod
    def _tool_summary(call: ToolCall) -> str:
        """Safe human-readable summary of a tool call for semantic events.

        Never includes file contents or raw output — only the operation shape.
        """
        name = call.tool_name
        args = call.arguments
        if name == "run_command":
            return f"run: {str(args.get('command', ''))[:120]}"
        if name in ("read_file", "write_file", "edit_file", "list_files"):
            p = args.get("path") or args.get("file_path") or ""
            return f"{name}: {str(p)[:120]}"
        if name in ("glob", "grep"):
            return f"{name}: {str(args.get('pattern', ''))[:80]}"
        if name.startswith("git_"):
            return name
        return name

    @staticmethod
    def _is_test_command(command: str) -> bool:
        """Detect whether a shell command runs a test suite."""
        cmd = command.lower()
        if any(marker in cmd for marker in _TEST_COMMAND_MARKERS):
            return True
        # e.g. "node test.js", "node tests/run.js"
        import re
        return bool(re.search(r"\b(node|python|python3|deno|bun)\s+\S*test\S*", cmd))

    @staticmethod
    def _as_agent_messages(calls: list[ToolCall]) -> list[AgentMessage]:
        """Represent dropped tool calls as AgentMessages for the compactor.

        ContextCompactor works on a generic message stream, so each dropped
        call becomes a `tool_call` message carrying the tool name and path in
        metadata (that is what it reads to report tools used and files
        modified) followed by a `tool_result` or `error` message for the
        outcome.
        """
        msgs: list[AgentMessage] = []
        for tc in calls:
            path = tc.arguments.get("path") or tc.arguments.get("file_path") or ""
            msgs.append(
                AgentMessage(
                    role="assistant",
                    content=f"{tc.tool_name}({path})" if path else tc.tool_name,
                    kind="tool_call",
                    metadata={"tool_name": tc.tool_name, "path": path},
                )
            )
            result = tc.result
            if result is None:
                continue
            failed = result.execution_failed
            msgs.append(
                AgentMessage(
                    role="tool",
                    content=(result.error or result.output or "")[:500] if failed
                    else (result.output or "(no output)")[:500],
                    kind="error" if failed else "tool_result",
                    metadata={"tool_name": tc.tool_name},
                )
            )
        return msgs

    def _compact_task_state(
        self,
        task: Task,
        compacted_count: int,
        compacted_calls: list[ToolCall] | None = None,
    ) -> str:
        """Build the compact TASK STATE summary for older tool history.

        The framing (goal, workspace, plan progress, latest failure) is
        task-specific and stays here; the roll-up of what the dropped calls
        actually did is delegated to ContextCompactor so there is one
        implementation of that summarization in the codebase.
        """
        completed_steps = [
            i.description for i in task.task_plan.items
            if i.status == TodoStatus.COMPLETED
        ]
        lines = [
            "TASK STATE (summary of earlier work)",
            f"Goal: {task.goal}",
        ]
        if self._workspace_has_files():
            lines.append(f"Workspace: {self._project_info.get('root')}")
        if completed_steps:
            lines.append("Completed: " + "; ".join(completed_steps[:6]))
        if self._modified_files:
            lines.append("Changed files: " + ", ".join(self._modified_files[-10:]))
        lines.append(f"{compacted_count} earlier tool call(s) were made and are summarized above.")
        if compacted_calls:
            # preserve_recent=0: every dropped call belongs in the summary,
            # nothing is held back to be replayed verbatim.
            summary = ContextCompactor(preserve_recent=0).compact(
                self._as_agent_messages(compacted_calls)
            )
            if summary.content:
                lines.append(summary.content)
        last_failure = next(
            (tc for tc in reversed(task.tool_calls)
             if tc.result and tc.result.execution_failed and not tc.result.is_perm_denied),
            None,
        )
        if last_failure and last_failure.result:
            cmd = last_failure.arguments.get("command", last_failure.tool_name)
            lines.append(f"Latest failure: {cmd} (exit code {last_failure.result.exit_code})")
            err = (last_failure.result.error or "")[:300]
            if err:
                lines.append(f"Error: {err}")
        return "\n".join(lines)

    def _context_token_budget(self) -> int:
        """The hard ceiling on estimated prompt tokens for one request."""
        budget = getattr(self.config, "context_token_budget", None)
        if not isinstance(budget, int) or budget <= 0:
            return DEFAULT_CONTEXT_TOKEN_BUDGET
        return budget

    @staticmethod
    def _tool_call_pair(tc: ToolCall) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build the assistant/tool message pair for one completed tool call.

        These two messages must always travel together: an assistant message
        announcing a tool_call with no matching tool message is a malformed
        request for most providers, so every trimming decision below operates
        on the pair rather than on individual messages.
        """
        assert tc.result is not None
        raw = tc.result.output or tc.result.error or "(no output)"
        return (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": truncate_to_tokens(raw, MAX_TOOL_RESULT_TOKENS),
            },
        )

    def _build_messages(
        self,
        task: Task,
        context: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the message list for the model, within a hard token ceiling.

        Recent tool calls go in verbatim (each individually capped), older
        history collapses into a single TASK STATE summary, and corrections
        (governor guidance) are appended last. If the result would still
        exceed `AgentConfig.context_token_budget`, the oldest tool call pairs
        are dropped — in pairs, never orphaned — until it fits. The returned
        list is therefore guaranteed to be within budget whenever the fixed
        prefix alone is.
        """
        budget = self._context_token_budget()

        prefix: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
        ]

        # Add context
        if context:
            for piece in context:
                prefix.append(
                    {"role": "system", "content": f"[Context: {piece.source}]\n{piece.content}"}
                )

        # Add task
        prefix.append({"role": "user", "content": task.goal})

        # Governor guidance (nudge / diagnosis / stagnation) — most recent last
        suffix: list[dict[str, Any]] = [
            {"role": "user", "content": correction}
            for correction in self._corrections[-3:]
        ]

        calls_with_results = [tc for tc in task.tool_calls if tc.result]

        # Step 1: decide the *maximum* number of recent calls worth sending
        # verbatim. Once the accumulated history is large, older calls stop
        # earning their place and are better represented by a summary. Note
        # this is a size test only: a long run of small results is cheap and
        # more useful to the model in full.
        history_tokens = sum(
            estimate_tokens(tc.result.output or "") + estimate_tokens(tc.result.error or "")
            for tc in calls_with_results
        )
        if history_tokens > HISTORY_TOKEN_BUDGET:
            candidates = calls_with_results[-KEEP_RECENT_CALLS:]
        else:
            candidates = list(calls_with_results)

        # Step 2: fit as many of those as the budget allows, newest first, so
        # the model always retains the most recent observation it acted on.
        # A summary is reserved for whenever anything gets left out.
        pairs = [self._tool_call_pair(tc) for tc in candidates]
        fixed_tokens = messages_tokens(prefix) + messages_tokens(suffix)
        summary_reserve = _SUMMARY_TOKEN_RESERVE if len(candidates) < len(calls_with_results) else 0
        available = budget - fixed_tokens - summary_reserve

        kept: list[tuple[dict[str, Any], dict[str, Any]]] = []
        used = 0
        for pair in reversed(pairs):
            cost = message_tokens(pair[0]) + message_tokens(pair[1])
            if kept and used + cost > available:
                break
            used += cost
            kept.insert(0, pair)
        dropped_count = len(calls_with_results) - len(kept)

        # Step 3: assemble. The summary sits between the goal and the verbatim
        # history so the model reads it as established background.
        messages = list(prefix)
        if dropped_count > 0:
            dropped_calls = calls_with_results[:dropped_count]
            messages.append(
                {
                    "role": "system",
                    "content": self._compact_task_state(task, dropped_count, dropped_calls),
                }
            )
        for assistant_msg, tool_msg in kept:
            messages.append(assistant_msg)
            messages.append(tool_msg)
        messages.extend(suffix)

        # Step 4: last resort. If the prefix, summary and a single mandatory
        # tool pair still overflow, shrink the largest tool result rather than
        # send a request the provider will reject outright.
        self._enforce_token_ceiling(messages, budget)
        return messages

    @staticmethod
    def _enforce_token_ceiling(messages: list[dict[str, Any]], budget: int) -> None:
        """Shrink oversized tool results in place until `messages` fits.

        Only `role: tool` messages are touched: the system prompt, the goal
        and the governor corrections are the parts the agent cannot function
        without, and unlike tool output they are bounded by construction.
        """
        for _ in range(len(messages)):
            overflow = messages_tokens(messages) - budget
            if overflow <= 0:
                return
            tool_msgs = [m for m in messages if m.get("role") == "tool" and m.get("content")]
            if not tool_msgs:
                return
            biggest = max(tool_msgs, key=lambda m: len(m["content"]))
            # Target the *content* budget, not the message total: the per-message
            # framing overhead is not something truncation can reclaim, so
            # subtracting the overflow from the whole message undershoots.
            target = estimate_tokens(biggest["content"]) - overflow
            if target <= _MIN_TOOL_RESULT_TOKENS:
                # Cannot recover the overflow from this message alone; reduce
                # it to a stub and let the next pass attack the runner-up.
                biggest["content"] = truncate_to_tokens(
                    biggest["content"], _MIN_TOOL_RESULT_TOKENS
                )
                if len(tool_msgs) == 1:
                    return
                continue
            biggest["content"] = truncate_to_tokens(biggest["content"], target)

    def _tool_schemas(self) -> list[dict[str, Any]]:
        """Get LLM-compatible tool schemas."""
        return [t.to_llm_schema() for t in self.tools.values()]

    def _check_repeated_deny(self, call: ToolCall) -> bool:
        """Check if this exact tool call has already been denied.

        Returns True if we should block the repeated denied call.
        """
        args_key = json.dumps(call.arguments, sort_keys=True)
        for prev in self._recent_denials:
            if prev[0] == call.tool_name and prev[1] == args_key:
                return True
        return False

    def _record_denial(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Record a permission denial for loop guard tracking."""
        args_key = json.dumps(arguments, sort_keys=True)
        self._recent_denials.append((tool_name, args_key))
        if len(self._recent_denials) > 50:
            self._recent_denials = self._recent_denials[-50:]
        self._consecutive_denials += 1

    def _reset_denial_tracking(self) -> None:
        """Reset consecutive denial counter after a successful tool call."""
        self._consecutive_denials = 0

    def _record_failure(self, tool_name: str, arguments: dict[str, Any], exit_code: int = -1) -> None:
        """Record a tool execution failure for repetition detection."""
        args_key = json.dumps(arguments, sort_keys=True)
        self._recent_failures.append((tool_name, args_key, exit_code))
        if len(self._recent_failures) > 50:
            self._recent_failures = self._recent_failures[-50:]
        self._consecutive_failures += 1

    def _reset_failure_tracking(self) -> None:
        """Reset consecutive failure counter after a successful tool call."""
        self._consecutive_failures = 0

    def _is_repeating_failure(self, call: ToolCall) -> bool:
        """Check if this exact operation has failed too many times.

        Hard-blocks blind retries at REPEAT_BLOCK_THRESHOLD; diagnosis
        mode is triggered earlier, at REPEAT_DIAGNOSIS_THRESHOLD.
        """
        key = self._action_key(call.tool_name, call.arguments)
        return self._failure_counts.get(key, 0) >= REPEAT_BLOCK_THRESHOLD

    async def _trigger_diagnosis(self, call: ToolCall, attempts: int) -> None:
        """Switch from blind execution into diagnosis mode."""
        self._diagnosis_active = True
        self._diagnosis_iterations = 0
        command = call.arguments.get("command", call.tool_name)
        await self._emit_phase("diagnosing")
        await self._emit_thinking("Repeated failure detected — diagnosing before retrying.")
        await self.event_bus.emit(
            Event(
                type="diagnosis.triggered",
                source="agent_loop",
                data={
                    "tool": call.tool_name,
                    "command": command,
                    "attempts": attempts,
                },
            )
        )
        self._corrections.append(
            f"WARNING: The same operation has failed {attempts} times in a row: "
            f"{command}. Do NOT run it again unchanged. Switch to diagnosis:\n"
            "1. Read the failing test/source files involved.\n"
            "2. Inspect the stderr/stack trace and identify the failing assertion or error.\n"
            "3. Determine the root cause in the IMPLEMENTATION code.\n"
            "4. Apply one targeted fix to the implementation. "
            "Do NOT weaken, delete, or skip tests to force a pass.\n"
            "5. Then re-run the command once."
        )

    async def _record_execution_outcome(self, call: ToolCall, result: ToolResult) -> None:
        """Update governor state after a tool execution."""
        key = self._action_key(call.tool_name, call.arguments)
        command = call.arguments.get("command", "")

        if result.execution_failed and not result.is_perm_denied:
            self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
            if call.tool_name == "run_command" and self._is_test_command(command):
                self._had_test_failure = True
                await self._emit_phase("testing")
            if self._failure_counts[key] == REPEAT_DIAGNOSIS_THRESHOLD:
                await self._trigger_diagnosis(call, self._failure_counts[key])
            return

        # Successful execution
        self._failure_counts.pop(key, None)
        if call.tool_name == "run_command":
            if self._is_test_command(command):
                await self._emit_phase("testing")
                if self._diagnosis_active:
                    # Tests pass again — diagnosis resolved
                    self._diagnosis_active = False
                    self._diagnosis_iterations = 0
        elif call.tool_name in ("write_file", "edit_file"):
            path = call.arguments.get("path", call.arguments.get("file_path", ""))
            if path and path not in self._modified_files:
                self._modified_files.append(path)
            # Invalidate context-reuse snapshot so next read gets fresh content
            if path:
                self.context_reuse.invalidate(path)
            # Code changed: earlier command failures may now be obsolete
            self._failure_counts.clear()
            await self._emit_phase("fixing" if self._diagnosis_active else "implementing")

    def _iteration_made_progress(self, iter_calls: list[ToolCall]) -> bool:
        """True when this iteration produced meaningful progress.

        Progress = a new (never-attempted) action, or a successful file
        modification. Repeating known actions or emitting only prose is
        not progress.
        """
        if not iter_calls:
            return False
        for tc in iter_calls:
            key = self._action_key(tc.tool_name, tc.arguments)
            if key not in self._seen_actions:
                return True
            if (
                tc.tool_name in ("write_file", "edit_file")
                and tc.result is not None
                and tc.result.status == ToolResultStatus.SUCCESS
            ):
                return True
        return False

    # ── Failure reason classification ─────────────────────────────────

    def _classify_failure_reason(self, err: str) -> str:
        """Map an error string to a typed FailureReason value."""
        low = (err or "").lower()
        if "429" in low or "rate limit" in low or "too many requests" in low:
            return "model_rate_limited"
        if "402" in low or "payment required" in low:
            return "payment_required"
        if "401" in low or "unauthorized" in low:
            return "provider_auth_failure"
        if "403" in low or "forbidden" in low:
            return "provider_auth_failure"
        if "all" in low and "failed" in low:
            return "model_unavailable"
        if "network" in low or "connection" in low or "timeout" in low:
            return "model_unavailable"
        return "unknown"

    # ── Workflow helpers ──────────────────────────────────────────────

    def _workflow_context(self) -> Any:
        """Build a WorkflowContext for the currently-running task."""
        from harness_core.agent.workflows import WorkflowContext
        return WorkflowContext(
            workspace=self.workspace_root,
            tools=self.tools,
            event_bus=self.event_bus,
            modified_files=self._modified_files,
            git_commit=getattr(self._active_task, "git_commit", None) if self._active_task else None,
            git_push=getattr(self._active_task, "git_push", None) if self._active_task else None,
            record_tool_call=self._record_workflow_tool,
        )

    async def _record_workflow_tool(
        self, tool_name: str, args: dict[str, Any], result: ToolResult
    ) -> None:
        """Record one workflow-driven tool execution as a real ToolCall.

        Workflows (git push / test / explain) execute tools directly rather
        than through the model loop; without this hook their operations were
        invisible to the runtime accounting (0 tool calls for a real push).
        This makes every executed operation a first-class ToolCall with
        events, iteration/tool accounting and execution stats.
        """
        task = self._active_task
        if task is None:
            return
        call = ToolCall(
            id=f"wf-{tool_name}-{len(task.tool_calls)}",
            tool_name=tool_name,
            arguments=dict(args),
            result=result,
        )
        task.tool_calls.append(call)
        task.iterations += 1
        self.budget.record_iteration()
        self.budget.record_tool_call()
        self._seen_actions.add(self._action_key(tool_name, args))

        await self._emit_event("tool.call", {"tool": tool_name, "args": args})
        event_data: dict[str, Any] = {
            "tool": tool_name,
            "status": result.status.value,
            "output_len": len(result.output or ""),
        }
        if tool_name == "run_command":
            # Phase 10.5: the exact command rides on the result event so the
            # runtime convergence governor can detect repeated commands.
            event_data["command"] = str(args.get("command", ""))
        if result.exit_code is not None:
            event_data["exit_code"] = result.exit_code
        if result.error:
            event_data["error"] = result.error
        if result.stderr:
            event_data["stderr"] = result.stderr
        if result.metadata:
            event_data["metadata"] = dict(result.metadata)
        await self._emit_event("tool.result", event_data)

        task.execution_stats.record_attempt()
        if result.status == ToolResultStatus.SUCCESS:
            task.execution_stats.record_success(tool_name)
            self._completed_operations.add(self._action_key(tool_name, args))
        elif result.status == ToolResultStatus.PERMISSION_DENIED:
            task.execution_stats.record_permission_denied(tool_name)
        else:
            task.execution_stats.record_failure(tool_name)
            if tool_name == "run_command":
                self._record_failure(tool_name, args, result.exit_code or -1)

        # Structured git / test accounting identical to the model-loop path.
        await self._postprocess_result(task, call, result)
        if tool_name in ("write_file", "edit_file"):
            path = args.get("path", args.get("file_path", ""))
            if path and path not in self._modified_files:
                self._modified_files.append(path)
            # Invalidate context-reuse snapshot so next read gets fresh content
            if path:
                self.context_reuse.invalidate(path)
        await self._todo_result(task, call, result)

    def _apply_workflow_result(self, task: Task, result: Any) -> None:
        """Copy workflow outputs onto the task and emit per-step events.

        Emits the structured todo lifecycle events that the UX layer
        listens for. Called once per workflow run.
        """
        from harness_core.observability.events import Event
        data = result.data or {}
        if data.get("commit"):
            task.git_commit = data["commit"]
        if data.get("push"):
            task.git_push = data["push"]
        for op in result.completed_operations:
            self._completed_operations.add(op)
            task.completed_operations.append(op)
            
        if "context_pieces" in data:
            if not hasattr(task, "workflow_context"):
                task.workflow_context = []
            task.workflow_context.extend(data["context_pieces"])
            

        # Stash TODO events for emission by an async wrapper
        for item in task.task_plan.items:
            if not getattr(item, "_workflow_emitted", False):
                item._workflow_emitted = True
                ev_type = (
                    "todo.completed"
                    if item.status == TodoStatus.COMPLETED
                    else "todo.failed" if item.status == TodoStatus.FAILED
                    else "todo.started" if item.status == TodoStatus.IN_PROGRESS
                    else None
                )
                if ev_type:
                    # Defer to the loop's event loop via the existing bus
                    # by storing pending events for an async emit pass.
                    self._pending_workflow_events.append(Event(
                        type=ev_type,
                        source="workflow",
                        data={
                            "todo_id": item.id,
                            "title": item.description,
                            "status": item.status.value,
                            "evidence": item.evidence,
                            "error": item.error,
                        },
                    ))

    async def _flush_workflow_events(self) -> None:
        """Emit any deferred workflow events on the bus."""
        pending = self._pending_workflow_events
        self._pending_workflow_events = []
        for ev in pending:
            await self.event_bus.emit(ev)
        await self._emit_todo_update(self._active_task) if self._active_task else None

    # ── Plan validation ────────────────────────────────────────────────

    def _validate_plan_steps(self, steps: list[str]) -> list[str]:
        """Keep only actionable engineering tasks; drop conversational prose.

        Every TODO must begin with an actionable engineering verb. Questions,
        requests for information, claims of missing visibility, and chatter are
        rejected — the workspace is discoverable via tools.
        """
        return sanitize_todo_titles(steps)

    def _default_plan_steps(self, goal: str = "") -> list[str]:
        """Deterministic fallback plan from the task + workspace.

        Never conversational. Used when the model proposes nothing usable.
        """
        return fallback_todo_plan(goal, self._project_info)

    # ── Completion verification ────────────────────────────────────────

    async def _verify_task_completion(self, task: Task) -> None:
        """Verify claimed completion truthfully.

        Runs test-integrity review plus ecosystem verification checks
        when the task modified files. On failure the task is marked
        FAILED — success is never claimed without evidence.
        """
        if not self.config.verify_on_complete or not self._modified_files:
            return

        # Test integrity review (Phase: don't let agents weaken tests)
        try:
            integrity = check_test_integrity(self.workspace_root, self._modified_files)
        except Exception:
            integrity = None
        if integrity and integrity.suspicious:
            await self.event_bus.emit(
                Event(
                    type="test_integrity.warning",
                    source="agent_loop",
                    data={
                        "warning": integrity.warning,
                        "files": integrity.test_files_modified,
                    },
                )
            )

        await self._emit_phase("verifying")
        await self.event_bus.emit(
            Event(
                type="verification.started",
                source="agent_loop",
                data={"files": list(self._modified_files)},
            )
        )

        try:
            checks = await self.verification_engine.detect_ecosystem()
        except Exception:
            checks = []

        if not checks:
            # No automated checks for this ecosystem — verify files exist
            missing = []
            for f in self._modified_files:
                p = Path(f)
                if not p.is_absolute():
                    try:
                        p = Path(self.workspace_root) / f
                    except TypeError:
                        p = Path(f)
                if not p.exists():
                    missing.append(f)
            if missing:
                task.status = TaskStatus.FAILED
                task.verification_passed = False
                task.verification_summary = f"modified files missing: {', '.join(missing)}"
                task.error = (
                    "Verification failed: modified files do not exist on disk: "
                    f"{', '.join(missing)}"
                )
            else:
                task.verification_passed = True
                task.verification_summary = "changed files present; no automated checks detected"
            await self.event_bus.emit(
                Event(
                    type="verification.completed",
                    source="agent_loop",
                    data={"passed": task.verification_passed, "checks_run": 0},
                )
            )
            return

        report = await self.verification_engine.run_checks(checks[:2])
        passed = report.all_passed
        task.verification_passed = passed
        details = "; ".join(
            f"{r.check_name}: {'passed' if r.passed else 'FAILED'}"
            for r in report.results
        )
        task.verification_summary = details
        await self.event_bus.emit(
            Event(
                type="verification.completed",
                source="agent_loop",
                data={
                    "passed": passed,
                    "checks_run": report.checks_run,
                    "checks_passed": report.checks_passed,
                },
            )
        )

        if not passed:
            failed_output = ""
            for r in report.results:
                if not r.passed:
                    snippet = (r.output or r.error or "")[-800:]
                    failed_output += f"\n[{r.check_name}]\n{snippet}\n"
            task.status = TaskStatus.FAILED
            task.error = (
                "Verification failed: the implementation does not pass the "
                f"project's checks.\n{details}\n{failed_output}".strip()
            )

    def _should_block_completion(self, task: Task) -> bool:
        """Determine if the task should NOT be marked COMPLETED.

        Hard invariant: TOOL FAILURE ≠ TASK SUCCESS

        Returns True when the task must NOT transition to COMPLETED.
        """
        if not task.tool_calls:
            return False  # No tool calls — model can finish freely

        # Check if any tool calls failed (execution failure, not permission denied)
        failed_executions = [
            tc for tc in task.tool_calls
            if tc.result is not None
            and tc.result.execution_failed
            and not tc.result.is_perm_denied
        ]

        if not failed_executions:
            return False  # All tool calls succeeded

        # There are failed executions. Check if there was any subsequent success
        # after the last failure (recovery happened).
        last_failure_idx = -1
        for i, tc in enumerate(task.tool_calls):
            if tc.result and tc.result.execution_failed and not tc.result.is_perm_denied:
                last_failure_idx = i

        # Check if there's a success after the last failure
        has_recovery = False
        if last_failure_idx >= 0:
            for tc in task.tool_calls[last_failure_idx + 1:]:
                if tc.result and tc.result.status == ToolResultStatus.SUCCESS:
                    has_recovery = True
                    break

        if has_recovery:
            return False  # Agent recovered successfully

        # Failed executions exist with no recovery — block completion
        return True

    async def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event stamped with this loop's agent/task/run identity.

        Identity is provided by the WorkerAgent that owns the loop; the
        dashboard uses it to attribute activity truthfully (Phase 10).
        Payloads are sanitized through the semantic contract (no secrets,
        bounded text) before hitting the bus (milestone Part 17).
        """
        payload = semantic.redact_sensitive(dict(data))
        if self.task_id and not payload.get("task_id"):
            payload["task_id"] = self.task_id
        if self.agent_id and not payload.get("agent_id"):
            payload["agent_id"] = self.agent_id
        if self.run_id and not payload.get("run_id"):
            payload["run_id"] = self.run_id
        await self.event_bus.emit(
            Event(type=event_type, source=self.agent_id or "agent_loop", data=payload)
        )

    async def _emit_semantic(self, event_type: str, data: dict[str, Any], quiet: bool = False) -> None:
        """Emit a canonical semantic event via the shared contract."""
        await self.event_bus.emit(
            semantic.make_event(
                event_type,
                self.agent_id or "agent_loop",
                data,
                run_id=self.run_id,
                task_id=self.task_id,
                agent_id=self.agent_id,
                quiet=quiet,
            )
        )

    async def _emit_phase(self, phase: str) -> None:
        """Emit a task phase change event for progress tracking."""
        if phase != self._current_phase:
            self._current_phase = phase
            await self._emit_event("task.phase", {"phase": phase})

    async def _emit_thinking(self, message: str, task: Task | None = None) -> None:
        """Emit a thinking status event (high-level execution intent)."""
        if task:
            task.thinking = message
        await self.event_bus.emit(
            Event(
                type="thinking.status",
                source="agent_loop",
                data={"message": message},
            )
        )

    async def _emit_todo_update(self, task: Task) -> None:
        """Emit current TODO list state (display + structured items)."""
        items = task.task_plan.display()
        completed = task.task_plan.completed_count
        total = task.task_plan.total_count
        await self.event_bus.emit(
            Event(
                type="todo.updated",
                source="agent_loop",
                data={
                    "items": items,
                    "todos": task.task_plan.to_event_items(),
                    "completed": completed,
                    "failed": task.task_plan.failed_count,
                    "total": total,
                },
            )
        )

    async def _emit_todo_event(self, event_type: str, item: TodoItem) -> None:
        """Emit a per-TODO lifecycle event with structured metadata."""
        await self.event_bus.emit(
            Event(
                type=event_type,
                source="agent_loop",
                data={
                    "todo_id": item.id,
                    "title": item.description,
                    "status": item.status.value,
                    "evidence": item.evidence,
                    "error": item.error,
                },
            )
        )

    async def _todo_started(self, task: Task, call: ToolCall) -> None:
        """Mark the best-matching TODO in progress from a real tool start."""
        before = select_todo(task.task_plan, call.tool_name, call.arguments)
        was_pending = before is not None and before.status == TodoStatus.PENDING
        item = apply_tool_started(task.task_plan, call)
        if item is not None and was_pending:
            await self._emit_todo_event("todo.started", item)
            await self._emit_todo_update(task)

    async def _todo_result(self, task: Task, call: ToolCall, result: ToolResult) -> None:
        """Update TODO state from the real tool outcome (evidence-based)."""
        item = apply_tool_result(task.task_plan, call, result)
        if item is None:
            return
        if item.status == TodoStatus.COMPLETED and item.completed_at:
            await self._emit_todo_event("todo.completed", item)
            await self._emit_todo_update(task)
        elif item.status == TodoStatus.FAILED:
            await self._emit_todo_event("todo.failed", item)
            await self._emit_todo_update(task)

    async def _postprocess_result(self, task: Task, call: ToolCall, result: ToolResult) -> None:
        """Enrich a tool result with diagnosis, test accounting, and git state."""
        tool = call.tool_name

        # Command failure diagnosis + test accounting
        if tool == "run_command":
            command = str(call.arguments.get("command", ""))
            if result.execution_failed:
                diagnosis = classify_command_failure(
                    command,
                    exit_code=result.exit_code,
                    stdout=result.output or "",
                    stderr=result.stderr or "",
                    timed_out=(result.status == ToolResultStatus.TIMEOUT),
                )
                result.metadata["diagnosis_category"] = diagnosis.category
                result.metadata["diagnosis_reason"] = diagnosis.reason
                hint = (
                    f"\nDiagnosis: {diagnosis.category} — {diagnosis.reason}"
                )
                if diagnosis.category in ("command_syntax", "missing_executable"):
                    hint += (
                        "\nDo NOT retry with different shell wrappers (cd, cmd /c, powershell). "
                        "Fix the underlying command or use the workspace-aware run_command."
                    )
                if result.error:
                    result.error = result.error + hint
                else:
                    result.error = hint.strip()
            # Test accounting (success or failure) if this ran a suite
            if self._is_test_command(command):
                counts = parse_test_counts(result.output or "")
                if result.status == ToolResultStatus.SUCCESS and counts is None:
                    # Passed but no parseable summary — still a passing run
                    task.tests_run = task.tests_run or 1
                    task.tests_passed = task.tests_passed if task.tests_passed is not None else task.tests_run
                elif counts is not None:
                    passed, total = counts
                    task.tests_run = total
                    task.tests_passed = passed
                await self._emit_event(
                    "test.completed",
                    {
                        "command": command,
                        "passed": task.tests_passed,
                        "total": task.tests_run,
                        "success": result.status == ToolResultStatus.SUCCESS,
                    },
                )

        # Structured git accounting
        elif tool == "git_commit":
            await self._emit_phase("committing")
            if result.status == ToolResultStatus.SUCCESS:
                commit_hash = result.metadata.get("commit_hash") or self._parse_commit_hash(result.output or "")
                if commit_hash:
                    task.git_commit = commit_hash
        elif tool == "git_add" or tool == "git_stage":
            await self._emit_phase("committing")
        elif tool == "git_push":
            await self._emit_phase("pushing")
            if result.status == ToolResultStatus.SUCCESS:
                remote = result.metadata.get("remote", "origin")
                branch = result.metadata.get("branch", "")
                task.git_push = f"{remote}/{branch}" if branch else remote

    @staticmethod
    def _parse_commit_hash(output: str) -> str:
        """Extract a short commit hash from `git commit` output."""
        import re
        m = re.search(r"\[([^\s\]]+)\s+([0-9a-f]{7,40})\]", output)
        if m:
            return m.group(2)
        m = re.search(r"\b([0-9a-f]{40})\b", output)
        return m.group(1)[:12] if m else ""

    async def _stagnation_recovery(self, task: Task) -> bool:
        """Stagnation stopped the loop — complete truthfully if evidence is green.

        If files were changed, the last test run (if any) succeeded, and
        verification passes, the work is done even though the model kept
        issuing no-progress actions. Otherwise the task genuinely failed.
        The completion invariant is the gate: can_complete_task() must pass.
        """
        if not self._modified_files:
            return False
        for tc in reversed(task.tool_calls):
            if tc.tool_name == "run_command" and self._is_test_command(
                str(tc.arguments.get("command", ""))
            ):
                if tc.result is not None and tc.result.execution_failed:
                    return False
                break
        await self._verify_task_completion(task)
        if task.status == TaskStatus.FAILED:
            return False
        await self._reconcile_todos(task)

        # Completion invariant: runtime truth, not stagnation mercy
        if not can_complete_task(task):
            blockers = completion_blockers(task)
            task.status = TaskStatus.FAILED
            task.error = (
                "Stopped repeating no-progress iterations but "
                + "; ".join(blockers)
            )
            return False

        task.status = TaskStatus.COMPLETED
        task.error = None
        task.result = (
            "Stopped repeating no-progress iterations; final state verified. "
            + (task.result or "")
        )
        await self._emit_phase("complete")
        return True

    async def _reconcile_todos(self, task: Task) -> None:
        """Complete/fail open TODOs using real execution evidence."""
        did_inspect = any(
            tc.tool_name in ("read_file", "list_files", "glob", "grep")
            and tc.result and tc.result.status == ToolResultStatus.SUCCESS
            for tc in task.tool_calls
        )
        did_implement = bool(self._modified_files)
        tests_passed: bool | None = None
        if task.tests_run > 0 and task.tests_passed is not None:
            tests_passed = task.tests_passed >= task.tests_run
        did_commit = bool(task.git_commit)
        did_push = bool(task.git_push)

        changed = reconcile_on_evidence(
            task.task_plan,
            did_inspect=did_inspect,
            did_implement=did_implement,
            tests_passed=tests_passed,
            verified=task.verification_passed,
            did_commit=did_commit,
            did_push=did_push,
            tests_run=task.tests_run,
        )
        if changed:
            for item in changed:
                event_type = (
                    "todo.completed" if item.status == TodoStatus.COMPLETED else "todo.failed"
                )
                await self._emit_todo_event(event_type, item)
            await self._emit_todo_update(task)

    def _get_failure_summary(self, task: Task) -> str:
        """Build a summary of failed tool calls for the task error message."""
        failures = []
        for tc in task.tool_calls:
            if tc.result and tc.result.execution_failed and not tc.result.is_perm_denied:
                cmd = tc.arguments.get("command", tc.tool_name)
                exit_code = tc.result.exit_code
                stderr = tc.result.stderr
                parts = [f"Command: {cmd}"]
                if exit_code is not None:
                    parts.append(f"Exit code: {exit_code}")
                if stderr:
                    # Truncate long stderr
                    truncated = stderr[:500] + ("..." if len(stderr) > 500 else "")
                    parts.append(f"stderr: {truncated}")
                failures.append("\n  ".join(parts))
        return "\n\nFailed commands:\n" + "\n---\n".join(failures) if failures else ""

    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a tool call with permission and governor checking.

        Guarantees call.result is populated on every path (including
        denials and blocked retries) so accounting stays truthful.
        """
        result = await self._execute_tool_checked(call)
        call.result = result
        return result

    # File tools whose "path" argument must stay inside the workspace.
    _PATH_TOOLS_REQUIRED = {"read_file", "write_file", "edit_file"}
    _PATH_TOOLS_OPTIONAL = {"list_files", "glob", "grep"}
    _CWD_TOOLS = {"run_command", "git_status", "git_diff", "git_log", "git_add",
                  "git_commit", "git_push", "git_remote", "git_identity"}

    def _confine_call(self, call: ToolCall) -> ToolResult | None:
        """Enforce the workspace boundary on paths and working directory.

        Returns a denial ToolResult if a path escapes the workspace, else None.
        """
        name = call.tool_name
        args = call.arguments

        if name in self._PATH_TOOLS_REQUIRED or name in self._PATH_TOOLS_OPTIONAL:
            raw = args.get("path") or args.get("file_path")
            if raw or name in self._PATH_TOOLS_REQUIRED:
                resolved, err = resolve_in_workspace(self.workspace_root, raw)
                if err is not None:
                    return ToolResults.permission_denied(f"Path confinement: {err}")
                if resolved is not None:
                    if "path" in args or name in self._PATH_TOOLS_REQUIRED:
                        args["path"] = str(resolved)
                    if "file_path" in args:
                        args["file_path"] = str(resolved)

        if name in self._CWD_TOOLS:
            cwd, _ = cwd_in_workspace(self.workspace_root, args.get("cwd"))
            args["cwd"] = cwd

        return None

    async def _execute_tool_checked(self, call: ToolCall) -> ToolResult:
        """Execute a tool call with permission checking.

        Added validation to ensure required arguments are present and to
        provide structured errors for missing arguments, improving reliability.
        """
        tool = self.tools.get(call.tool_name)
        if not tool:
            self._reset_denial_tracking()
            return ToolResults.unknown_tool(call.tool_name)

        # Validate required arguments up front so the model receives a
        # structured error instead of a tool-level exception.
        required_args = tool.schema.parameters.get("required", [])
        missing = [arg for arg in required_args if arg not in call.arguments]
        if missing:
            # Record a failure to enforce bounded correction attempts.
            self._record_failure(call.tool_name, call.arguments, exit_code=-1)
            return ToolResults.missing_argument(call.tool_name, missing)
        # Hard workspace boundary: confine file paths and working directory.
        confinement_denial = self._confine_call(call)
        if confinement_denial is not None:
            self._record_denial(call.tool_name, call.arguments)
            return confinement_denial

        # Check if this exact call was already denied
        if self._check_repeated_deny(call):
            self._consecutive_denials += 1
            return ToolResults.permission_denied(
                "Permission denied (already rejected). "
                "This command cannot be retried under the current policy. "
                "Ask for user approval or choose a different approach."
            )

        # Check permission
        permission = self.permission_manager.check_permission(call.tool_name, call.arguments)
        if permission == "deny":
            self._record_denial(call.tool_name, call.arguments)
            return ToolResults.permission_denied("Permission denied by policy")
        if permission == "ask" and not self.permission_manager.request_approval(
            call.tool_name, str(call.arguments)
        ):
            self._record_denial(call.tool_name, call.arguments)
            return ToolResults.permission_denied(
                "Permission denied. This command requires approval. "
                "Ask the user for permission or use a different approach."
            )

        # Check for repeated failures of the same operation
        if self._is_repeating_failure(call):
            key = self._action_key(call.tool_name, call.arguments)
            count = self._failure_counts.get(key, 0)
            return ToolResults.error(
                f"BLOCKED: this exact operation has already failed {count} times. "
                f"Running it again without changes is not allowed. "
                f"Diagnose the root cause (read the relevant files and error output), "
                f"apply a fix to the implementation, then retry.",
                metadata={"operation": key, "failure_count": count, "blocked": True},
                retryable=False,
            )

        # Phase 8: skip operations that already succeeded (duplicate prevention).
        # Tools whose results are idempotent: git_status, git_remote, git_log,
        # git_diff, read_file, list_files, glob. Other tools always run.
        op_key = self._action_key(call.tool_name, call.arguments)
        if (
            call.tool_name
            in {
                "git_status", "git_remote", "git_log", "git_diff",
                "read_file", "list_files", "glob",
            }
            and op_key in self._completed_operations
        ):
            # Return the cached "already done" result without re-executing.
            return ToolResults.success(
                "(already executed — cached)",
                metadata={"cached": True, "operation": op_key},
            )

        # Execute with a bounded timeout so a hung tool can't freeze the CLI.
        call_timeout = float(call.arguments.get("timeout") or tool.schema.timeout_seconds or 30.0)
        call_timeout = min(max(call_timeout, 1.0), 300.0)
        start = time.time()
        try:
            result = await asyncio.wait_for(tool.execute(call.arguments), timeout=call_timeout)
            call.duration_ms = (time.time() - start) * 1000
            call.result = result
            self._reset_denial_tracking()
            if result.execution_failed:
                self._record_failure(
                    call.tool_name, call.arguments, result.exit_code or -1
                )
            else:
                # Record successful operation for Phase 8 duplicate prevention
                self._completed_operations.add(op_key)
                self._reset_failure_tracking()
            await self._record_execution_outcome(call, result)
            return result
        except TimeoutError:
            call.duration_ms = (time.time() - start) * 1000
            self._reset_denial_tracking()
            self._record_failure(call.tool_name, call.arguments, -1)
            timeout_result = ToolResults.timeout(
                f"Command timed out after {call_timeout:.0f}s",
                timeout_seconds=call_timeout,
            )
            call.result = timeout_result
            await self._record_execution_outcome(call, timeout_result)
            return timeout_result
        except Exception as e:
            call.duration_ms = (time.time() - start) * 1000
            self._reset_denial_tracking()
            self._record_failure(call.tool_name, call.arguments, -1)
            # Unexpected exceptions at the loop level stay retryable: the loop
            # cannot tell a transient fault from a deterministic bug, and the
            # repeated-failure guard above bounds how often we retry.
            error_result = ToolResults.from_exception(e, retryable=True)
            call.result = error_result
            await self._record_execution_outcome(call, error_result)
            return error_result

    # ── Steering / ask-user runtime API (Parts 7, 14) ─────────────────

    async def _enqueue_steering(self, text: str) -> Any:
        """Queue a steering message WITHOUT emitting steering.received.

        Used by the event bridge — the original event already announced the
        steering to consumers, so re-emitting would recurse forever.
        """
        msg = await self.steering.submit_steering_message(text)
        msg.task_id = self.task_id or (self._active_task.id if self._active_task else "")
        return msg

    async def submit_steering_message(self, text: str) -> Any:
        """Queue a mid-run user instruction; applied at the next safe boundary."""
        msg = await self._enqueue_steering(text)
        await self._emit_semantic(
            semantic.STEERING_RECEIVED,
            steering_event_data(msg, applied=False),
        )
        return msg

    async def drain_steering_messages(self) -> list[Any]:
        """Drain pending steering messages (used by tests and the UI layer)."""
        return await self.steering.drain_steering_messages()

    async def has_pending_steering(self) -> bool:
        return await self.steering.has_pending_steering()

    async def ask_user_question(
        self,
        question: str,
        choices: list[str] | None = None,
        explanation: str = "",
        timeout_seconds: float | None = None,
    ) -> Any:
        """Ask the user a question and pause until answered/cancelled.

        Backend contract (Part 14): emits agent.question_requested, waits,
        emits agent.question_answered. On timeout/cancel the loop continues
        gracefully with no answer rather than hanging or failing.
        """
        q = await self.ask_user.ask(question, choices=choices, explanation=explanation)
        q.run_id = self.run_id
        q.task_id = self.task_id or (self._active_task.id if self._active_task else "")
        self.run_metrics.record_ask_user()
        await self._emit_semantic(
            semantic.AGENT_QUESTION_REQUESTED,
            q.to_event_payload(),
        )
        result = await self.ask_user.wait_for_answer(q, timeout_seconds=timeout_seconds)
        await self._emit_semantic(
            semantic.AGENT_QUESTION_ANSWERED,
            question_answered_payload(result),
        )
        self.ask_user.cleanup(q.id)
        return result

    def cancel(self, reason: str = "User cancelled") -> None:
        """Cooperative cancellation flag checked at the iteration boundary."""
        self._cancelled = True
        self.steering.cancel()

    # ── Context pipeline integration (Parts 2-6) ──────────────────────

    async def _discover_context(self, goal: str) -> tuple[list[Any], Any | None]:
        """Run the context intelligence pipeline for the task.

        Returns (context pieces, ContextSelection | None). Deterministic
        discovery always runs; the AI relevance stage is optional inside the
        pipeline and falls back cleanly. Emits the canonical context.*
        events. Any pipeline failure degrades to the legacy assemble_context
        path — context discovery must never fail the run.
        """
        await self._emit_semantic(
            semantic.CONTEXT_DISCOVERY_STARTED,
            {"task": goal[:200]},
        )
        self.run_metrics.stage_start("discovery")
        selection = None
        context: list[Any] = []
        try:
            request = ContextRequest(
                task=goal,
                run_id=self.run_id,
                task_id=self.task_id,
                agent_id=self.agent_id,
            )
            selection = await self._context_pipeline.discover(
                request,
                project_files=(self._project_info or {}).get("files", []),
            )
            context = selection.to_context_pieces()
        except Exception:
            # Deterministic degradation: legacy context assembly still runs.
            selection = None
        if selection is None or not context:
            context = await self.context_engine.assemble_context(goal, self._project_info)
        self._last_context_selection = selection

        # Accounting + events.
        ctx_tokens = 0
        if selection is not None:
            ctx_tokens = selection.snapshot.tokens_used
            self.run_metrics.record_context(
                ctx_tokens, files=selection.snapshot.file_paths()
            )
            ai = selection.snapshot.ai_ranking_used
            failed = selection.snapshot.ai_ranking_failed
            self.run_metrics.record_ai_relevance(used=ai, failed=failed, skipped=not (ai or failed))
            for cand in selection.snapshot.selected:
                await self._emit_semantic(
                    semantic.CONTEXT_FILE_SELECTED,
                    {"path": cand.path, "score": cand.score, "freshness": cand.freshness.value},
                    quiet=True,
                )
        self.run_metrics.stage_end("discovery")
        await self._emit_semantic(
            semantic.CONTEXT_DISCOVERY_COMPLETED,
            {
                "files": selection.snapshot.file_paths() if selection else [],
                "tokens": ctx_tokens,
                "candidates": selection.snapshot.candidates_considered if selection else 0,
                "ai_ranking_used": bool(selection and selection.snapshot.ai_ranking_used),
                "ai_ranking_failed": bool(selection and selection.snapshot.ai_ranking_failed),
            },
        )
        return context, selection

    async def _maybe_compact(self, task: Task) -> bool:
        """Compact conversation context when the budget demands it.

        Uses the existing ContextCompactor (deterministic) with an optional
        model-assisted summary via ModelSummarizer. Emits
        context.compaction_started/completed. Never fails the run.
        """
        from harness_core.context.budgets import BudgetClass, ContextBudgetManager

        manager = getattr(self, "_ctx_budget_manager", None)
        if manager is None:
            manager = ContextBudgetManager()
            self._ctx_budget_manager = manager
        manager.reset()
        # Account current assembled history by class.
        for tc in task.tool_calls:
            if tc.result is None:
                continue
            raw = (tc.result.output or "") + (tc.result.error or "")
            manager.record(BudgetClass.TOOL_OUTPUT, estimate_tokens(raw))
        if task.goal:
            manager.record(BudgetClass.USER_CRITICAL, estimate_tokens(task.goal))
        if task.result:
            manager.record(BudgetClass.ASSISTANT, estimate_tokens(task.result))

        if not manager.compaction_recommended:
            return False

        await self._emit_semantic(semantic.CONTEXT_COMPACTION_STARTED, manager.status())
        self.run_metrics.stage_start("compaction")

        # Compact older tool history via the existing deterministic compactor.
        calls_with_results = [tc for tc in task.tool_calls if tc.result]
        preserve = 10
        older = calls_with_results[:-preserve] if len(calls_with_results) > preserve else []
        summary_text = ""
        if older:
            agent_msgs = self._as_agent_messages(older)
            # Optional cheap-model summary; None → deterministic path.
            from harness_core.context.summarizer import ModelSummarizer

            summarizer = ModelSummarizer(
                model=getattr(self.router, "_summary_model", None) or self.provider
                if self.router is None else getattr(self.router, "_summary_model", None),
            )
            model_summary = await summarizer.summarize(agent_msgs, protected_intent=task.goal)
            deterministic = ContextCompactor(preserve_recent=0).compact(agent_msgs)
            summary_text = model_summary or deterministic.content
            task.compaction_note = (
                f"[compacted {len(older)} older tool call(s) before iteration "
                f"{task.iterations + 1}]"
            )
            task.tool_calls = list(calls_with_results[-preserve:])

        saved = sum(
            estimate_tokens((tc.result.output or "") + (tc.result.error or ""))
            for tc in older
        )
        self.run_metrics.record_compaction(saved)
        self.run_metrics.stage_end("compaction")
        status = manager.status()
        status["tokens_saved"] = saved
        status["compacted_calls"] = len(older)
        status["summary_mode"] = "model" if summary_text and not summary_text.startswith("Tool calls made") else "deterministic"
        await self._emit_semantic(semantic.CONTEXT_COMPACTION_COMPLETED, status)
        return True

    async def run(self, goal: str) -> Task:
        """Run the agent loop for a given goal."""
        task = Task(goal=goal, max_iterations=self.config.max_iterations)
        self._active_task = task
        self._cancelled = False
        self.steering.reset()
        self.run_metrics = RunMetrics(run_id=self.run_id)

        # Canonical run lifecycle (Part 8): run.started first.
        await self._emit_semantic(
            semantic.RUN_STARTED,
            {"goal": goal[:200], "run_id": self.run_id},
        )

        # Reset per-task governor state (loops may be reused across tasks)
        self._corrections = []
        self._seen_actions = set()
        self._failure_counts = {}
        self._stagnation_counter = 0
        self._diagnosis_active = False
        self._diagnosis_iterations = 0
        self._no_tool_nudges = 0
        self._modified_files = []
        self._had_test_failure = False
        self._last_model_used = ""
        self._current_phase = ""
        self._recent_denials = []
        self._consecutive_denials = 0
        self._recent_failures = []
        self._consecutive_failures = 0
        self._completed_operations = set()
        self._pending_workflow_events = []
        self.budget.reset()

        await self.event_bus.emit(
            Event(type="task.started", source="agent_loop", data={"goal": goal})
        )
        await self._emit_phase("understanding")

        # Intent detection event (canonical intent.detected) — deterministic
        # classification reused from the existing intent module.
        try:
            from harness_core.agent.intent import classify_intent

            intent = classify_intent(goal)
            await self._emit_semantic(
                semantic.INTENT_DETECTED,
                {"intent": "read_only" if intent.read_only else "engineering", "goal": goal[:120]},
            )
        except Exception:
            pass

        # Phase 16: intent fast paths — deterministic workflows run without
        # unbounded LLM iteration when the intent matches.
        from harness_core.agent.workflows import (
            classify_workflow,
            run_explain_workflow,
            run_git_push_workflow,
        )
        workflow_name = classify_workflow(goal)
        if workflow_name in ("git_push", "explain", "test"):
            await self._emit_phase("workflow")
            if workflow_name == "git_push":
                wf_result = await run_git_push_workflow(
                    ctx=self._workflow_context(),
                    plan=task.task_plan,
                )
            elif workflow_name == "test":
                from harness_core.agent.workflows import run_test_workflow
                wf_result = await run_test_workflow(
                    ctx=self._workflow_context(),
                    plan=task.task_plan,
                )
            else:  # explain
                wf_result = await run_explain_workflow(
                    ctx=self._workflow_context(),
                    plan=task.task_plan,
                )
            self._apply_workflow_result(task, wf_result)
            await self._flush_workflow_events()
            if wf_result.success:
                # Completion invariant: workflow success ≠ task COMPLETED.
                # Every workflow must pass can_complete_task before finishing.
                await self._reconcile_todos(task)
                if can_complete_task(task):
                    if workflow_name != "explain":
                        await self._emit_phase("complete")
                        task.status = TaskStatus.COMPLETED
                else:
                    blockers = completion_blockers(task)
                    failed_required = [
                        i for i in task.task_plan.items
                        if i.required and i.status == TodoStatus.FAILED
                    ]
                    if failed_required or task.execution_stats.has_unresolved_failures:
                        task.status = TaskStatus.FAILED
                        task.failure_reason = "required_work_failed"
                    else:
                        # Required work merely did not happen: PARTIAL, never
                        # COMPLETE and never a fabricated FAILURE.
                        task.status = TaskStatus.PARTIAL
                        task.failure_reason = "required_work_incomplete"
                    task.error = (
                        "Workflow ended with required work unresolved: "
                        + "; ".join(blockers)
                    )
            else:
                await self._emit_phase("complete")
                task.status = TaskStatus.FAILED
                task.error = wf_result.failure_reason
                task.failure_reason = wf_result.failure_reason
                
            # Emit final event and return ONLY if we are fully done.
            # Explain falls through to the LLM loop so it can summarize the gathered context.
            if workflow_name != "explain" or task.status == TaskStatus.FAILED:
                await self.event_bus.emit(
                    Event(
                        type="task.completed",
                        source="agent_loop",
                        data={
                            "task_id": task.id,
                            "status": task.status.value,
                            "failure_reason": task.failure_reason,
                            "iterations": task.iterations,
                            "tool_calls": len(task.tool_calls),
                            "stats": task.execution_stats.summary(),
                            "attempted": task.execution_stats.attempted,
                            "succeeded": task.execution_stats.succeeded,
                            "failed": task.execution_stats.failed,
                            "recovered": task.execution_stats.recovered,
                            "unresolved": task.execution_stats.unresolved,
                            "verification_passed": task.verification_passed,
                            "verification_summary": task.verification_summary,
                            "files_changed": list(self._modified_files),
                            "completed_operations": list(self._completed_operations),
                            "todos": task.task_plan.to_event_items(),
                            "todos_completed": task.task_plan.completed_count,
                            "todos_failed": task.task_plan.failed_count,
                            "todos_total": task.task_plan.total_count,
                            "tests_run": task.tests_run,
                            "tests_passed": task.tests_passed,
                            "models_used": list(task.models_used),
                            "model_fallbacks": task.model_fallbacks,
                            "git_commit": task.git_commit,
                            "git_push": task.git_push,
                            "paused_reason": task.paused_reason,
                        },
                    )
                )
                return task

        # Classify task if task_aware router is available
        task_type = None
        task_profile = None
        classification_confidence = 0.0
        if self.task_aware is not None:
            # Build a minimal request for classification
            classify_request = CompletionRequest(messages=[{"role": "user", "content": goal}])
            task_type, task_profile, classification_confidence = self.task_aware.classify_task(classify_request)
            await self.event_bus.emit(
                Event(
                    type="task.classified",
                    source="agent_loop",
                    data={
                        "task_type": task_type.value if task_type else "unknown",
                        "confidence": classification_confidence,
                    },
                )
            )

        # Discover project — workspace intelligence before planning
        project_info = await self.context_engine.discover_project()
        self._project_info = project_info

        # Harness 2.0: intelligent context pipeline (deterministic + optional
        # AI relevance + reuse integration). Falls back to legacy assembly.
        context, _selection = await self._discover_context(goal)
        
        # Inject workflow-gathered context if any (e.g., from explain workflow)
        if hasattr(task, "workflow_context") and task.workflow_context:
            context.extend(task.workflow_context)

        # Planning phase: ask model to create a concise plan grounded in
        # the discovered workspace (never in imagined context).
        await self._emit_phase("planning")
        try:
            plan_user_content = goal
            snapshot = self._workspace_snapshot_text()
            if snapshot:
                plan_user_content += (
                    "\n\nThe workspace has already been discovered:\n"
                    f"{snapshot}\n"
                    "Base the plan on these real files. Do not ask the user for "
                    "information that is discoverable with tools."
                )
            plan_messages = [
                {"role": "system", "content": (
                    "You are planning an engineering task. Respond with a numbered list of steps.\n"
                    "Be concise: 3-6 steps maximum. Each step must be one short, EXECUTABLE line\n"
                    "(an action the agent will perform), e.g. 'Inspect calculator.js'.\n"
                    "No questions. No explanations. No requests for more information.\n"
                    "Example:\n"
                    "1. Inspect current code\n"
                    "2. Implement changes\n"
                    "3. Run tests\n"
                    "4. Verify results"
                )},
                {"role": "user", "content": plan_user_content},
            ]
            plan_request = CompletionRequest(messages=plan_messages)
            if self.router is not None:
                plan_result = await self.router.execute(
                    plan_request,
                    routing_mode_override=self.config.routing_mode
                )
                if plan_result.succeeded and plan_result.response:
                    plan_text = plan_result.response.content or ""
                else:
                    plan_text = ""
            else:
                plan_response = await self.provider.generate(plan_request)
                plan_text = plan_response.content or ""
            # Parse plan steps from response
            plan_steps = []
            for line in plan_text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Remove numbering prefixes like "1.", "1)", "- "
                import re
                cleaned = re.sub(r"^[\d]+[.)\s]+[-•*]?\s*", "", line).strip()
                if cleaned:
                    plan_steps.append(cleaned)
            # Validate: TODOs must be actionable engineering tasks (verb-first).
            # Conversational prose is rejected and never displayed.
            plan_steps = self._validate_plan_steps(plan_steps)
            if not plan_steps:
                plan_steps = self._default_plan_steps(goal)

            # INTENT GUARD (Phase 10.6): a read-only request ("explain this
            # project") must never acquire REQUIRED modification/test/fix
            # tasks — even when the model's proposed plan over-decomposes.
            # Only inspection/analysis steps survive; the plan cannot mutate
            # the repository.
            from harness_core.agent.intent import classify_intent, is_read_only_verb
            from harness_core.agent.todos import todo_category
            intent = classify_intent(goal)
            if intent.read_only:
                plan_steps = [
                    s for s in plan_steps
                    if todo_category(s) == "inspect" or is_read_only_verb(s)
                ]
                if not plan_steps:
                    plan_steps = [
                        "Discover workspace structure",
                        "Read key project files",
                        "Summarize findings",
                    ]
            if plan_steps:
                task.plan = plan_steps
                # Create dynamic task plan (runtime owns status from here on)
                for step in plan_steps:
                    task.task_plan.add(step)
                await self.event_bus.emit(
                    Event(
                        type="plan.created",
                        source="agent_loop",
                        data={
                            "steps": plan_steps,
                            "task_id": task.id,
                            "todos": task.task_plan.to_event_items(),
                        },
                    )
                )
                await self._emit_todo_update(task)
        except Exception:
            pass  # Planning is best-effort; don't fail the task

        # Emit initial thinking
        await self._emit_thinking("I understand the task. Starting execution.", task)
        await self._emit_phase("implementing")

        while task.iterations < task.max_iterations:
            # ── Safe step boundary (Part 7) ────────────────────────────
            # Cooperative cancellation check before re-arming a request.
            if self._cancelled or self.steering.cancelled:
                task.status = TaskStatus.CANCELLED
                task.error = task.error or "Cancelled by user"
                await self._emit_semantic(
                    semantic.RUN_CANCELLED, {"reason": "user_requested"}
                )
                break

            task.iterations += 1
            task.status = TaskStatus.EXECUTING
            self.budget.record_iteration()

            await self._emit_event(
                "iteration.started", {"iteration": task.iterations, "task_id": task.id}
            )

            # Steering injection at the safe step boundary: queued user
            # messages become corrections for the next request. The in-flight
            # provider request is never mutated; FIFO order is preserved.
            if self.steering.has_pending_steering_sync():
                pending = await self.steering.drain_steering_messages()
                for steering_msg in pending:
                    # Stamp run identity at drain time — the submitter may
                    # predate the run (message queued before run started).
                    steering_msg.run_id = self.run_id
                    steering_msg.task_id = (
                        self.task_id or (self._active_task.id if self._active_task else "")
                    )
                    self._corrections.append(steering_msg.render())
                    await self._emit_semantic(
                        semantic.STEERING_APPLIED,
                        steering_event_data(steering_msg, applied=True),
                    )

            # Budget-driven compaction (Parts 4/5) at the boundary.
            try:
                await self._maybe_compact(task)
            except Exception:
                pass  # compaction must never fail the run

            # Build messages
            messages = self._build_messages(task, context)

            # Check budget
            ok, reason = self.budget.check_all()
            if not ok:
                task.status = TaskStatus.FAILED
                task.error = f"Budget exceeded: {reason}"
                break

            # Call model — via router if available, else direct provider
            request = CompletionRequest(
                messages=messages,
                model=self.config.model_preference,
                tools=self._tool_schemas() if self.tools else None,
            )

            self.run_metrics.record_model_call()
            await self._emit_semantic(
                semantic.MODEL_STARTED,
                {"model": self.config.model_preference or "auto", "iteration": task.iterations},
            )
            try:
                if self.router is not None:
                    response = None
                    for attempt in range(2):
                        fallback_result = await self.router.execute(
                            request,
                            routing_mode_override=self.config.routing_mode
                        )
                        if fallback_result.succeeded:
                            response = fallback_result.response
                            prev_model = self._last_model_used
                            self._last_model_used = fallback_result.model_used
                            # Account models + switches (quiet, change-only)
                            if fallback_result.model_used and fallback_result.model_used not in task.models_used:
                                task.models_used.append(fallback_result.model_used)
                            if prev_model and fallback_result.model_used and fallback_result.model_used != prev_model:
                                task.model_fallbacks += 1
                                await self.event_bus.emit(
                                    Event(
                                        type="model.switched",
                                        source="agent_loop",
                                        data={
                                            "from": prev_model,
                                            "to": fallback_result.model_used,
                                            "reason": "unavailable",
                                        },
                                    )
                                )
                            break
                        if attempt == 0:
                            # Failed models were marked unhealthy (402/401/429);
                            # a rebuilt chain skips them and reaches viable models.
                            await self.event_bus.emit(
                                Event(
                                    type="model.error",
                                    source="agent_loop",
                                    data={
                                        "error": fallback_result.final_error or "All models failed",
                                        "retrying": True,
                                    },
                                )
                            )
                            continue
                        raise RuntimeError(fallback_result.final_error or "All models failed")
                else:
                    response = await self.provider.generate(request)
            except Exception as e:
                err_str = str(e)
                failure_reason = self._classify_failure_reason(err_str)
                task.failure_reason = failure_reason
                await self.event_bus.emit(
                    Event(
                        type="model.error",
                        source="agent_loop",
                        data={
                            "error": err_str,
                            "reason": failure_reason,
                        },
                    )
                )
                # Phase 11: transient model failures after real progress
                # should PAUSE (state preserved, resumable), not FAIL.
                # "Real progress" = files modified or actual tool calls
                # executed (not just auto-completed read-only planning TODOs).
                has_progress = (
                    bool(self._modified_files)
                    or len(task.tool_calls) > 0
                )
                transient = failure_reason in (
                    "model_unavailable", "model_rate_limited",
                    "provider_auth_failure", "payment_required",
                )
                if transient and has_progress:
                    task.status = TaskStatus.PAUSED
                    task.paused_reason = _PAUSED_MESSAGES.get(
                        failure_reason, f"Model error: {failure_reason}"
                    )
                    task.error = task.paused_reason
                    await self.event_bus.emit(
                        Event(
                            type="task.paused",
                            source="agent_loop",
                            data={
                                "reason": failure_reason,
                                "error": err_str,
                            },
                        )
                    )
                else:
                    task.status = TaskStatus.FAILED
                    task.error = f"Provider error: {e}"
                break

            # Process response
            if response.content:
                await self._emit_thinking(response.content, task)
                if not task.result:
                    task.result = response.content
                else:
                    task.result += "\n\n" + response.content

            iter_calls: list[ToolCall] = []
            if response.tool_calls:
                # Execute tool calls
                for tool_call_data in response.tool_calls:
                    func = tool_call_data.get("function", {})
                    call = ToolCall(
                        id=tool_call_data.get("id", ""),
                        tool_name=func.get("name", ""),
                        arguments=json.loads(func.get("arguments", "{}")),
                    )
                    task.tool_calls.append(call)
                    iter_calls.append(call)
                    self.budget.record_tool_call()

                    quiet = call.tool_name in _READONLY_TOOLS
                    if quiet:
                        self.run_metrics.record_tool_call(quiet=True)
                    await self._emit_event(
                        "tool.call", {"tool": call.tool_name, "args": call.arguments, "quiet": quiet}
                    )
                    # Canonical tool lifecycle (Part 10): tool.started carries a
                    # safe summary, not raw arguments; ghost/read-only tools are
                    # marked quiet so the UI can collapse them (Part 13).
                    await self._emit_semantic(
                        semantic.TOOL_STARTED,
                        semantic.tool_lifecycle_payload(
                            tool_name=call.tool_name,
                            call_id=call.id,
                            status="started",
                            summary=self._tool_summary(call),
                            affected_files=[
                                str(v) for k, v in call.arguments.items()
                                if k in ("path", "file_path") and isinstance(v, str)
                            ],
                        ),
                        quiet=quiet,
                    )

                    # Runtime-owned TODO state: mark matching item in progress
                    await self._todo_started(task, call)

                    result = await self._execute_tool(call)

                    # Enrich with diagnosis / test accounting / git state
                    await self._postprocess_result(task, call, result)
                    # Evidence-based TODO transitions (never from model prose)
                    await self._todo_result(task, call, result)

                    # Track execution stats
                    task.execution_stats.record_attempt()
                    if result.status == ToolResultStatus.SUCCESS:
                        task.execution_stats.record_success(call.tool_name)
                    elif result.status == ToolResultStatus.PERMISSION_DENIED:
                        task.execution_stats.record_permission_denied(call.tool_name)
                    else:
                        task.execution_stats.record_failure(call.tool_name)

                    event_data: dict[str, Any] = {
                        "tool": call.tool_name,
                        "status": result.status.value,
                        "output_len": len(result.output),
                    }
                    if call.tool_name == "run_command":
                        # Phase 10.5: the exact command rides on the result
                        # event for the convergence governor's repeated-command
                        # detection.
                        event_data["command"] = str(call.arguments.get("command", ""))
                    if result.exit_code is not None:
                        event_data["exit_code"] = result.exit_code
                    if result.error:
                        event_data["error"] = result.error
                    if result.stderr:
                        event_data["stderr"] = result.stderr
                    # Phase 10.5: structured git metadata (commit_hash, remote,
                    # branch, ...) travels on the event so dashboards can show
                    # evidence-backed Git state instead of model prose.
                    if result.metadata:
                        event_data["metadata"] = dict(result.metadata)

                    await self._emit_event("tool.result", event_data)
                    # Canonical tool lifecycle (Part 10): completed/failed with
                    # bounded metadata — no raw output dumps in the event stream.
                    _failed = result.execution_failed
                    await self._emit_semantic(
                        semantic.TOOL_FAILED if _failed else semantic.TOOL_COMPLETED,
                        semantic.tool_lifecycle_payload(
                            tool_name=call.tool_name,
                            call_id=call.id,
                            status="failed" if _failed else "completed",
                            summary=self._tool_summary(call),
                            duration_ms=call.duration_ms,
                            error_class=(
                                result.failure_category if _failed else ""
                            ),
                            affected_files=[
                                str(v) for k, v in call.arguments.items()
                                if k in ("path", "file_path") and isinstance(v, str)
                            ],
                            extra={
                                "exit_code": result.exit_code,
                                "error": (result.error or "")[:300] if _failed else None,
                            },
                        ),
                        quiet=quiet,
                    )

                    # Check if we've hit tool call limit
                    if len(task.tool_calls) >= self.config.max_tool_calls:
                        task.status = TaskStatus.FAILED
                        task.error = "Tool call limit reached"
                        break
            else:
                # No tool calls — model wants to finish.
                # GUARD: a workspace task must not complete with zero tool calls.
                # Models that claim "I don't have visibility" are wrong — the
                # workspace is real and accessible through tools.
                if (
                    not task.tool_calls
                    and self.tools
                    and self._workspace_has_files()
                    and self._no_tool_nudges < MAX_NO_TOOL_NUDGES
                ):
                    self._no_tool_nudges += 1
                    files_preview = ", ".join(
                        (self._project_info or {}).get("files", [])[:10]
                    )
                    # Rotate away from the model that ignored its tools, so the
                    # next iteration tries a different (hopefully tool-capable) model.
                    if self.router is not None and self._last_model_used:
                        self.router.health.record_no_tool_usage(self._last_model_used)
                    self._corrections.append(
                        "You responded without using any tools. That is not acceptable here: "
                        f"the workspace at {self._project_info.get('root')} is real and "
                        f"contains these files: {files_preview}. "
                        "Use your tools (read_file, list_files, grep, run_command, ...) to "
                        "inspect the project and do the work. Do NOT claim you lack "
                        "visibility — proceed with the task using tools now."
                    )
                    await self.event_bus.emit(
                        Event(
                            type="execution.nudge",
                            source="agent_loop",
                            data={"reason": "zero_tool_calls", "nudge": self._no_tool_nudges},
                        )
                    )
                    continue

                # When the model finishes with text only (no tool calls in this
                # iteration), the remaining pending TODOs are evaluated:
                #
                # - OPTIONAL TODOs: always skipped (model finished, optional work
                #   not performed).
                # - REQUIRED TODOs with no tool surface / empty workspace:
                #   skipped with authorization (work was physically impossible).
                # - REQUIRED TODOs when nudge budget is exhausted: NOT skipped.
                #   The completion invariant below correctly classifies this as
                #   PARTIAL — "model stopped responding" is not the same as
                #   "work is done". Required work that did not happen must
                #   remain PENDING so the user sees the honest gap.
                # - All other required TODOs: NOT skipped; the completion
                #   invariant below decides whether PARTIAL or FAILED is honest.
                if not iter_calls:
                    cannot_work = (
                        not self.tools or not self._workspace_has_files()
                    )
                    for item in task.task_plan.items:
                        if item.status not in (
                            TodoStatus.PENDING, TodoStatus.IN_PROGRESS
                        ):
                            continue
                        # Nudge exhaustion does NOT authorize skipping required
                        # work — only the absence of a tool surface does.
                        if item.required and not cannot_work:
                            continue  # keep required work pending for invariant check
                        if cannot_work and item.required:
                            skip_reason = "No tool surface / empty workspace; work not performable"
                        else:
                            skip_reason = "Optional work not performed; model finished with text response"
                        task.task_plan.skip_id(
                            item.id,
                            skip_reason,
                            authorized=cannot_work,
                        )

                # HARD INVARIANT: TOOL FAILURE ≠ TASK SUCCESS
                # The runtime execution results are the source of truth.
                # A model text response claiming success does NOT override
                # actual tool failures.
                if self._should_block_completion(task):
                    task.status = TaskStatus.FAILED
                    failure_summary = self._get_failure_summary(task)
                    task.error = (
                        "Task cannot be marked complete: required tool operations failed."
                        f"{failure_summary}\n\n"
                        "Diagnose the failures and fix the implementation. "
                        "Do not claim success when commands fail."
                    )
                    await self.event_bus.emit(
                        Event(
                            type="task.failed",
                            source="agent_loop",
                            data={
                                "task_id": task.id,
                                "reason": "tool_failures_not_recovered",
                                "failed_tools": len([
                                    tc for tc in task.tool_calls
                                    if tc.result and tc.result.execution_failed
                                ]),
                            },
                        )
                    )
                    break

                # Verification phase — truthful completion requires evidence.
                await self._verify_task_completion(task)
                if task.status == TaskStatus.FAILED:
                    await self._reconcile_todos(task)
                    await self.event_bus.emit(
                        Event(
                            type="task.failed",
                            source="agent_loop",
                            data={
                                "task_id": task.id,
                                "reason": "verification_failed",
                            },
                        )
                    )
                    break

                # Reconcile remaining TODOs against real execution evidence
                await self._reconcile_todos(task)

                # COMPLETION INVARIANT (Phase 10.6)
                # No code path sets COMPLETED without passing through
                # can_complete_task. 3/5 is not completion; the runtime is
                # the source of truth, not the model's text. When required
                # work merely *did not happen* (nothing failed), the honest
                # terminal state is PARTIAL — never COMPLETE, never FAILED.
                if not can_complete_task(task):
                    blockers = completion_blockers(task)
                    failed_required = [
                        i for i in task.task_plan.items
                        if i.required and i.status == TodoStatus.FAILED
                    ]
                    if failed_required or task.execution_stats.has_unresolved_failures:
                        task.status = TaskStatus.FAILED
                        task.failure_reason = "required_work_failed"
                        await self.event_bus.emit(
                            Event(
                                type="task.failed",
                                source="agent_loop",
                                data={
                                    "task_id": task.id,
                                    "reason": "required_work_failed",
                                    "blockers": blockers,
                                    "completed_todos": task.task_plan.completed_count,
                                    "total_todos": task.task_plan.total_count,
                                },
                            )
                        )
                    else:
                        task.status = TaskStatus.PARTIAL
                        task.failure_reason = "required_work_incomplete"
                    task.error = (
                        "Task ended without completing all required work: "
                        + "; ".join(blockers)
                    )
                    break

                await self._emit_phase("complete")
                task.status = TaskStatus.COMPLETED
                break

            # Progress & stagnation detection (governor)
            progress = self._iteration_made_progress(iter_calls)
            for tc in iter_calls:
                self._seen_actions.add(self._action_key(tc.tool_name, tc.arguments))
            if progress:
                self._stagnation_counter = 0
            else:
                self._stagnation_counter += 1
                if self._stagnation_counter == STAGNATION_WARN_AT:
                    await self.event_bus.emit(
                        Event(
                            type="progress.stalled",
                            source="agent_loop",
                            data={"stagnant_iterations": self._stagnation_counter},
                        )
                    )
                    self._corrections.append(
                        "No meaningful progress detected in the last iterations "
                        "(same actions repeated, no new inspection or code changes). "
                        "Stop repeating the same operations. Either diagnose the "
                        "blocker by reading the relevant files and errors, or take a "
                        "different concrete action."
                    )
                elif self._stagnation_counter >= STAGNATION_STOP_AT:
                    # If the evidence is already green, finish truthfully instead
                    # of failing a completed task.
                    if await self._stagnation_recovery(task):
                        break
                    task.status = TaskStatus.FAILED
                    task.error = (
                        "Stopped: no meaningful progress detected for "
                        f"{self._stagnation_counter} consecutive iterations. "
                        "The same operations were repeated without advancing the task. "
                        "Partial work (if any) is preserved in the workspace."
                    )
                    await self.event_bus.emit(
                        Event(
                            type="task.failed",
                            source="agent_loop",
                            data={"task_id": task.id, "reason": "stagnation"},
                        )
                    )
                    break

            # Read-only tool call budget: prevent infinite file-reading loops.
            # If the model keeps reading files without writing/modifying anything,
            # force it to produce a text answer.
            if iter_calls:
                readonly_in_iter = sum(
                    1 for tc in iter_calls
                    if tc.tool_name in _READONLY_TOOLS
                )
                wrote_in_iter = sum(
                    1 for tc in iter_calls
                    if tc.tool_name in ("write_file", "edit_file")
                )
                if readonly_in_iter > 0 and wrote_in_iter == 0:
                    self._readonly_tool_calls += readonly_in_iter
                elif wrote_in_iter > 0:
                    # Any write resets the read-only counter
                    self._readonly_tool_calls = 0

                if self._readonly_tool_calls >= _MAX_READONLY_TOOL_CALLS:
                    self._corrections.append(
                        f"You have made {self._readonly_tool_calls} read-only tool calls "
                        "without modifying any files. STOP calling tools and produce "
                        "your final text response NOW. Summarize what you found and "
                        "answer the user's question. Do not call any more tools."
                    )
                    self._readonly_tool_calls = 0  # Reset so we don't spam

            # Diagnosis mode budget: never loop forever in diagnosis
            if self._diagnosis_active:
                self._diagnosis_iterations += 1
                if self._diagnosis_iterations > MAX_DIAGNOSIS_ITERATIONS:
                    task.status = TaskStatus.FAILED
                    task.error = (
                        "Unable to resolve the repeated failure safely after "
                        f"{MAX_DIAGNOSIS_ITERATIONS} diagnostic iterations. "
                        "Stopping automatic repair rather than repeatedly "
                        "modifying files. Review the failing command output manually."
                    )
                    await self.event_bus.emit(
                        Event(
                            type="task.failed",
                            source="agent_loop",
                            data={"task_id": task.id, "reason": "diagnosis_exhausted"},
                        )
                    )
                    break

            # Safety valve: check AFTER processing tool calls (works for both
            # tool-call and text-response iterations)
            if self._consecutive_denials >= 3:
                task.status = TaskStatus.FAILED
                task.error = (
                    "Task blocked: too many consecutive permission denials. "
                    "The current permission policy prevents required operations. "
                    "Configure permissions or run with interactive approval enabled."
                )
                break

        if task.status not in (
            TaskStatus.COMPLETED, TaskStatus.FAILED,
            TaskStatus.PAUSED, TaskStatus.CANCELLED, TaskStatus.PARTIAL,
        ):
            if self._consecutive_denials >= 3:
                task.status = TaskStatus.FAILED
                task.error = task.error or "Task blocked by permission policy"
            elif (
                not task.execution_stats.has_unresolved_failures
                and not [
                    i for i in task.task_plan.items
                    if i.required and i.status == TodoStatus.FAILED
                ]
            ):
                # Phase 10.6: nothing actually failed — the run simply ended
                # before all required work completed. PARTIAL is honest;
                # FAILED would claim a failure that did not happen and
                # COMPLETE would claim work that did not happen.
                task.status = TaskStatus.PARTIAL
                task.failure_reason = "required_work_incomplete"
                task.error = (
                    task.error
                    or "Iteration budget reached before all required work completed."
                )
            else:
                task.status = TaskStatus.FAILED
                task.error = task.error or (
                    "The model did not produce a final answer within the iteration "
                    "budget. Try rephrasing your request or using a different model."
                )

        # ── Canonical run terminal events (Part 8) ─────────────────────
        terminal = task.status in (
            TaskStatus.COMPLETED, TaskStatus.PARTIAL,
        )
        if task.status == TaskStatus.FAILED:
            await self._emit_semantic(
                semantic.RUN_FAILED,
                {"reason": task.failure_reason or "unknown", "error": (task.error or "")[:300]},
            )
        elif task.status == TaskStatus.CANCELLED:
            await self._emit_semantic(
                semantic.RUN_CANCELLED, {"reason": "user_requested"}
            )
        elif terminal:
            metrics = self.run_metrics.finish(
                failure_reason="" if task.status == TaskStatus.COMPLETED else (task.failure_reason or "")
            )
            await self._emit_semantic(
                semantic.RUN_COMPLETED,
                {"status": task.status.value, "metrics": metrics},
            )

        # Record performance if task_aware is available
        if self.task_aware is not None and self.router is not None:
            decisions = self.router.get_routing_decisions()
            model_used = decisions[-1].selected_model if decisions else "unknown"
            provider_used = decisions[-1].selected_provider if decisions else "unknown"
            self.task_aware.record_task_result(
                model_id=model_used,
                provider=provider_used,
                task_type=task_type.value if task_type else "unknown",
                success=task.status == TaskStatus.COMPLETED,
                tool_calls=len(task.tool_calls),
                iterations=task.iterations,
            )

        await self.event_bus.emit(
            Event(
                type="task.completed",
                source="agent_loop",
                data={
                    "task_id": task.id,
                    "status": task.status.value,
                    "failure_reason": task.failure_reason,
                    "iterations": task.iterations,
                    "tool_calls": len(task.tool_calls),
                    "stats": task.execution_stats.summary(),
                    "attempted": task.execution_stats.attempted,
                    "succeeded": task.execution_stats.succeeded,
                    "failed": task.execution_stats.failed,
                    "recovered": task.execution_stats.recovered,
                    "unresolved": task.execution_stats.unresolved,
                    "verification_passed": task.verification_passed,
                    "verification_summary": task.verification_summary,
                    "files_changed": list(self._modified_files),
                    "completed_operations": list(self._completed_operations),
                    "todos": task.task_plan.to_event_items(),
                    "todos_completed": task.task_plan.completed_count,
                    "todos_failed": task.task_plan.failed_count,
                    "todos_total": task.task_plan.total_count,
                    "tests_run": task.tests_run,
                    "tests_passed": task.tests_passed,
                    "models_used": list(task.models_used),
                    "model_fallbacks": task.model_fallbacks,
                    "git_commit": task.git_commit,
                    "git_push": task.git_push,
                    "paused_reason": task.paused_reason,
                },
            )
        )

        return task

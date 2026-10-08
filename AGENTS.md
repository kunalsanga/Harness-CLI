# Harness — Agent Orientation

> Context status: verified against repository structure on 2026-09-15.
> Update this file when architectural boundaries change.

**Full architecture:** [`docs/AI_CONTEXT.md`](docs/AI_CONTEXT.md)
**Task-to-context map:** [`docs/AI_CONTEXT.md#task-to-context-navigation`](docs/AI_CONTEXT.md)
**Persistent multi-agent context:** [`.agent-context/`](.agent-context/CURRENT_STATE.md)

---

## Required Context Startup

Before repository work, read these four concise shared context files in order:

1. `.agent-context/CURRENT_STATE.md`
2. `.agent-context/PROJECT.md`
3. `.agent-context/ARCHITECTURE.md`
4. `.agent-context/DECISIONS.md`

Then use the task-to-context map below to select only the relevant subsystem
`AGENTS.md` and its **READ FIRST** files. Read `.agent-context/KNOWN_ISSUES.md`,
`ROADMAP.md`, `TESTING.md`, or `CHANGELOG.md` only when they matter to the task.
Do not read every context file or every subsystem `AGENTS.md` by default.

---

## What Is Harness

A model-agnostic, autonomous software-engineering CLI. It uses a
conversation-first TUI, a structured task planning system, a multi-agent
runtime, and a pluggable provider layer to execute engineering tasks.

Entrypoint: `uv run harness` → interactive shell  
Named entry: `harness` (defined in `pyproject.toml`)

---

## Execution Flow (summary)

```
uv run harness
   → cli/main.py (_main_callback)
   → cli/interactive.py (InteractiveShell)
   → runtime/runtime.py (EngineeringRuntime)
   → planning/planner.py (Planner)
   → agents/domain.py (TaskGraph)
   → agents/scheduler.py (Scheduler)
   → agents/worker.py (WorkerAgent)
   → agent/loop.py (AgentLoop)
   → routing/router.py (ModelRouter)
   → providers/<name>.py (ModelProvider)
```

---

## Context Budget Rule

**Do not read every AGENTS.md file.**

1. Read this file.
2. Identify the relevant subsystem from the table in `docs/AI_CONTEXT.md`.
3. Read **only that subsystem's** `AGENTS.md`.
4. Read only the files listed under **READ FIRST** in that file.
5. Expand outward only when evidence requires it.

---

## Key Testing Commands

```bash
uv run pytest tests/ -x -q                   # full suite
uv run pytest tests/unit/test_<area>.py -x   # targeted
uv run harness --help                         # CLI smoke test
uv run python -c "import harness_core; print('OK')"  # import check
```

---

## Architectural Invariants (never break)

- Do not create a second `EngineeringRuntime`.
- Do not bypass `EngineeringRuntime` → `Scheduler` → `WorkerAgent` → `AgentLoop`.
- Do not bypass `ModelRouter` / `ModelGateway`.
- Do not duplicate provider abstractions (always subclass `ModelProvider`).
- `EventBus` is the only observability mechanism — do not add parallel logging.
- `9router` (`NineRouterProvider`) owns its own model selection — do not expand its internals into Harness routing.
- Do not expose credentials, chain-of-thought, or tool arguments in EventBus events.
- Do not silently claim verification — `VerificationEngine` must run.
- Do not allow stale task context to leak across tasks.
- Do not create files outside the active workspace scope.

---

## Multi-Agent Rule

This repository may be worked on by multiple AI coding agents — Claude Code,
Google Antigravity, Freebuff, and Harness agents.

Before modifying architecture:

1. Read the relevant context file.
2. Inspect the current implementation.
3. Do not assume a previous agent's implementation is correct.
4. Preserve existing architectural boundaries unless the task explicitly requires changing them.
5. Update the relevant `AGENTS.md` / `AI_CONTEXT.md` when an architectural boundary changes.

Never create a duplicate implementation of an existing subsystem merely because its behavior is not immediately obvious.

---

## Token-Efficient Agent Workflow

```
STEP 1  Read AGENTS.md (this file) + identify subsystem
STEP 2  Read that subsystem's AGENTS.md
STEP 3  Read only the files listed under READ FIRST
STEP 4  Grep for targeted symbols/references
STEP 5  Inspect only the relevant implementation
STEP 6  Implement the smallest correct change
STEP 7  Run focused tests: uv run pytest tests/unit/test_<area>.py -x
STEP 8  Run broader tests if appropriate
STEP 9  Verify the actual user scenario
```

Do NOT: grep the entire repo, read every source file, re-read unchanged files,
reconstruct architecture from scratch, or modify unrelated subsystems.

---

## Context Maintenance

Update the relevant context file whenever any of these change:
- execution flow, module ownership, provider architecture, routing,
  task planning, CLI architecture, tool architecture, verification,
  recovery, configuration.

Do **not** update for minor implementation details.

After significant completed work, update `.agent-context/CURRENT_STATE.md` and
append a concise entry to `.agent-context/CHANGELOG.md`. Update `KNOWN_ISSUES.md`
when an issue is found or resolved, `DECISIONS.md` only for an accepted decision,
and `ROADMAP.md` only when phase/status changes. Record uncertainty explicitly;
do not turn an unverified report into a fact. Keep this context agent-agnostic.

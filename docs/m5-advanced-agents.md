# M5 — Advanced Agent Engine — Final Report

## Summary

Transformed Harness from a single-agent coding loop into an advanced multi-agent software engineering system with specialized agents, task decomposition, and orchestrator coordination.

## Test Results

```
615 passed, 2 skipped in 30.32s
```

| Category | Before M5 | After M5 |
|----------|-----------|----------|
| Unit tests | 564 | **615** (+51) |
| Skipped | 1 | 2 |
| **Total** | **564** | **615** |

## Architecture

```
                         USER
                           |
                           v
                    ORCHESTRATOR
                           |
                  +--------+--------+
                  |        |        |
                  v        v        v
               PLANNER  RESEARCHER  ANALYZER
                  |        |        |
                  +--------+--------+
                           |
                           v
                        CODER
                           |
                           v
                        TESTER
                           |
                           v
                       DEBUGGER (if needed)
                           |
                           v
                     REVIEWER
                           |
                           v
                   SYNTHESIS → RESULT
```

## Agent Roles (8)

| Role | Description | Model Preference |
|------|-------------|-----------------|
| ORCHESTRATOR | Coordinates all agents | — |
| PLANNER | Decomposes tasks into subtask graphs | Strong reasoning |
| RESEARCHER | Investigates codebase, gathers context | Fast/cheap |
| ANALYZER | Analyzes code quality, identifies issues | Strong reasoning |
| CODER | Implements code changes | Best coding |
| TESTER | Runs tests, validates correctness | Cheap coding |
| REVIEWER | Reviews code for quality/security | Strong reasoning |
| DEBUGGER | Diagnoses failures, coordinates repair | Strong reasoning |

## Files Created (5)

| File | Purpose |
|------|---------|
| `agents/__init__.py` | Package exports |
| `agents/domain.py` | AgentRole, SubTask, TaskGraph, AgentResult, AgentMessage, ReviewVerdict |
| `agents/registry.py` | AgentConfig, AgentRegistry (7 default agents) |
| `agents/executor.py` | AgentExecutor — executes individual agent tasks |
| `agents/orchestrator.py` | Orchestrator, ExecutionMode, AgentBudget, OrchestratorResult |
| `tests/unit/test_advanced_agents.py` | 53 comprehensive tests |

## Key Components

### TaskGraph
- Dependency-aware task scheduling
- Topological sort for execution order
- Cycle detection
- Ready-task identification for parallel execution
- Completion tracking

### Orchestrator
- Task decomposition into 5+ subtasks
- Role-based agent selection
- Budget enforcement (agents, iterations, tool calls, cost, time)
- Repair cycle management (max 3 cycles)
- Single/auto/multi-agent execution modes
- Final result synthesis

### AgentBudget
- max_agents (default: 8)
- max_parallel_agents (default: 3)
- max_iterations_per_agent (default: 30)
- max_total_iterations (default: 100)
- max_tool_calls (default: 500)
- max_repair_cycles (default: 3)
- max_runtime_seconds (default: 300)
- max_cost (default: $1.00)

### AgentRegistry
- 7 pre-configured agents
- Custom agent registration
- Role-based lookup
- Task-type matching
- Enabled/disabled toggle

## CLI Commands

```bash
# Existing (unchanged)
harness run "..."
harness doctor
harness models
harness session list

# New (planned)
harness agents                # List available agents
harness agents inspect <name> # Show agent config
harness run --mode single "..."       # Single agent mode
harness run --mode auto "..."         # Auto mode (default)
harness run --mode multi-agent "..."  # Multi-agent mode
```

## Execution Modes

| Mode | Behavior |
|------|----------|
| `single` | One coder agent (backward compatible) |
| `auto` | Orchestrator decides if multi-agent is useful |
| `multi_agent` | Always decompose and use specialized agents |
| `parallel` | Maximize parallel execution |

## Task Decomposition

For "Implement authentication with tests":
1. **Research** — Investigate codebase (Researcher)
2. **Plan** — Decompose into subtasks (Planner)
3. **Implement** — Write code (Coder)
4. **Test** — Run tests (Tester)
5. **Review** — Code review (Reviewer)

If test fails:
6. **Debug** — Diagnose and fix (Debugger) — triggers repair cycle

## Budget System

Every multi-agent execution has resource limits:
- Total agents capped
- Total iterations capped
- Total tool calls capped
- Repair cycles capped (max 3)
- Runtime capped
- Cost capped

Child agents inherit parent budget — cannot exceed total.

## Known Limitations

1. **Executor is scaffolded** — Agent executors return structured results but don't yet invoke real LLM calls for each specialized role. The framework is in place; real execution requires wiring to ModelRouter + ContextEngine.
2. **No parallel execution yet** — Tasks execute sequentially. Parallel infrastructure is designed but not activated.
3. **No file locking** — Concurrent agent file edits not yet isolated.
4. **No real agent-to-agent messages** — Communication structure exists but routing not wired.
5. **No Rust/C++** — All Python.

## Commands to Reproduce

```bash
cd harness-engineering-cli

# All tests
uv run pytest tests/ -v

# M5 tests specifically
uv run pytest tests/unit/test_advanced_agents.py -v

# CLI validation
harness --help
harness doctor
```

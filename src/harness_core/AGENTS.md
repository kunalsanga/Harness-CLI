# harness_core — Package Root Context

> Context status: verified against repository structure on 2026-09-15.

This is the root of the `harness_core` Python package. It serves as the bridge between the global architecture and individual subsystems.

**Full architecture:** [`/docs/AI_CONTEXT.md`](/docs/AI_CONTEXT.md)

---

## Subsystem Map

| Subsystem | Directory | Purpose |
|---|---|---|
| CLI | `cli/` | Typer entrypoint, InteractiveShell, TUI |
| Agent loop | `agent/` | AgentLoop, TodoItem, steering, completion |
| Multi-agent | `agents/` | TaskGraph, Scheduler, WorkerAgent, MessageBus |
| Runtime | `runtime/` | EngineeringRuntime, ProjectState, governance |
| Planning | `planning/` | Planner, Plan, PlanningResult |
| Providers | `providers/` | ModelProvider implementations |
| Routing | `routing/` | ModelRouter, fallback, scoring, health |
| Context | `context/` | ContextPipeline, ContextReuseManager, compaction |
| Memory | `memory/` | MemoryManager, RAG retrieval, knowledge graph |
| Recovery | `recovery/` | FailureClassifier, RecoveryOrchestrator |
| Verification | `verification/` | VerificationEngine, integrity checks |
| Tools | `tools/` | Filesystem, shell, git, search tools |
| Permissions | `permissions/` | PermissionManager, tool gating |
| Config | `config/` | CredentialResolver, HarnessConfig, dotenv |
| Observability | `observability/` | EventBus, metrics, semantic events |
| Session | `session/` | Session lifecycle and persistence |
| Models | `models/` | Model metadata registry, empirical data |
| Security | `security/` | Security utilities (currently minimal) |
| Extensions | `extensions/` | Plugin extension points |
| Hooks | `hooks/` | Hook system |
| MCP | `mcp/` | MCP server integration |
| Plugins | `plugins/` | Plugin management |

---

## READ FIRST

`__init__.py` — exports and package-level metadata.

Then navigate to the relevant subsystem directory and read its `AGENTS.md`.

## ARCHITECTURAL INVARIANTS

- `harness_core` is a single Python package installed via `hatchling`.
- Do not create top-level modules outside this package for Harness functionality.
- Subsystem boundaries are enforced by directory — do not import across unrelated subsystems without strong justification.
- `observability/events.py:EventBus` is the only cross-cutting dependency all subsystems share.

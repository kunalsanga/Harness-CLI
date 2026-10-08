# Harness Architecture

Last verified: 2026-09-30. See `docs/AI_CONTEXT.md` for the wider module map;
that document and README include legacy/multi-provider descriptions and should
not override the active path described here.

## Interactive request path

```text
uv run harness
  → src/harness_core/cli/main.py::_main_callback
  → cli/interactive.py::InteractiveShell
  → runtime/runtime.py::EngineeringRuntime.execute_interactive
  → agents/domain.py::TaskGraph (one interactive task)
  → agents/scheduler.py::Scheduler
  → agents/worker.py::WorkerAgent
  → agent/loop.py::AgentLoop
  → agent/loop.py existing short plan/TODO (engineering requests only)
  → providers/factory.py::create_providers
  → routing/router.py::ModelRouter (discovery, normalized capability filtering)
  → routing/fallback.py::FallbackEngine (normalized provider-neutral errors)
  → providers/<adapter>.py::ModelProvider
  → provider API
```

`factory.py` resolves project providers from `.harness/config.yaml` and credentials
through `CredentialResolver`. Without a providers list, OpenRouter remains the
default adapter. Interactive and `harness run` construct a router over configured
adapters; AgentLoop and runtime auxiliary model calls use that router. Adapters
own vendor request/response formats, streaming, tool translation, discovery,
and HTTP error details.
The interactive path classifies the request and constructs one TaskGraph node;
it does not invoke `planning/planner.py::Planner`. `EngineeringRuntime.run()`
is a separate path that does invoke the Planner.

## Model discovery and routing

`ModelRouter.refresh_models()` discovers metadata through configured adapters
and merges configured declarations when discovery is unavailable. Capability
values are tri-state; unknown values do not meet a task requirement. OpenRouter's
dynamic free route comes from adapter routing hints. Concrete OpenRouter free
models come from its catalog rather than a list in core routing.

`ModelInfo` is the normalized model record. `ProviderRequestError` carries
normalized authentication, rate-limit, availability, network, timeout,
provider, invalid-request, payment, and tool categories. `FallbackEngine`
rotates only bounded retryable/availability failures; authentication and tool
errors stop rotation. Attempts reuse the same CompletionRequest, including
messages and tool results.

Adapter configuration is documented in `docs/providers.md`. Paid model
selection is opt-in through `routing.allow_paid_models`; free mode remains a
hard constraint.

## UI and observability

- `cli/interactive.py` owns the REPL, commands, provider wiring, and EventBus
  subscriptions.
- `cli/conversation.py` projects task and tool events into conversation output.
- `cli/runtime_dashboard.py` renders unified runtime events.
- `observability/events.py::EventBus` is the shared event channel. Do not add a
  parallel progress/event system or fabricate progress.
- Tool output should be concise; raw provider diagnostics belong behind verbose
  or explicit debug output.

## Interactive execution state and timing

- `agent/types.py::TaskExecutionTimeline` is attached to each AgentLoop `Task`
  and owns the canonical `ExecutionState`: IDLE, UNDERSTANDING, PLANNING,
  THINKING, WORKING, REPAIRING, VERIFYING, COMPLETED, FAILED, CANCELLED, PAUSED.
- Existing `TaskStatus` and `task.phase` remain intact. AgentLoop maps actual
  phase changes to canonical state and emits `execution.state`; the conversation
  renderer projects that event rather than polling runtime internals.
- Task/phase durations use `time.monotonic()`. Phase durations are exclusive
  wall-clock segments. Model request and tool durations are nested diagnostics,
  never added again to task elapsed time. The timer spans model failover.
- The existing AgentLoop plan and runtime-owned TODO states are rendered and
  included with status in each working prompt. Read-only requests skip the
  additional plan call; simple engineering requests request 2–3 steps, larger
  requests up to 6. No second planner was added.

## Invariants

- Keep one `EngineeringRuntime` and the
  `Scheduler → WorkerAgent → AgentLoop` execution chain.
- Route model calls through `ModelRouter` / `FallbackEngine`.
- Keep `ModelProvider` as the provider abstraction.
- Preserve task context across model failover; do not restart a task merely to
  switch models.
- Tool execution errors are task/tool outcomes, not model authentication or
  rate-limit errors.
- Do not leak credentials, hidden reasoning, or full tool arguments in events
  or normal UI output.
- Do not claim verification unless `VerificationEngine` ran.

## Repository packaging note

`src/harness_core/models/` contains runtime model registry/discovery modules
used by the CLI and router. Keep the root local-cache ignore anchored as
`/models/`; an unanchored `models/` pattern would hide this active source
package from Git and packaging inputs.

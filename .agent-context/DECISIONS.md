# Accepted Decisions and Working Constraints

Last verified: 2026-09-29. These constraints come from the project instructions
and recent user requests; update only when the user changes them.

## Active provider and model policy

- Preserve the current OpenRouter interactive integration.
- Use the configured free-model pool during free-mode development; do not add a
  paid fallback or silently route to paid models.
- Share the existing `OPENROUTER_API_KEY` credential through the existing
  credential resolver. Do not create a second credential store.
- Do not add or reactivate Ollama, direct NVIDIA, Groq, 9router, or LiteLLM as
  interactive fallbacks without an explicit change in direction. Their source
  modules may remain in the repository.

## Execution architecture

- Do not rewrite AgentLoop, ModelRouter, FallbackEngine, TaskGraph, Scheduler,
  Planner, or provider architecture for UI work.
- Preserve `EngineeringRuntime → Scheduler → WorkerAgent → AgentLoop` and
  `ModelRouter → FallbackEngine` boundaries.
- Fail over inside the same task. Do not ask the user to retry after a
  recoverable model limit or model-specific outage.
- Authentication/access failures should not rotate through models when the
  credential/account restriction is shared. Rate limits are not authentication
  failures. Determine categories from status/exception evidence, never guesses.
- Tool execution failures are tool failures; do not label them as provider
  failures or trigger model failover for them.

## User-facing behavior

- Show only real progress and failover events.
- Keep normal UI output concise and useful; retain raw diagnostics for explicit
  verbose/debug use.
- Distinguish configured models, currently available models, and tool
  capability. Unknown capability must remain unknown.
- Keep `EventBus` as the observability channel; never publish credentials,
  hidden reasoning, or oversized tool arguments.
- Do not claim tests or verification ran unless they actually ran.

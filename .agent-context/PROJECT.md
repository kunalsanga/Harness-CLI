# Harness Project

Last verified: 2026-09-29

Harness is a Python command-line software engineering agent. It accepts natural
language tasks, plans and executes work through an agent runtime, uses workspace
tools, and reports results through a terminal interface.

## Project identity

- Package: `harness-engineering`; import package: `harness_core`.
- Python requirement: `>=3.12`.
- Console entry point: `harness = harness_core.cli.main:app`.
- Main interactive entry: `uv run harness`.
- Current interactive provider: OpenRouter, instantiated by
  `src/harness_core/cli/interactive.py`.
- Current OpenRouter development strategy: configured free-model pool with
  bounded fallback; the pool must not silently select paid models.

## Scope and terminology

The repository is broader than its current interactive setup. Provider modules
for other backends and older CLI capabilities still exist in the checkout.
“OpenRouter-only” in current task instructions means preserve the active
interactive provider and free-pool path; it does not mean those other files have
been deleted. Check the actual entry path before making provider claims.

“Configured” means present in the static pool. “Available” requires a live
request or health evidence. “Tool-capable” requires a discovered or otherwise
verified capability. Do not conflate these states.

## Product priorities

Keep the interactive shell compact, professional, and honest about progress,
model selection, failover, errors, and verification. Preserve the execution
architecture unless a task explicitly calls for an architectural change.

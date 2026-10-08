# cli/ — CLI & Interactive Shell Subsystem

> Context status: verified against repository structure on 2026-09-15.

**Responsibility:** Typer CLI entrypoint, all CLI commands, the conversation-first interactive TUI, the live runtime dashboard, auth commands, and shell utilities.

---

## READ FIRST

- `main.py` — Typer app, all commands, `_main_callback` (no-subcommand → interactive shell)
- `interactive.py` — `InteractiveShell` (conversation loop, prompt_toolkit REPL)

## READ IF NEEDED

- `conversation.py` — conversation state management and rendering
- `runtime_dashboard.py` — live `EventBus`-driven execution dashboard
- `ui.py` — shared Rich UI helpers and display utilities
- `completion.py` — shell completion helpers
- `auth.py` — `auth_app` (credential management subcommands)

## DO NOT READ FOR NORMAL CLI TASKS

- `agent/loop.py` — loop internals; CLI does not access them directly
- `runtime/runtime.py` — runtime is invoked through `InteractiveShell`, not read directly
- `providers/` — provider selection is runtime/routing concern

---

## Entrypoint Flow

```
uv run harness
   → main.py:app
   → _main_callback (no subcommand)
   → interactive.py:run_interactive()
   → InteractiveShell.run()
        → reads user input (prompt_toolkit)
        → calls EngineeringRuntime.execute()
        → renders via EventBus subscriptions
```

Explicit subcommands (`harness run`, `harness shell`, `harness doctor`, etc.) are defined in `main.py`.

---

## Dependencies

**Depends on:**
- `runtime/runtime.py` — `EngineeringRuntime`
- `observability/events.py` — `EventBus` (for dashboard rendering)
- `config/` — `HarnessConfig`, `CredentialResolver`
- `session/` — session loading/saving

**Used by:**
- Nothing — `cli/` is the outermost layer

---

## Architectural Invariants

- `main.py` loads `.env` at startup before any environment reads (`config/dotenv.py:load_dotenv()`).
- The `_main_callback` must remain the entry point for the interactive shell — do not move it.
- All TUI rendering must be driven by `EventBus` events — do not poll internal runtime state.
- Windows UTF-8 encoding is forced at startup in `main.py` — do not remove this.
- Auth (`auth.py`) uses `CredentialResolver` — never write raw credentials to stdout.

---

## Common Mistakes

| Mistake | Correct approach |
|---|---|
| Adding business logic to `main.py` command handlers | Delegate to runtime/services; keep handlers thin |
| Polling runtime state for TUI updates | Subscribe to `EventBus` events |
| Reading provider state directly in CLI | Use `harness providers` commands or runtime interfaces |
| Removing Windows UTF-8 encoding setup | Keep it — required for Rich/prompt_toolkit on Windows |
| Adding a new subcommand without registering `add_typer` | Register in `main.py` |

---

## Tests

- `tests/unit/test_interactive_shell.py`
- `tests/unit/test_production_ux.py`
- `tests/unit/test_completion_ux.py`
- `tests/unit/test_runtime_dashboard.py`
- `tests/unit/test_welcome_screen.py`
- `tests/unit/test_ux_interaction.py`
- `tests/unit/test_ux_upgrade.py`
- `tests/unit/test_m8_productization.py`

## Next: inspect

TUI/UX issue → `interactive.py` + `conversation.py`.
Dashboard issue → `runtime_dashboard.py`.
Auth issue → `auth.py` + `config/credentials.py`.
New CLI command → `main.py` + relevant service module.

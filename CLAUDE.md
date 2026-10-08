# Harness — Claude Code Instructions

> Context status: verified against repository structure on 2026-09-15.
> Update this file when architectural boundaries change.

You are working on **Harness**, a model-agnostic, autonomous software-engineering CLI.

**Start here:** [`AGENTS.md`](AGENTS.md) for generic orientation.
**Full architecture:** [`docs/AI_CONTEXT.md`](docs/AI_CONTEXT.md)

---

## Claude-Specific Operating Instructions

### File Discipline
- Do not create scratch files (`.txt`, debug scripts, logs) in the repository root.
- All temporary exploration should stay in memory or `tests/`.
- Always check whether a module already exists before creating a new one.

### Tool Use
- Do not shell-grep the entire repository before reading context files.
- Prefer targeted searches: `grep -r "ClassName" src/harness_core/<subsystem>/`.
- Do not read test files to understand architecture — read the source modules listed in `docs/AI_CONTEXT.md`.

### Verification
- Do not claim a fix is complete without running `uv run pytest tests/unit/test_<area>.py -x`.
- Do not claim imports work without running `uv run python -c "import harness_core"`.
- `VerificationEngine` (`src/harness_core/verification/engine.py`) must not be bypassed.

### Secrets & Credentials
- Never print, log, or embed API keys or `.env` values.
- `CredentialResolver` (`src/harness_core/config/credentials.py`) is the only credential access point.
- `mask_secret()` must be used for any display of credential values.

### Architecture Discipline
- Do not add a second runtime — `EngineeringRuntime` already exists.
- Do not add a second scheduler — `Scheduler` already exists.
- Subclass `ModelProvider` (`src/harness_core/providers/base.py`) for all new providers.
- `EventBus` in `src/harness_core/observability/events.py` is the only observability channel.

### 9router
- `NineRouterProvider` sends `model="qd/auto"` to the 9router gateway.
- The gateway selects the underlying model — Claude should never expand that selection into Harness routing logic.
- Credential key: `NINE_ROUTER_API_KEY` (resolved via `CredentialResolver`).

### Multi-Agent Safety
See `AGENTS.md` → Multi-Agent Rule.

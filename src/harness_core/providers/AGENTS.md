# providers/ — Model Provider Subsystem

> Context status: verified against repository structure on 2026-09-15.

**Responsibility:** All model provider implementations. Each provider connects Harness to a specific LLM backend (Groq, NVIDIA, OpenRouter, Ollama, 9router, LiteLLM). All providers implement the `ModelProvider` ABC.

---

## READ FIRST

- `base.py` — `ModelProvider` ABC, `CompletionRequest`, `CompletionResponse`, `ModelInfo` (always read first for any provider work)

## READ IF NEEDED (by provider)

- `factory.py` — configuration-driven provider construction (`create_providers`)
- `http_adapters.py` — OpenAI-compatible / OpenAI / Anthropic / Gemini HTTP adapters
- `openai.py` / `anthropic.py` / `gemini.py` / `openai_compatible.py` — thin re-exports
- `normalized.py` — wraps adapters to raise `ProviderRequestError`
- `routed.py` — `RoutedModelProvider` facade for planner/auxiliary calls via `ModelRouter`
- `ninerouter.py` — 9router automatic routing provider
- `groq.py` — Groq API integration
- `nvidia.py` — NVIDIA NIM integration
- `openrouter.py` — OpenRouter integration
- `ollama.py` — Local Ollama integration
- `litellm.py` — LiteLLM multi-provider integration
- `stream_events.py` — streaming event helpers shared across providers

## DO NOT READ FOR NORMAL PROVIDER TASKS

- `routing/` — routing selects providers; providers don't configure routing
- `cli/` — providers have no CLI concerns
- `agent/loop.py` — loop calls providers through `ModelRouter`, not directly

---

## Provider Contract

Every provider must implement:

```python
class ModelProvider(abc.ABC):
    @property
    def name(self) -> str: ...
    async def generate(self, request: CompletionRequest) -> CompletionResponse: ...
    async def stream(self, request: CompletionRequest): ...
    async def list_models(self) -> list[ModelInfo]: ...
    async def health_check(self) -> bool: ...
    async def close(self) -> None: ...  # optional, override if needed
```

- `model` field in `CompletionRequest` selects the specific model within the provider (or `qd/auto` for 9router).
- `CompletionResponse.model` should reflect the actual model used.

---

## 9router — Critical Architecture

```
NineRouterProvider
   → sends CompletionRequest(model="qd/auto")
   → HTTP POST to localhost:20128/v1
   → 9router gateway selects underlying model
   → response.model = actual model used (diagnostic only)
```

**Invariant:** Never expand `qd/auto` into a list of underlying models inside Harness. Never add Harness-side model selection logic to `NineRouterProvider`. The gateway owns model selection entirely.

Credential: `NINE_ROUTER_API_KEY` via `CredentialResolver`.

---

## Credential Resolution

All providers resolve credentials via `CredentialResolver` from `config/credentials.py`:

```
env var → .env file → ~/.harness/credentials.json
```

Provider env keys:
| Provider | Key |
|---|---|
| Groq | `GROQ_API_KEY` |
| NVIDIA | `NVIDIA_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| 9router | `NINE_ROUTER_API_KEY` |
| Ollama | (host-based, no key) |

---

## Adding a New Provider

1. Create `providers/<name>.py`.
2. Subclass `ModelProvider` from `base.py`.
3. Implement all abstract methods.
4. Add the provider env key to `config/credentials.py:_PROVIDER_ENV_KEYS`.
5. Register the provider in the appropriate CLI/runtime initialization path.
6. Add tests in `tests/unit/test_<name>_provider.py`.

---

## Dependencies

**Depends on:**
- `config/credentials.py` — credential resolution

**Used by:**
- `routing/router.py` — `ModelRouter` wraps providers
- `agent/loop.py` — `AgentLoop` calls `ModelProvider` via `ModelRouter`

---

## Architectural Invariants

- Always subclass `ModelProvider` — never duck-type a provider.
- Never expose raw API keys in `CompletionResponse`, `EventBus` events, or logs.
- `health_check()` must be safe to call frequently — no side effects.
- Provider failures must surface as exceptions, not silent empty responses.

---

## Common Mistakes

| Mistake | Correct approach |
|---|---|
| Not subclassing `ModelProvider` | Always subclass from `base.py` |
| Hardcoding credentials | Use `CredentialResolver` |
| Adding model selection inside `NineRouterProvider` | The gateway owns model selection |
| Silently swallowing API errors | Raise; let `AgentLoop`/`RecoveryOrchestrator` handle |
| Accessing env vars directly in providers | Use `CredentialResolver` |

---

## Tests

- `tests/unit/test_ninerouter.py`
- `tests/unit/test_groq_discovery.py`
- `tests/unit/test_nvidia_provider.py`
- `tests/unit/test_free_routing.py` (OpenRouter free tier)
- `tests/unit/test_provider_auth_failure.py`
- `tests/unit/test_provider_error_propagation.py`
- `tests/unit/test_empirical_routing.py`

## Next: inspect

Routing issues → `routing/AGENTS.md`.
Credential issues → `config/credentials.py`.

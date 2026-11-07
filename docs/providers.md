# Model providers

Harness separates task execution from model vendors. AgentLoop and the task
runtime consume the ModelProvider contract through ModelRouter. Provider
adapters own API formats, credentials, discovery, streaming, tool-call
translation, and HTTP error normalization.

## Supported Providers

Harness supports multi-provider routing with automatic failover across:
1. **OpenRouter**: Default multi-model gateway with dynamic free routing (`OPENROUTER_API_KEY`).
2. **OpenAI**: Direct integration supporting GPT-4o, o1, o3-mini, and compatible gateways (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `OPENAI_MODEL`).
3. **Azure OpenAI**: Dedicated Azure endpoints and model deployments (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`).
4. **Groq**: Ultra low-latency LPU inference for open models (`GROQ_API_KEY`, optional `GROQ_BASE_URL`, `GROQ_MODEL`).
5. **NVIDIA NIM**: Enterprise-grade inference endpoints for open models (`NVIDIA_NIM_API_KEY` / `NVIDIA_API_KEY`, optional `NVIDIA_NIM_BASE_URL`, `NVIDIA_NIM_MODEL`).
6. **FreeLLMAPI**: Free model provider with automatic discovery and fallbacks (`FREELLMAPI_API_KEY`, optional `FREELLMAPI_BASE_URL`, `FREELLMAPI_MODEL`).
7. **Additional Providers**: Anthropic, Google Gemini, Ollama (local), 9router, LiteLLM, and arbitrary OpenAI-compatible endpoints.

## Configure providers

Credentials are resolved by CredentialResolver from a provider environment
variable, local .env, or the Harness credential store. Keep .env local and
never commit it. Provider priority and fallback settings live in
`.harness/config.yaml` or environment variables:

```yaml
providers:
  openrouter:
    enabled: true
  nvidia:
    enabled: true
    model: meta/llama-3.1-70b-instruct
  freellmapi:
    enabled: true
  openai:
    enabled: true
    model: gpt-4o

routing:
  strategy: auto
  allow_paid_models: false
  provider_priority: ["openrouter", "nvidia", "freellmapi", "openai"]
```

Common environment variables:
- `HARNESS_PROVIDER_PRIORITY`: Comma-separated list (e.g. `openrouter,nvidia,freellmapi,openai`)
- `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
- `FREELLMAPI_API_KEY`, `FREELLMAPI_BASE_URL`, `FREELLMAPI_MODEL`
- `NVIDIA_NIM_API_KEY`, `NVIDIA_NIM_BASE_URL`, `NVIDIA_NIM_MODEL`

Capabilities are tri-state. A missing capability declaration means unknown,
not supported or unsupported. For adapters whose catalog does not report
capabilities, Harness infers known capabilities (e.g. tool calling and context
length for known instruct/chat models) or uses project configuration.
The router only selects a candidate when its required capabilities are
affirmatively known.

## Adapters & Auto-Discovery

The provider factory auto-discovers configured providers with available credentials.
Catalog discovery is adapter-specific; OpenRouter, NVIDIA NIM, FreeLLMAPI, and OpenAI
dynamically discover models from their `/models` or `/health` endpoints when available.

## Intelligent Failover & Routing

The router constructs a multi-provider model chain according to task requirements,
cost policies, and provider priority. FallbackEngine coordinates execution with
automatic fault tolerance:
- **Rate Limit (429) & Quota Exhaustion**: Provider marked degraded/exhausted; router fails over to the next candidate provider.
- **Authentication Failure (401/403)**: Emits `model.failover` and switches to the next configured provider.
- **Model-Specific Payment Required (402)**: Rotates to the next candidate model or provider.
- **Timeouts & Transient Network Errors (5xx)**: Exponential backoff with retry before rotating candidate.
- **State Preservation**: Request messages, task state, and tool results are preserved intact across failover transitions.
- **Token Usage Attribution**: Token counts and estimated costs are tracked and attributed per provider and per model on the Task.

## Pre-Deployment Verification Commands

- `harness providers status`: Display status table with health, masked credentials, base URLs, and model count.
- `harness providers test`: Non-destructive connectivity and latency probe across all configured providers.
- `harness config validate`: Validate configuration files, priority order, and credentials without executing tasks.
- `harness models list --provider <name>`: Discover and list models for a specific provider.
- Interactive TUI: `/providers` and `/provider <name>` slash commands.

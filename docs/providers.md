# Model providers

Harness separates task execution from model vendors. AgentLoop and the task
runtime consume the ModelProvider contract through ModelRouter. Provider
adapters own API formats, credentials, discovery, streaming, tool-call
translation, and HTTP error normalization.

## Configure providers

Credentials are resolved by CredentialResolver from a provider environment
variable, local .env, or the Harness credential store. Keep .env local and
never commit it. Provider selection and endpoint/model metadata live in
.harness/config.yaml:

    providers:
      openrouter:
        enabled: true
      anthropic:
        enabled: true
        model: claude-sonnet-4-20250514
        supports_tools: true
        supports_streaming: true
      local-qwen:
        type: openai-compatible
        enabled: true
        endpoint: http://localhost:8000/v1
        api_key_env: LOCAL_LLM_KEY
        model: qwen
        supports_tools: true
        is_local: true

    routing:
      strategy: auto
      allow_paid_models: false

Common credentials include OPENROUTER_API_KEY, OPENAI_API_KEY,
ANTHROPIC_API_KEY, GEMINI_API_KEY, NVIDIA_API_KEY, and GROQ_API_KEY.
Ollama can run without a key. allow_paid_models remains opt-in; use free or
local models unless paid model selection is explicitly enabled.

Capabilities are tri-state. A missing capability declaration means unknown,
not supported or unsupported. For adapters whose catalog does not report
capabilities, declare them for configured models in the project configuration.
The router only selects a candidate when its required capabilities are
affirmatively known.

## Adapters

The factory currently constructs OpenRouter, OpenAI, Anthropic, Gemini,
NVIDIA, Groq, Ollama, 9router, LiteLLM, and named OpenAI-compatible endpoint
adapters. Catalog discovery is adapter-specific; Anthropic and configured
compatible endpoints can use static model metadata. OpenRouter's catalog is
discovered at runtime, and its dynamic free route is provided as a normalized
routing hint by the OpenRouter adapter.

There is no requirement that every adapter implement the same features. A
provider may return unknown capabilities or an empty catalog. This does not
make its model eligible for a request that requires an unknown capability.

## Routing and failover

The router scores normalized ModelInfo candidates and filters by task
requirements, health, free/local mode, and the explicit paid-model policy.
FallbackEngine consumes provider-neutral error categories. Temporary
availability, rate-limit, network, and timeout failures can use bounded
failover. Authentication, invalid-request, payment, and tool errors do not
cause open-ended model rotation. Calls retain the same request messages and
task context across candidate attempts.

OpenRouter is the default adapter when a project has no explicit provider
configuration. It is not a dependency of the agent, planner, scheduler, or
router.

Useful commands: harness providers list; harness providers configure <name>;
harness models list; harness doctor.

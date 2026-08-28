# Providers

Harness supports multiple AI model providers. You need at least one.

## OpenRouter (Recommended)

Access 300+ models including free tiers.

### Setup

1. Create an account at [openrouter.ai](https://openrouter.ai)
2. Get an API key at [openrouter.ai/keys](https://openrouter.ai/keys)
3. Set the environment variable:

```bash
# Linux/macOS
export OPENROUTER_API_KEY="sk-or-v1-..."

# Windows PowerShell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

### Free Models

Harness automatically discovers free models. Run:

```bash
harness models list --free
```

## Ollama (Local)

Run AI models locally — no API key, no data leaves your machine.

### Setup

1. Install Ollama: [ollama.com/download](https://ollama.com/download)
2. Start the server: `ollama serve`
3. Pull a model: `ollama pull codellama`

### Usage

```bash
harness models local              # List local models
harness run --model codellama "..."  # Use a specific model
```

## LiteLLM

Unified API for 100+ LLM providers.

### Setup

```bash
export LITELLM_API_KEY="your-key"
```

## Verifying Configuration

```bash
harness doctor       # Check all providers
harness providers list  # See provider status
```

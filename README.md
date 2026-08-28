# Harness Engineering CLI

A model-agnostic, autonomous software-engineering harness.

## What is this?

Harness Engineering CLI is not another AI coding chatbot. It's an engineering harness that:

- **Plans** before implementing
- **Inspects** the repository
- **Executes** tools to modify code
- **Verifies** changes work
- **Recovers** from failures
- **Delivers** verified results

The LLM is a replaceable reasoning component. The harness is the product.

## Installation

```bash
# Clone the repository
git clone https://github.com/kunalsanga/Harness-CLI.git
cd Harness-CLI

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

```bash
# Set up your API key
cp .env.example .env
# Edit .env and add your OpenRouter API key

# Initialize a project
harness init

# Run an engineering task
harness run "Fix all failing tests without modifying the tests"

# Check system health
harness doctor

# List available models
harness models
```

## Configuration

Harness reads API keys from environment variables:

| Variable | Purpose | Required |
|----------|---------|----------|
| `OPENROUTER_API_KEY` | OpenRouter multi-model gateway | Yes (for cloud models) |
| `LITELLM_API_KEY` | LiteLLM unified abstraction | No |
| `OLLAMA_HOST` | Ollama local inference | No (default: localhost:11434) |

**Never commit API keys or `.env` files to version control.**

## Architecture

```
GOAL
↓
UNDERSTAND
↓
PLAN
↓
CONTEXT
↓
SELECT MODEL
↓
EXECUTE
↓
OBSERVE
↓
VERIFY
↓
EVALUATION
↓
SUCCESS?
├── YES → DELIVER
└── NO  → DIAGNOSE → REPLAN → EXECUTE
```

## Core Principles

1. **Model Independence** — Use any model from any provider
2. **Verification First** — Success is evidence-based
3. **Recovery** — Failures are part of the loop
4. **Security** — Permissions and sandboxing
5. **Observability** — Every run is traceable

## Development

```bash
# Run all tests
uv run pytest tests/ -v

# Run benchmarks
uv run pytest tests/benchmarks/ -v -s

# Run with free models
harness run --mode free "Your task here"

# Run with budget limits
harness run --max-cost 0.50 --max-iterations 5 "Your task here"
```

## Documentation

- [Vision](docs/vision.md)
- [Architecture](docs/architecture.md)
- [Competitive Analysis](docs/competitive-analysis.md)
- [Roadmap](docs/roadmap.md)
- [Security Model](docs/security-model.md)
- [Model Routing](docs/model-routing.md)
- [Performance](docs/performance.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

## License

License decision required before public release.

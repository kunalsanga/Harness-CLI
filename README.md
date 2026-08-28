# Harness Engineering CLI

A model-agnostic, autonomous AI software-engineering agent that runs in your terminal.

## What is Harness?

Harness is not another AI chatbot. It's an engineering harness that:

- **Plans** before implementing
- **Inspects** your repository
- **Executes** tools to modify code
- **Verifies** changes work
- **Recovers** from failures
- **Delivers** verified results

The LLM is a replaceable reasoning component. The harness is the product.

## Installation

```bash
pip install harness-engineering
```

Or from source:

```bash
git clone https://github.com/kunalsanga/Harness-CLI.git
cd Harness-CLI
pip install -e .
```

## Quick Start

```bash
# 1. Setup a provider (OpenRouter recommended — free models available)
export OPENROUTER_API_KEY="sk-or-v1-..."
harness setup

# 2. Initialize a project
cd your-project
harness init

# 3. Run a task
harness run "Fix the failing tests"

# 4. Resume later
harness session list
harness session resume <id>
```

## Supported Providers

| Provider | API Key | Free Models | Local |
|----------|---------|-------------|-------|
| [OpenRouter](https://openrouter.ai) | Required | 20+ free models | No |
| [Ollama](https://ollama.com) | None | All local | Yes |
| [LiteLLM](https://litellm.ai) | Required | Depends on backend | No |

### Free Models

Harness supports free models out of the box:

```bash
harness models list --free
harness models recommend --task "Fix failing tests"
```

### Local Models (Ollama)

```bash
ollama serve
ollama pull codellama
harness models local
harness run --model codellama "Fix the tests"
```

## Commands

```bash
harness --help                    # General help
harness setup                     # Interactive setup wizard
harness doctor                    # System health check
harness init                      # Initialize a project
harness run "task description"    # Run a coding task

harness providers list            # List providers
harness providers configure openrouter  # Setup provider

harness models list               # List all models
harness models list --free        # Free models only
harness models recommend --task "..."  # Best model for a task
harness models inspect <model>    # Model details
harness models compare <a> <b>    # Compare models

harness session list              # Recent sessions
harness session show <id>         # Session details
harness session resume <id>       # Resume a session

harness agents list               # Available agents
harness agents inspect <name>     # Agent details

harness tools list                # Available tools
harness plugin list               # Installed plugins
harness mcp list                  # MCP servers
harness hooks list                # Lifecycle hooks
```

## Multi-Agent Orchestration

Harness can decompose complex tasks into specialized agents:

```bash
harness run --mode multi-agent "Implement auth, add tests, review changes"
```

Available agents: Planner, Researcher, Analyzer, Coder, Tester, Reviewer, Debugger.

## Architecture

```
USER → ORCHESTRATOR → SPECIALIZED AGENTS → MODEL ROUTER → TOOLS → VERIFICATION → RESULT
                            ↓
                    SESSION PERSISTENCE
```

- **Model-agnostic**: Works with OpenRouter, Ollama, LiteLLM, or any OpenAI-compatible API
- **Persistent sessions**: Resume work across interruptions
- **Empirical intelligence**: Learns which models work best for which tasks
- **Verification-first**: Always runs tests/checks before reporting success
- **Secure**: Conservative defaults, permission controls, no source upload

## Configuration

```yaml
# .harness/config.yaml
routing:
  strategy: auto
  prefer_free: true

budgets:
  max_cost_per_task: 1.0
  max_tool_calls: 100
```

See [Configuration Guide](docs/configuration.md) for full reference.

## Platform Support

| Platform | Status |
|----------|--------|
| Windows (PowerShell/CMD) | First-class |
| Linux | First-class |
| macOS | First-class |

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -q

# Build package
python -m build
```

## Security

- API keys stored in environment variables, never in source code
- Conservative default permissions
- Workspace boundary enforcement
- No telemetry or source upload by default
- See [Security Model](docs/security-model.md)

## License

MIT License — see [LICENSE](LICENSE)

## Contributing

See [Development Guide](docs/development.md) for setup instructions.

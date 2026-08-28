# Contributing to Harness Engineering CLI

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/kunalsanga/Harness-CLI.git
cd Harness-CLI

# Install dependencies
uv sync

# Copy environment template
cp .env.example .env

# Add your OpenRouter API key to .env
# OPENROUTER_API_KEY=your_key_here

# Run tests
uv run pytest tests/ -v
```

## Running Tests

```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run benchmarks
uv run pytest tests/benchmarks/ -v -s

# Run full test suite
uv run pytest tests/ -v
```

## Code Style

- Python 3.12+
- Type hints on all public functions
- Docstrings for public APIs
- Follow existing code patterns in the project

## Adding a New Provider

1. Create a new file in `src/harness_core/providers/`
2. Implement the `ModelProvider` interface from `base.py`
3. Register it in the CLI's provider initialization
4. Add tests in `tests/unit/`

## Adding a New Tool

1. Create or extend files in `src/harness_core/tools/`
2. Implement the `Tool` interface from `base.py`
3. Define a `ToolSchema` for the tool
4. Add permission rules if needed
5. Add tests

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `uv run pytest tests/ -v`
6. Submit a pull request

## Reporting Issues

- Use GitHub Issues for bug reports
- Include steps to reproduce
- Include your Python version and OS
- Include test output if relevant

## Security

See [SECURITY.md](SECURITY.md) for reporting security vulnerabilities.

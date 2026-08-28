# Quick Start

## 1. Install

```bash
pip install harness-engineering
```

## 2. Setup

```bash
harness setup
```

Or configure a provider manually:

```bash
# OpenRouter (recommended — free models available)
export OPENROUTER_API_KEY="sk-or-v1-..."

# Or Ollama (local, no key needed)
ollama serve
ollama pull codellama
```

## 3. Initialize a project

```bash
cd your-project
harness init
```

## 4. Run a task

```bash
harness run "Fix the failing tests"
```

## 5. Review changes

Harness will:
1. Inspect your repository
2. Classify the task
3. Select the best model
4. Execute tools to make changes
5. Run verification
6. Report results

## 6. Resume later

```bash
harness session list
harness session resume <session-id>
```

## Useful Commands

```bash
harness doctor                    # Check system health
harness models list --free        # See free models
harness models recommend --task "..."  # Get model recommendation
harness agents list               # See available agents
harness session list              # See recent sessions
```

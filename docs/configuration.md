# Configuration

## Configuration Hierarchy

Configuration is resolved in this order (highest wins):

1. **CLI flags** — `--model xyz`
2. **Session config** — per-session overrides
3. **Project config** — `.harness/config.yaml`
4. **User config** — `~/.harness/config.yaml`
5. **Built-in defaults**

## Project Configuration

Created by `harness init` in `.harness/config.yaml`:

```yaml
routing:
  strategy: auto
  prefer_free: true
  fallback: true

budgets:
  max_cost_per_task: 1.0
  max_tool_calls: 100
  max_iterations: 30

permissions:
  bash: ask
  edit: allow
  network: ask
  git_push: ask
```

## Environment Variables

Override any config with environment variables:

```bash
HARNESS_MODEL_DEFAULT=auto
HARNESS_ROUTING_MODE=auto
HARNESS_AGENT_MAX_AGENTS=8
```

## Sensitive Values

Harness automatically redacts sensitive values when displaying configuration:
- API keys
- Tokens
- Passwords
- Credentials

## Validation

```bash
harness config show      # Show effective config
harness config validate  # Check for errors
```

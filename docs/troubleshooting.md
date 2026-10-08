# Troubleshooting

> Current task execution uses OpenRouter. Ollama is not an active fallback in
> the normal interactive runtime.

## Common Issues

### "No model provider configured"

**Cause:** The OpenRouter API key is missing or the service cannot be reached.

**Fix:**
```bash
# Option 1: OpenRouter
export OPENROUTER_API_KEY="sk-or-v1-..."
harness doctor

```

### "Rate limited (429)"

**Cause:** Too many requests to the provider.

**Fix:**
- Wait a minute and retry
- Use a different model
- Use the configured free-model failover; local providers are not activated by
  the current normal task runtime.

### "Provider connection failed"

**Cause:** Network issue or provider down.

**Fix:**
- Check your internet connection
- Verify the API key is correct
- Try: `harness doctor`

### "Permission denied"

**Cause:** Harness security policy blocked the operation.

**Fix:**
- This is expected for dangerous operations
- Review what the agent tried to do
- Adjust permissions in `.harness/config.yaml` if needed

### "Tests failing after agent run"

**Cause:** Agent changes introduced a regression.

**Fix:**
- Review the changes: `git diff`
- Run tests manually: `pytest` (or your test command)
- Resume the session: `harness session resume <id>`
- The agent can debug and fix its own failures

### Windows-specific issues

**Unicode errors:** Harness forces UTF-8 on Windows. If you see encoding errors, ensure your terminal supports UTF-8.

**Path issues:** Harness uses forward slashes internally. This is handled automatically.

## Getting Help

```bash
harness --help            # General help
harness doctor            # System health check
harness status            # Current status
```

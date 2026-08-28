# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Harness Engineering CLI, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email security concerns to the project maintainer.

## What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Timeline

- Acknowledgment: within 48 hours
- Initial assessment: within 1 week
- Fix timeline: depends on severity

## Security Considerations

### API Keys

- Never commit `.env` files or API keys to version control
- Use environment variables for all credentials
- The `.env.example` file contains placeholders only

### Agent Permissions

- Harness uses a permission system (allow/ask/deny)
- Destructive operations require explicit approval
- The workspace sandbox limits file access

### Model Providers

- API keys are passed via environment variables
- Keys are never sent to the model as part of prompts
- Provider communication uses standard HTTPS

### Dependencies

- Dependencies are pinned in `uv.lock`
- Run `uv lock --upgrade` periodically to get security updates
- Report any dependency vulnerabilities through the same process

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |

## Scope

This security policy covers the Harness Engineering CLI codebase itself. Issues with third-party model providers (OpenRouter, Ollama, etc.) should be reported to those providers directly.

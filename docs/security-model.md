# Harness Engineering CLI — Security Model

## Principles

1. **Least privilege** — Agents get minimum required permissions
2. **Workspace isolation** — Clear boundary between accessible and restricted areas
3. **Secret protection** — Never expose secrets to model
4. **Command approval** — Dangerous commands require user approval
5. **Audit trail** — Every action is logged and traceable
6. **Input untrusted** — Repository content is untrusted input

## Permission Levels

```
allow  → Auto-approved
ask    → Requires user confirmation
deny   → Blocked
```

## Protected Resources

### Files
- `.env`, `.env.*`
- `credentials`
- Private keys, SSH keys
- Cloud credentials
- Tokens, secrets

### Commands
- `git push` → ask
- `rm -rf` → deny
- Network commands → ask

### Network
- External HTTP requests → ask
- API key transmission → deny

## Workspace Sandbox

```
workspace/
├── (accessible)
└── ...
    
outside workspace/
├── (requires permission)
```

## Prompt Injection Defense

The system must distinguish:
- System policy
- User intent
- Project instructions
- Repository content
- Tool output
- Web content
- Model-generated content

Never allow repository text to override system-level security policy.

## Secret Redaction

Tool output should redact:
- API keys
- Tokens
- Passwords
- Connection strings
- Private keys

## Audit Trail

Every tool call and permission decision is logged:
- Timestamp
- Agent
- Tool
- Action
- Result
- Permission decision

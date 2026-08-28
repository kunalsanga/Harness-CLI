# M7 — Universal Developer Platform & Extensibility

## Test Results

```
706 passed in 45.07s
```

| Metric | Before M7 | After M7 |
|--------|-----------|----------|
| Tests | 649 | **706** (+57) |

## What Was Implemented

### Extension Architecture (`extensions/`)
- `ExtensionManifest` — metadata, types (TOOL/PROVIDER/AGENT/HOOK/MCP), state lifecycle
- `ExtensionRegistry` — register/unregister, lifecycle states, statistics
- `ExtensionLoader` — discovers YAML/JSON/Python manifests, loads modules safely
- `ExtensionContext` — controlled API surface for extensions (tools, providers, agents, hooks, events)

### Plugin System (`plugins/`)
- `PluginManager` — install, enable, disable, remove, inspect, list, load
- Global plugin directory (`~/.harness/plugins/`)
- Plugin discovery, manifest validation, safe error isolation
- CLI: `harness plugin list|install|enable|disable|remove|inspect`

### MCP Foundation (`mcp/`)
- `MCPClient` — stdio-based JSON-RPC communication with MCP servers
- Server lifecycle: start, initialize, discover tools, call tools, shutdown
- Tool discovery and invocation
- CLI: `harness mcp list`

### Hook System (`hooks/`)
- `HookRegistry` — 12 lifecycle events (before/after run, agent, tool, model, etc.)
- Priority ordering, error isolation, rejection support
- Plugin-source tracking
- CLI: `harness hooks list`

### Configuration System (`config/`)
- `HarnessConfig` — 5-level precedence (CLI > Session > Project > Global > Defaults)
- Environment variable override with type coercion
- Secret redaction in display
- Validation
- CLI: `harness config show|get|set|validate`

### Sample Plugins (`examples/plugins/`)
1. **hello-world** — minimal tool plugin example
2. **code-quality** — static analysis tool plugin
3. **security-reviewer** — custom agent plugin

### CLI Commands Added
| Command | Description |
|---------|-------------|
| `harness plugin list` | List installed plugins |
| `harness plugin install <path>` | Install a plugin |
| `harness plugin enable/disable/remove` | Manage plugins |
| `harness plugin inspect <name>` | Show plugin details |
| `harness tools list` | List available tools |
| `harness tools inspect <name>` | Show tool details |
| `harness mcp list` | List MCP servers |
| `harness hooks list` | List registered hooks |

## Files Created (12)
| File | Purpose |
|------|---------|
| `extensions/__init__.py` | Package exports |
| `extensions/manifest.py` | Manifest model, types, validation |
| `extensions/registry.py` | Extension registry |
| `extensions/loader.py` | Discovery and loading |
| `extensions/context.py` | Extension context API |
| `plugins/__init__.py` | Package exports |
| `plugins/manager.py` | Plugin lifecycle management |
| `mcp/__init__.py` | Package exports |
| `mcp/client.py` | MCP client (stdio JSON-RPC) |
| `hooks/__init__.py` | Package exports |
| `hooks/hooks.py` | Hook registry and execution |
| `config/__init__.py` | Package exports |
| `config/config.py` | Hierarchical configuration |

## Files Modified (2)
| File | Changes |
|------|---------|
| `cli/main.py` | Added plugin/tools/mcp/hooks CLI commands |
| `tests/unit/test_m7_extensions.py` | 56 comprehensive tests |

## Files Created (examples)
| File | Purpose |
|------|---------|
| `examples/plugins/hello-world/` | Sample tool plugin |
| `examples/plugins/code-quality/` | Sample code analysis plugin |
| `examples/plugins/custom-agent/` | Sample security reviewer agent |

## Known Limitations
1. Plugins run in-process (not sandboxed)
2. MCP only supports stdio transport
3. Plugin dependencies not isolated (share Python environment)
4. No cloud marketplace
5. No vector memory in extensions

## Recommended M8
- MCP HTTP/SSE transport
- Plugin sandboxing
- Extension marketplace
- Cloud sync

# Changelog

All notable changes to Harness Engineering CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2026-08-29

### Added

#### Core Engine (M1)
- Autonomous agent engine with tool execution loop
- Read/write/edit filesystem tools
- Glob and grep search tools
- Shell command execution with safety controls
- Git integration (status, diff, log)
- File listing and navigation

#### Model Routing (M2)
- Model-agnostic routing across providers
- OpenRouter provider with full API integration
- Ollama local model provider
- Automatic fallback chains (429, 500, timeout, invalid response)
- Budget controls (iterations, tool calls, cost)

#### Context Intelligence (M3)
- Task classification with confidence scoring
- 14-dimension task-aware model scoring
- Context construction engine
- Task-specific model routing
- Performance measurement

#### Empirical Model Intelligence (M3.8)
- SQLite-backed execution history
- Task-specific model performance tracking
- Sample-size confidence calculation
- Time-decay for historical performance
- Evidence provenance (declared vs measured vs observed)
- Routing explanations with reasoning

#### Session Intelligence (M4)
- Persistent sessions across interruptions
- Run tracking with status lifecycle
- Checkpointing at safe boundaries
- Session memory (decisions, discoveries, constraints, TODOs, warnings, errors, solutions)
- Session export (JSON, Markdown)
- Session diff and status

#### Advanced Agent Engine (M5)
- 8 specialized agent roles (Orchestrator, Planner, Researcher, Analyzer, Coder, Tester, Reviewer, Debugger)
- Task graph with dependency resolution
- Multi-agent orchestration
- Agent budgets and cost controls
- Repair cycle management
- Structured agent results

#### Native Performance (M6)
- Rust native filesystem foundation (gitignore-aware search, grep, file indexing, hashing)
- Python fallback when native unavailable
- Dependency-aware parallel scheduler
- File ownership tracking for concurrent edits
- Cancellation and graceful shutdown
- Performance benchmarking suite

#### Developer Platform (M7)
- Extension architecture (Tool, Provider, Agent, Hook, MCP types)
- Plugin system with manifest validation
- MCP client (stdio JSON-RPC)
- Lifecycle hooks (12 events with priority ordering)
- Hierarchical configuration (5-level precedence)
- Sample plugins (hello-world, code-quality, security-reviewer)

#### Productization (M8)
- MIT License
- Proper Python packaging (pip install, uv tool install)
- Provider setup CLI
- Interactive setup flow
- User-facing error system with helpful messages
- First-run experience
- Free model workflow
- Session resume UX
- Comprehensive documentation
- CHANGELOG

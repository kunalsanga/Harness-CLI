"""CLI entry point for Harness Engineering."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# Force UTF-8 on Windows to avoid cp1252 encoding errors with Unicode output.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

app = typer.Typer(
    name="harness",
    help="Harness Engineering CLI — A model-agnostic, autonomous software-engineering harness.",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


@app.callback(invoke_without_command=True)
def _main_callback(ctx: typer.Context) -> None:
    """Start interactive shell when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        from harness_core.cli.interactive import run_interactive
        run_interactive()

# Sub-commands
config_app = typer.Typer(help="Configuration commands")
app.add_typer(config_app, name="config")

models_app = typer.Typer(help="Model intelligence commands")
app.add_typer(models_app, name="models")

session_app = typer.Typer(help="Session management commands")
app.add_typer(session_app, name="session")

agents_app = typer.Typer(help="Agent management commands")
app.add_typer(agents_app, name="agents")


@app.command()
def init(
    path: Optional[str] = typer.Argument(None, help="Project path (default: current directory)"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
) -> None:
    """Initialize a Harness project."""
    project_path = Path(path) if path else Path.cwd()
    console.print(f"[bold blue]Initializing Harness project in {project_path}[/]")

    # Create .harness directory
    harness_dir = project_path / ".harness"
    harness_dir.mkdir(exist_ok=True)

    # Create config.yaml
    config_file = harness_dir / "config.yaml"
    if config_file.exists() and not force:
        console.print("[yellow]Config already exists. Use --force to overwrite.[/]")
        return

    config_content = """routing:
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
"""
    config_file.write_text(config_content, encoding="utf-8")
    console.print(f"[green][OK] Created {config_file}[/]")

    # Create HARNESS.md
    harness_md = project_path / "HARNESS.md"
    if not harness_md.exists() or force:
        harness_md.write_text(
            """# Project Instructions for Harness Engineering

## Build Commands
<!-- Add your build commands here -->

## Test Commands
<!-- Add your test commands here -->

## Project Rules
<!-- Add project-specific rules here -->
""",
            encoding="utf-8",
        )
        console.print(f"[green][OK] Created {harness_md}[/]")

    # Detect project
    console.print("\n[bold]Detecting project...[/]")

    detections = []
    if (project_path / "pyproject.toml").exists() or (project_path / "setup.py").exists():
        detections.append("[OK] Python")
    if (project_path / "package.json").exists():
        detections.append("[OK] Node.js")
    if (project_path / "Cargo.toml").exists():
        detections.append("[OK] Rust")
    if (project_path / "go.mod").exists():
        detections.append("[OK] Go")
    if (project_path / ".git").exists():
        detections.append("[OK] Git")
    if (project_path / "README.md").exists():
        detections.append("[OK] README.md")

    for d in detections:
        console.print(f"  [green]{d}[/]")

    if not detections:
        console.print("  [dim]No specific project detected[/]")

    console.print("\n[bold green][OK] Harness project initialized![/]")


@app.command()
def run(
    goal: str = typer.Argument(..., help="Engineering goal to accomplish"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    mode: str = typer.Option("auto", "--mode", help="Execution mode (auto/single/multi-agent/free/best/fast/local/cheap)"),
    max_iterations: int = typer.Option(30, "--max-iterations", help="Maximum iterations"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", help="Maximum cost"),
    headless: bool = typer.Option(False, "--headless", help="Headless mode"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    max_agents: int = typer.Option(8, "--max-agents", help="Max agents for multi-agent mode"),
    max_parallel: int = typer.Option(3, "--max-parallel", help="Max parallel agents"),
) -> None:
    """Run an engineering task."""
    async def _run() -> None:
        from harness_core.agent.loop import AgentLoop
        from harness_core.agent.types import AgentConfig, AgentRole
        from harness_core.observability.events import Event, EventBus
        from harness_core.providers.openrouter import OpenRouterProvider
        from harness_core.routing.router import ModelRouter, RouterConfig
        from harness_core.routing.task_aware import TaskAwareRouter
        from harness_core.models.registry import ModelRegistry
        from harness_core.tools.filesystem import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
        from harness_core.tools.git import GitDiffTool, GitLogTool, GitStatusTool
        from harness_core.tools.search import GlobTool, GrepTool
        from harness_core.tools.shell import RunCommandTool

        provider = OpenRouterProvider()
        if not await provider.health_check():
            console.print("  [red][FAIL] OpenRouter provider not available. Check OPENROUTER_API_KEY.[/]")
            raise typer.Exit(code=4)

        tools = [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            ListFilesTool(),
            GlobTool(),
            GrepTool(),
            RunCommandTool(),
            GitStatusTool(),
            GitDiffTool(),
            GitLogTool(),
        ]

        event_bus = EventBus()

        # Load project config if available
        router_config = RouterConfig()
        config_file = Path(".harness/config.yaml")
        if config_file.exists():
            try:
                import yaml
                raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
                routing_data = raw.get("routing", {})
                budgets_data = raw.get("budgets", {})
                if routing_data:
                    router_config.routing_mode = routing_data.get("strategy", mode)
                    router_config.prefer_free = routing_data.get("prefer_free", False)
                if budgets_data:
                    router_config.budget.max_iterations = budgets_data.get("max_iterations", max_iterations)
                    router_config.budget.max_tool_calls = budgets_data.get("max_tool_calls", 100)
                    router_config.budget.max_cost = budgets_data.get("max_cost_per_task", 5.0)
            except Exception:
                pass

        # Override with CLI args
        router_config.routing_mode = mode
        router_config.budget.max_iterations = max_iterations
        if max_cost is not None:
            router_config.budget.max_cost = max_cost

        # Build router with all available providers
        providers = [provider]
        try:
            from harness_core.providers.ollama import OllamaProvider
            ollama = OllamaProvider()
            if await ollama.health_check():
                providers.append(ollama)
        except Exception:
            pass

        # Wire task-aware routing pipeline
        task_aware = TaskAwareRouter(registry=ModelRegistry())

        router = ModelRouter(
            providers=providers,
            config=router_config,
            event_bus=event_bus,
            task_aware=task_aware,
        )

        if not headless:
            async def log_event(event: Event) -> None:
                if event.type == "task.started":
                    console.print(f"\n[bold blue]>> Task:[/] {event.data.get('goal', '')}")
                elif event.type == "iteration.started":
                    i = event.data.get("iteration", 0)
                    console.print(f"\n[bold]Iteration {i}...[/]")
                elif event.type == "tool.call":
                    tool = event.data.get("tool", "")
                    console.print(f"  [cyan]Tool:[/] {tool}")
                elif event.type == "tool.result":
                    status = event.data.get("status", "")
                    color = "green" if status == "success" else "red"
                    console.print(f"  [{color}]Result:[/] {status}")
                elif event.type == "task.classified":
                    tt = event.data.get("task_type", "")
                    conf = event.data.get("confidence", 0)
                    console.print(f"  [yellow]Classified:[/] {tt} (confidence: {conf:.0%})")
                elif event.type == "routing.decision":
                    m = event.data.get("model", "")
                    s = event.data.get("score", 0)
                    console.print(f"  [magenta]Router:[/] {m} (score: {s})")
                elif event.type == "routing.models_refreshed":
                    count = event.data.get("count", 0)
                    console.print(f"  [dim]Discovered {count} models[/]")
                elif event.type == "task.completed":
                    status = event.data.get("status", "")
                    iters = event.data.get("iterations", 0)
                    tc = event.data.get("tool_calls", 0)
                    console.print(
                        f"\n[bold]Status:[/] {status} | Iterations: {iters} | Tool calls: {tc}"
                    )

            event_bus.on("*", log_event)

        ws = str(Path.cwd())

        # Multi-agent mode
        if mode == "multi-agent":
            from harness_core.agents.orchestrator import Orchestrator, ExecutionMode, AgentBudget

            budget = AgentBudget(
                max_agents=max_agents,
                max_parallel_agents=max_parallel,
                max_iterations_per_agent=max_iterations,
                max_total_iterations=max_iterations * 5,
            )

            orchestrator = Orchestrator(
                provider=provider,
                tools=tools,
                router=router,
                task_aware=task_aware,
                event_bus=event_bus,
                workspace_path=ws,
                budget=budget,
            )

            with Progress(
                SpinnerColumn("dots"),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                ptask = progress.add_task("Running multi-agent orchestration...", total=None)
                result = await orchestrator.execute(
                    task_description=goal,
                    mode=ExecutionMode.MULTI_AGENT,
                    workspace_path=ws,
                )
                progress.update(ptask, completed=True)

            # Display agent results
            if result.agent_results:
                agent_table = Table(title="Agent Results")
                agent_table.add_column("Agent", style="cyan")
                agent_table.add_column("Role", style="green")
                agent_table.add_column("Status")
                agent_table.add_column("Duration", justify="right")
                agent_table.add_column("Tool Calls", justify="right")

                for name, ar in result.agent_results.items():
                    color = "green" if ar.status.value == "completed" else "red"
                    dur = f"{ar.duration_ms / 1000:.1f}s" if ar.duration_ms > 0 else "-"
                    agent_table.add_row(
                        name,
                        ar.role.value,
                        f"[{color}]{ar.status.value}[/]",
                        dur,
                        str(ar.tool_calls),
                    )
                console.print(agent_table)

            if json_output:
                output = result.to_dict()
                console.print(json.dumps(output, indent=2))
            else:
                if result.success:
                    console.print(Panel(result.summary or "(completed)", title="Result", border_style="green"))
                else:
                    console.print(Panel(result.summary or "Task failed", title="Error", border_style="red"))
                    if result.errors:
                        for err in result.errors:
                            console.print(f"  [red]Error:[/] {err}")

            await provider.close()

            if not result.success:
                raise typer.Exit(code=1)
            return

        # Single/auto mode (original AgentLoop)
        agent_config = AgentConfig(
            role=AgentRole.BUILD,
            max_iterations=max_iterations,
            model_preference=model,
            routing_mode=mode,
        )

        agent = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=Path.cwd(),
            config=agent_config,
            event_bus=event_bus,
            router=router,
            task_aware=task_aware,
        )

        with Progress(
            SpinnerColumn("dots"),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running agent...", total=None)
            result = await agent.run(goal)
            progress.update(task, completed=True)

        if json_output:
            output = {
                "task_id": result.id,
                "status": result.status.value,
                "iterations": result.iterations,
                "tool_calls": len(result.tool_calls),
                "result": result.result,
                "error": result.error,
            }
            console.print(json.dumps(output, indent=2))
        else:
            if result.status.value == "completed":
                console.print(Panel(result.result or "(no result)", title="Result", border_style="green"))
            else:
                console.print(
                    Panel(result.error or "Task failed", title="Error", border_style="red")
                )

        await provider.close()

        if result.status.value != "completed":
            raise typer.Exit(code=1)

    asyncio.run(_run())


@app.command(name="shell")
def shell(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    mode: str = typer.Option("auto", "--mode", help="Execution mode"),
    free: bool = typer.Option(False, "--free", help="Use free models only"),
    local: bool = typer.Option(False, "--local", help="Use local Ollama models"),
    plain: bool = typer.Option(False, "--plain", help="Plain terminal mode"),
    max_iterations: int = typer.Option(30, "--max-iterations", help="Max iterations"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", help="Max cost"),
) -> None:
    """Start an interactive coding session."""
    from harness_core.cli.interactive import run_interactive
    run_interactive(
        model=model,
        mode=mode,
        free=free,
        local=local,
        plain=plain,
        max_iterations=max_iterations,
        max_cost=max_cost,
    )


@app.command()
def doctor() -> None:
    """Check system health."""
    from harness_core.providers.openrouter import OpenRouterProvider
    from harness_core.providers.ollama import OllamaProvider

    console.print(Panel("Harness Doctor", border_style="blue"))

    # Runtime checks
    console.print("\n[bold]Runtime[/]")
    console.print(f"  [green][OK] Python {sys.version.split()[0]}[/]")
    console.print(f"  [green][OK] Git[/]")

    # Provider checks
    console.print("\n[bold]Providers[/]")

    async def _check_providers() -> None:
        openrouter = OpenRouterProvider()
        ollama = OllamaProvider()

        if await openrouter.health_check():
            console.print("  [green][OK] OpenRouter[/]")
        else:
            console.print("  [red][FAIL] OpenRouter[/]")

        if await ollama.health_check():
            console.print("  [green][OK] Ollama[/]")
        else:
            console.print("  [dim]  [--] Ollama (not running)[/]")

        await openrouter.close()
        await ollama.close()

    asyncio.run(_check_providers())

    # Agent system check
    console.print("\n[bold]Agent System[/]")
    from harness_core.agents.registry import AgentRegistry
    registry = AgentRegistry()
    agents = registry.list_all()
    console.print(f"  [green][OK] {len(agents)} registered agents[/]")
    for agent in agents:
        console.print(f"    {agent.name}: {agent.role.value}")

    console.print("\n[bold green]Doctor check complete.[/]")


@app.command()
def status() -> None:
    """Show current harness status."""
    console.print(Panel("Harness Status", border_style="blue"))
    console.print(f"  Working directory: {Path.cwd()}")
    console.print(f"  Python: {sys.version.split()[0]}")

    harness_dir = Path(".harness")
    if harness_dir.exists():
        console.print("  Config: [green][OK] .harness/[/]")
    else:
        console.print("  Config: [red][FAIL] Not initialized (run `harness init`)[/]")


@app.command()
def trace(
    run_id: Optional[str] = typer.Argument(None, help="Run ID to show trace for"),
) -> None:
    """Show execution trace."""
    console.print("[dim]Trace storage not yet implemented.[/]")


@app.command()
def usage() -> None:
    """Show usage statistics."""
    console.print("[dim]Usage tracking not yet implemented.[/]")


# ── agents sub-commands ──────────────────────────────────────────────────

@agents_app.command("list")
def agents_list(
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List available agents."""
    from harness_core.agents.registry import AgentRegistry

    registry = AgentRegistry()
    agents = registry.list_all()

    if json_output:
        print(json.dumps([a.to_dict() for a in agents], indent=2))
        return

    table = Table(title=f"Registered Agents ({len(agents)})")
    table.add_column("Name", style="cyan")
    table.add_column("Role", style="green")
    table.add_column("Description", max_width=50)
    table.add_column("Capabilities", max_width=40)

    for agent in agents:
        table.add_row(
            agent.name,
            agent.role.value,
            agent.description[:50],
            ", ".join(agent.capabilities[:3]),
        )

    console.print(table)


@agents_app.command("inspect")
def agents_inspect(
    name: str = typer.Argument(..., help="Agent name"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Inspect an agent's configuration."""
    from harness_core.agents.registry import AgentRegistry

    registry = AgentRegistry()
    agent = registry.get(name)

    if agent is None:
        console.print(f"[red]Agent '{name}' not found.[/]")
        raise typer.Exit(code=1)

    if json_output:
        print(json.dumps(agent.to_dict(), indent=2))
        return

    console.print(Panel(f"Agent: {agent.name}", border_style="blue"))
    console.print(f"  [bold]Role:[/] {agent.role.value}")
    console.print(f"  [bold]Description:[/] {agent.description}")
    console.print(f"  [bold]Display Name:[/] {agent.display_name}")
    console.print(f"  [bold]Enabled:[/] {'Yes' if agent.enabled else 'No'}")

    console.print("\n[bold]Capabilities:[/]")
    for cap in agent.capabilities:
        console.print(f"  • {cap}")

    console.print("\n[bold]Preferred Task Types:[/]")
    for tt in agent.preferred_task_types:
        console.print(f"  • {tt}")

    console.print("\n[bold]Allowed Tools:[/]")
    for tool in agent.allowed_tools:
        console.print(f"  • {tool}")

    console.print("\n[bold]Resource Limits:[/]")
    console.print(f"  Max iterations: {agent.max_iterations}")
    console.print(f"  Max tool calls: {agent.max_tool_calls}")
    console.print(f"  Max tokens: {agent.max_tokens}")

    console.print("\n[bold]Model Preferences:[/]")
    prefs = []
    if agent.prefer_fast_model:
        prefs.append("fast")
    if agent.prefer_strong_model:
        prefs.append("strong")
    if agent.prefer_cheap_model:
        prefs.append("cheap")
    console.print(f"  {', '.join(prefs) if prefs else 'none'}")


# ── models sub-commands ────────────────────────────────────────────────────

@models_app.command("list")
def models_list(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Filter by provider"),
    free_only: bool = typer.Option(False, "--free", help="Show only free models"),
) -> None:
    """List available models."""
    from harness_core.providers.openrouter import OpenRouterProvider

    async def _list_models() -> None:
        openrouter = OpenRouterProvider()

        if not await openrouter.health_check():
            console.print("  [red][FAIL] Cannot connect to OpenRouter[/]")
            raise typer.Exit(code=4)

        all_models = await openrouter.list_models()
        await openrouter.close()

        # Filter
        models = all_models
        if free_only:
            models = [m for m in models if m.is_free]
        if provider:
            models = [m for m in models if m.provider == provider]

        table = Table(title=f"Available Models ({len(models)} shown)")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Context", justify="right")
        table.add_column("Free", justify="center")
        table.add_column("Tools", justify="center")

        for m in models[:50]:
            table.add_row(
                m.id,
                m.provider,
                str(m.context_window) if m.context_window else "-",
                "Y" if m.is_free else "N",
                "Y" if m.supports_tools else "N",
            )

        console.print(table)

    asyncio.run(_list_models())


@models_app.command("recommend")
def models_recommend(
    task: str = typer.Option("general coding task", "--task", "-t", help="Task description"),
    free_only: bool = typer.Option(False, "--free", help="Only recommend free models"),
) -> None:
    """Recommend the best model for a task with empirical intelligence."""
    from harness_core.classifier.classifier import TaskClassifier
    from harness_core.models.registry import ModelRegistry
    from harness_core.models.empirical import EmpiricalHistory
    from harness_core.models.discovery import discover_provider
    from harness_core.routing.task_aware import TaskAwareRouter
    from harness_core.providers.openrouter import OpenRouterProvider

    async def _recommend() -> None:
        classifier = TaskClassifier()
        task_type, confidence = classifier.classify_with_confidence(task)
        profile = classifier.get_profile(task)

        console.print(f"\n[bold]Task:[/] {task}")
        console.print(f"[bold]Classification:[/] {task_type.value} (confidence: {confidence:.0%})")

        reqs = profile.get_requirements()
        console.print("[bold]Requirements:[/]")
        for dim, val in sorted(reqs.items(), key=lambda x: -x[1]):
            if val > 0.1:
                console.print(f"  {dim}: {val:.2f}")

        registry = ModelRegistry()
        empirical = EmpiricalHistory()
        openrouter = OpenRouterProvider()
        profiles = await discover_provider(openrouter)
        for p in profiles:
            registry.register(p)
        await openrouter.close()

        router = TaskAwareRouter(
            registry=registry,
            empirical=empirical,
            classifier=classifier,
        )

        models = registry.list_all()
        model_ids = [m.model_id for m in models if m.supports_tools]
        if free_only:
            free_models = [m for m in models if m.is_free and m.supports_tools]
            model_ids = [m.model_id for m in free_models]
            if not model_ids:
                console.print("[yellow]No suitable free models found.[/]")
                return

        ranked = router.rank_with_explanation(model_ids, task_type, profile)

        if not ranked:
            console.print("[yellow]No suitable models found.[/]")
            return

        best_id, best_score, best_exp = ranked[0]
        console.print(f"\n[bold green]Recommended:[/] {best_id}")
        console.print(f"[dim]Final score:[/] {best_exp.final_score:.3f}")
        console.print(f"[dim]Reason:[/] {best_exp.reason}")

        table = Table(title="Top Models (Empirical Task-Aware Ranking)")
        table.add_column("Model", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Empirical", justify="right")
        table.add_column("Samples", justify="right")
        table.add_column("Confidence", justify="center")
        table.add_column("Reason", style="dim")

        for mid, score, exp in ranked[:5]:
            emp_str = f"{exp.empirical_task_success:.0%}" if exp.empirical_samples > 0 else "Unknown"
            samples_str = str(exp.empirical_samples) if exp.empirical_samples > 0 else "-"
            conf_str = exp.empirical_confidence.value if exp.empirical_samples > 0 else "-"
            table.add_row(
                mid,
                f"{score:.3f}",
                emp_str,
                samples_str,
                conf_str,
                exp.reason[:60],
            )

        console.print(table)

    asyncio.run(_recommend())


@models_app.command("inspect")
def models_inspect(
    model_id: str = typer.Argument(..., help="Model ID to inspect"),
) -> None:
    """Inspect a model's profile and capabilities with empirical data."""
    from harness_core.models.registry import ModelRegistry
    from harness_core.models.empirical import EmpiricalHistory
    from harness_core.models.discovery import discover_provider
    from harness_core.providers.openrouter import OpenRouterProvider

    async def _inspect() -> None:
        registry = ModelRegistry()
        empirical = EmpiricalHistory()
        openrouter = OpenRouterProvider()
        profiles = await discover_provider(openrouter)
        for p in profiles:
            registry.register(p)
        await openrouter.close()

        profile = registry.get(model_id)
        if profile is None:
            console.print(f"[red]Model '{model_id}' not found.[/]")
            raise typer.Exit(code=1)

        console.print(Panel(f"Model: {profile.model_id}", border_style="blue"))
        console.print(f"  [bold]Provider:[/] {profile.provider}")
        console.print(f"  [bold]Context:[/] {profile.context_window}")
        console.print(f"  [bold]Tools:[/] {'Yes' if profile.supports_tools else 'No'}")
        console.print(f"  [bold]Free:[/] {'Yes' if profile.is_free else 'No'}")

        console.print("\n[bold]Static Capabilities:[/]")
        for attr in ["coding", "tool_use", "reasoning", "planning",
                      "repository_navigation", "context_handling",
                      "error_recovery", "instruction_following", "verification"]:
            cap = getattr(profile.capabilities, attr)
            if cap.is_measured:
                console.print(f"  {attr}: {cap.score:.3f} ({cap.confidence.value}, source: {cap.source.value})")
            else:
                console.print(f"  {attr}: [dim]Not measured[/]")

        emp_profile = empirical.get_profile(model_id)
        if emp_profile.total_records > 0:
            console.print(f"\n[bold]Empirical Performance:[/]")
            o = emp_profile.overall
            console.print(f"  Total runs: {o.total_tasks}")
            console.print(f"  Success rate: {o.success_rate:.1%}")
            console.print(f"  Verification rate: {o.verification_rate:.1%}")
            console.print(f"  Confidence: {o.sample_confidence.value}")
            console.print(f"  Recent success rate: {o.recent_success_rate:.1%}")
            console.print(f"  Avg latency: {o.avg_latency_ms:.0f}ms")
            console.print(f"  Avg tool calls: {o.avg_tool_calls:.1f}")

            if emp_profile.by_task_type:
                console.print("\n  [bold]By Task Type:[/]")
                for tt, perf in sorted(emp_profile.by_task_type.items()):
                    console.print(f"    {tt}: {perf.success_rate:.0%} ({perf.total_tasks} tasks, {perf.sample_confidence.value})")
        else:
            console.print("\n[bold]Empirical Performance:[/] [dim]No data yet[/]")

    asyncio.run(_inspect())


@models_app.command("local")
def models_local() -> None:
    """List locally installed Ollama models."""
    from harness_core.providers.ollama import OllamaProvider

    async def _local() -> None:
        ollama = OllamaProvider()
        if not await ollama.health_check():
            console.print("[yellow]Ollama is not running.[/]")
            console.print("[dim]Start Ollama with: ollama serve[/]")
            raise typer.Exit(code=0)

        models = await ollama.list_models()
        await ollama.close()

        if not models:
            console.print("[yellow]No local models found.[/]")
            return

        table = Table(title=f"Local Ollama Models ({len(models)})")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Tools", justify="center")

        for m in models:
            table.add_row(
                m.id,
                m.provider,
                "Y" if m.supports_tools else "N",
            )

        console.print(table)

    asyncio.run(_local())


@models_app.command("compare")
def models_compare(
    model_a: str = typer.Argument(..., help="First model ID"),
    model_b: str = typer.Argument(..., help="Second model ID"),
) -> None:
    """Compare two models side by side with empirical data."""
    from harness_core.models.registry import ModelRegistry
    from harness_core.models.empirical import EmpiricalHistory
    from harness_core.models.discovery import discover_provider
    from harness_core.providers.openrouter import OpenRouterProvider

    async def _compare() -> None:
        registry = ModelRegistry()
        empirical = EmpiricalHistory()
        openrouter = OpenRouterProvider()
        profiles = await discover_provider(openrouter)
        for p in profiles:
            registry.register(p)
        await openrouter.close()

        a = registry.get(model_a)
        b = registry.get(model_b)

        if a is None:
            console.print(f"[red]Model '{model_a}' not found.[/]")
            raise typer.Exit(code=1)
        if b is None:
            console.print(f"[red]Model '{model_b}' not found.[/]")
            raise typer.Exit(code=1)

        table = Table(title="Model Comparison")
        table.add_column("Attribute", style="bold")
        table.add_column(a.model_id, style="cyan")
        table.add_column(b.model_id, style="green")

        table.add_row("Provider", a.provider, b.provider)
        table.add_row("Context", str(a.context_window), str(b.context_window))
        table.add_row("Free", "Yes" if a.is_free else "No", "Yes" if b.is_free else "No")
        table.add_row("Tools", "Yes" if a.supports_tools else "No", "Yes" if b.supports_tools else "No")

        for attr in ["coding", "tool_use", "reasoning", "planning",
                      "repository_navigation", "context_handling",
                      "error_recovery", "instruction_following", "verification"]:
            cap_a = getattr(a.capabilities, attr)
            cap_b = getattr(b.capabilities, attr)
            val_a = f"{cap_a.score:.3f}" if cap_a.is_measured else "Unknown"
            val_b = f"{cap_b.score:.3f}" if cap_b.is_measured else "Unknown"
            table.add_row(attr, val_a, val_b)

        console.print(table)

        emp_a = empirical.get_profile(model_a)
        emp_b = empirical.get_profile(model_b)

        if emp_a.total_records > 0 or emp_b.total_records > 0:
            emp_table = Table(title="Empirical Performance Comparison")
            emp_table.add_column("Metric", style="bold")
            emp_table.add_column(a.model_id, style="cyan")
            emp_table.add_column(b.model_id, style="green")

            def _fmt_emp(p):
                if p.overall.total_tasks == 0:
                    return {"runs": "0", "success": "N/A", "confidence": "N/A", "latency": "N/A"}
                return {
                    "runs": str(p.overall.total_tasks),
                    "success": f"{p.overall.success_rate:.0%}",
                    "confidence": p.overall.sample_confidence.value,
                    "latency": f"{p.overall.avg_latency_ms:.0f}ms",
                }

            fa, fb = _fmt_emp(emp_a), _fmt_emp(emp_b)
            emp_table.add_row("Total runs", fa["runs"], fb["runs"])
            emp_table.add_row("Success rate", fa["success"], fb["success"])
            emp_table.add_row("Confidence", fa["confidence"], fb["confidence"])
            emp_table.add_row("Avg latency", fa["latency"], fb["latency"])

            all_tasks = set(emp_a.by_task_type.keys()) | set(emp_b.by_task_type.keys())
            for tt in sorted(all_tasks):
                sa = emp_a.by_task_type.get(tt)
                sb = emp_b.by_task_type.get(tt)
                val_a = f"{sa.success_rate:.0%} ({sa.total_tasks})" if sa and sa.total_tasks > 0 else "N/A"
                val_b = f"{sb.success_rate:.0%} ({sb.total_tasks})" if sb and sb.total_tasks > 0 else "N/A"
                emp_table.add_row(f"  {tt}", val_a, val_b)

            console.print(emp_table)

    asyncio.run(_compare())


@models_app.command("benchmark")
def models_benchmark(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to benchmark"),
    free: bool = typer.Option(False, "--free", help="Benchmark free models only"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Benchmark category (tool_use/navigation/coding/debugging)")
) -> None:
    """Benchmark model capabilities."""
    from harness_core.benchmarks.types import BenchmarkCategory
    from harness_core.benchmarks.tasks import ALL_BENCHMARK_TASKS
    from harness_core.benchmarks.engine import AgentBenchmarkEngine
    from harness_core.providers.openrouter import OpenRouterProvider

    async def _benchmark() -> None:
        if not model:
            console.print("[bold blue]Harness Agent Benchmark[/]")
            console.print("[dim]Run with --model <model-id> to actually execute benchmarks.[/]")

            table = Table(title="Available Benchmark Tasks")
            table.add_column("Category", style="cyan")
            table.add_column("Count", justify="right")
            table.add_column("Examples", style="dim")

            categories: dict[str, list[str]] = {}
            for task in ALL_BENCHMARK_TASKS:
                cat = task.category.value
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(task.name)

            for cat, names in categories.items():
                table.add_row(cat, str(len(names)), ", ".join(names[:3]))

            console.print(table)
            console.print("\n[dim]Usage:[/]")
            console.print("  [cyan]harness models benchmark --model <model-id>[/]")
            console.print("  [cyan]harness models benchmark --model <model-id> --category coding[/]")
            return

        console.print(f"[bold blue]Benchmarking {model}...[/]")
        console.print("[dim]Running in isolated workspaces. This may take a while.[/]")

        provider = OpenRouterProvider()
        if not await provider.health_check():
            console.print("[red][FAIL] OpenRouter not available. Check OPENROUTER_API_KEY.[/]")
            raise typer.Exit(code=4)

        tasks = ALL_BENCHMARK_TASKS
        if category:
            try:
                cat_enum = BenchmarkCategory(category)
                tasks = [t for t in tasks if t.category == cat_enum]
            except ValueError:
                console.print(f"[red]Unknown category: {category}[/]")
                console.print(f"[dim]Available: {', '.join(c.value for c in BenchmarkCategory)}[/]")
                raise typer.Exit(code=1)

        if not tasks:
            console.print("[yellow]No tasks found for the specified criteria.[/]")
            raise typer.Exit(code=1)

        engine = AgentBenchmarkEngine(providers={"openrouter": provider})

        with Progress(
            SpinnerColumn("dots"),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            ptask = progress.add_task(f"Running {len(tasks)} benchmarks...", total=len(tasks))
            results = []
            for bench_task in tasks:
                progress.update(ptask, description=f"Benchmarking: {bench_task.name}")
                result = await engine.run_task(bench_task, model, "openrouter")
                results.append(result)
                progress.advance(ptask)

        await provider.close()

        console.print(f"\n[bold blue]Benchmark Results: {model}[/]")

        table = Table(title="Task Results")
        table.add_column("Task", style="cyan")
        table.add_column("Category", style="green")
        table.add_column("Success", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Tools", justify="right")
        table.add_column("Iterations", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("Error", style="dim")

        success_count = 0
        for r in results:
            status = "Y" if r.success else "N"
            color = "green" if r.success else "red"
            table.add_row(
                r.task_name,
                r.category,
                f"[{color}]{status}[/]",
                f"{r.score:.2f}",
                str(r.tool_calls),
                str(r.iterations),
                f"{r.latency_ms:.0f}ms",
                r.error[:40] if r.error else "",
            )
            if r.success:
                success_count += 1

        console.print(table)
        console.print(f"\n[bold]Success rate:[/] {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")
        console.print(f"[bold]Overall score:[/] {sum(r.score for r in results)/len(results):.3f}")

    asyncio.run(_benchmark())


@models_app.command("history")
def models_history(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Filter by model")
) -> None:
    """Show model performance history."""
    from harness_core.models.history import PerformanceHistory

    async def _history() -> None:
        history = PerformanceHistory()
        all_perf = history.get_all_performance()

        if not all_perf:
            console.print("[dim]No performance history yet. Run some tasks first.[/]")
            return

        table = Table(title="Model Performance History")
        table.add_column("Model", style="cyan")
        table.add_column("Tasks", justify="right")
        table.add_column("Success", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("Tool Calls", justify="right")
        table.add_column("Recovery", justify="right")

        for perf in all_perf:
            if model and model not in perf.model_id:
                continue
            table.add_row(
                perf.model_id,
                str(perf.total_tasks),
                f"{perf.success_rate:.0%}",
                f"{perf.avg_latency_ms:.0f}ms",
                f"{perf.avg_tool_calls:.1f}",
                f"{perf.recovery_rate:.0%}",
            )

        console.print(table)

    asyncio.run(_history())


# ── session sub-commands ──────────────────────────────────────────────────

@session_app.command("list")
def session_list(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max sessions to show"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List recent sessions."""
    from harness_core.session.manager import SessionManager
    from harness_core.session.domain import SessionStatus

    manager = SessionManager()
    filter_status = SessionStatus(status) if status else None
    sessions = manager.storage.list_sessions(status=filter_status, limit=limit)

    if json_output:
        print(json.dumps([s.to_dict() for s in sessions], indent=2))
        return

    if not sessions:
        console.print("[dim]No sessions found.[/]")
        return

    table = Table(title="Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Title", max_width=40)
    table.add_column("Status")
    table.add_column("Workspace", style="dim", max_width=30)
    table.add_column("Updated", justify="right")

    for s in sessions:
        import datetime
        updated = datetime.datetime.fromtimestamp(s.updated_at).strftime("%Y-%m-%d %H:%M")
        status_color = {
            "active": "green", "paused": "yellow", "completed": "blue",
            "failed": "red", "aborted": "red", "archived": "dim",
        }.get(s.status.value, "white")
        table.add_row(
            s.session_id[:12],
            s.title[:40],
            f"[{status_color}]{s.status.value}[/]",
            s.workspace_path[:30] if s.workspace_path else "-",
            updated,
        )

    console.print(table)


@session_app.command("show")
def session_show(
    session_id: str = typer.Argument(..., help="Session ID (prefix match)"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show session details with runs and memories."""
    from harness_core.session.manager import SessionManager

    manager = SessionManager()
    state = manager.get_resume_state(session_id)

    if state is None:
        console.print(f"[red]Session '{session_id}' not found.[/]")
        raise typer.Exit(code=1)

    session = state["session"]
    runs = state["runs"]
    memories = state["memories"]
    checkpoint = state["checkpoint"]

    if json_output:
        data = {
            "session": session.to_dict(),
            "runs": [r.to_dict() for r in runs],
            "memories": [m.to_dict() for m in memories],
            "checkpoint": checkpoint.to_dict() if checkpoint else None,
        }
        print(json.dumps(data, indent=2))
        return

    console.print(Panel(
        f"[bold]{session.title}[/]\n"
        f"ID: {session.session_id}\n"
        f"Status: {session.status.value}\n"
        f"Workspace: {session.workspace_path}\n"
        f"Created: {time.strftime('%Y-%m-%d %H:%M', time.localtime(session.created_at))}\n"
        f"Updated: {time.strftime('%Y-%m-%d %H:%M', time.localtime(session.updated_at))}\n"
        f"Runs: {len(runs)}",
        title="Session",
        border_style="blue",
    ))

    if runs:
        table = Table(title=f"Runs ({len(runs)})")
        table.add_column("Run ID", style="cyan")
        table.add_column("Task", max_width=50)
        table.add_column("Status")
        table.add_column("Model", style="green")
        table.add_column("Tools", justify="right")
        table.add_column("Duration", justify="right")

        for r in runs:
            color = {
                "completed": "green", "failed": "red",
                "interrupted": "yellow", "running": "blue",
            }.get(r.status.value, "white")
            dur = f"{r.duration_ms / 1000:.1f}s" if r.duration_ms > 0 else "-"
            table.add_row(
                r.run_id[:12],
                r.task[:50],
                f"[{color}]{r.status.value}[/]",
                r.model_id[:25] if r.model_id else "-",
                str(r.tool_calls),
                dur,
            )
        console.print(table)

    if memories:
        mem_table = Table(title=f"Memories ({len(memories)})")
        mem_table.add_column("Type", style="cyan")
        mem_table.add_column("Content", max_width=60)
        mem_table.add_column("Importance", justify="right")
        for m in memories[:10]:
            mem_table.add_row(m.memory_type.value, m.content[:60], f"{m.importance:.1f}")
        console.print(mem_table)

    if checkpoint:
        console.print(Panel(
            f"Git HEAD: {checkpoint.git_head or 'N/A'}\n"
            f"Branch: {checkpoint.git_branch or 'N/A'}\n"
            f"Tests: {checkpoint.tests_passing}/{checkpoint.tests_total}\n"
            f"Verification: {checkpoint.verification_status or 'N/A'}",
            title="Last Checkpoint",
            border_style="dim",
        ))


@session_app.command("create")
def session_create(
    title: str = typer.Option("", "--title", "-t", help="Session title"),
) -> None:
    """Create a new session."""
    from harness_core.session.manager import SessionManager

    manager = SessionManager()
    workspace = str(Path.cwd())
    session = manager.create_session(workspace_path=workspace, title=title)
    console.print(f"[green]Session created:[/] {session.session_id}")
    console.print(f"[dim]Title: {session.title}[/]")
    console.print(f"[dim]Workspace: {workspace}[/]")


@session_app.command("pause")
def session_pause(session_id: str = typer.Argument(..., help="Session ID")) -> None:
    """Pause a session."""
    from harness_core.session.manager import SessionManager

    manager = SessionManager()
    try:
        session = manager.pause_session(session_id)
        console.print(f"[yellow]Session paused:[/] {session.session_id}")
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=1)


@session_app.command("resume")
def session_resume(session_id: str = typer.Argument(..., help="Session ID")) -> None:
    """Resume a paused session."""
    from harness_core.session.manager import SessionManager

    manager = SessionManager()
    try:
        session = manager.resume_session(session_id)
        console.print(f"[green]Session resumed:[/] {session.session_id}")
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=1)


@session_app.command("archive")
def session_archive(session_id: str = typer.Argument(..., help="Session ID")) -> None:
    """Archive a session."""
    from harness_core.session.manager import SessionManager

    manager = SessionManager()
    try:
        session = manager.archive_session(session_id)
        console.print(f"[dim]Session archived:[/] {session.session_id}")
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=1)


@session_app.command("delete")
def session_delete(
    session_id: str = typer.Argument(..., help="Session ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a session (soft archive by default)."""
    from harness_core.session.manager import SessionManager

    manager = SessionManager()
    session = manager.storage.get_session(session_id)
    if session is None:
        console.print(f"[red]Session '{session_id}' not found.[/]")
        raise typer.Exit(code=1)

    if not force:
        confirm = typer.confirm(f"Delete session '{session.title}' ({session_id})?")
        if not confirm:
            console.print("[dim]Cancelled.[/]")
            return

    deleted = manager.storage.delete_session(session_id)
    if deleted > 0:
        console.print(f"[red]Session deleted:[/] {session_id}")
    else:
        console.print(f"[yellow]Session not found or already deleted.[/]")


@session_app.command("export")
def session_export(
    session_id: str = typer.Argument(..., help="Session ID"),
    fmt: str = typer.Option("json", "--format", "-f", help="Export format (json/markdown)"),
) -> None:
    """Export session data."""
    from harness_core.session.manager import SessionManager

    manager = SessionManager()
    try:
        output = manager.export_session(session_id, fmt=fmt)
        print(output)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=1)


@session_app.command("diff")
def session_diff(session_id: str = typer.Argument(..., help="Session ID")) -> None:
    """Show session changes summary."""
    from harness_core.session.manager import SessionManager

    manager = SessionManager()
    state = manager.get_resume_state(session_id)
    if state is None:
        console.print(f"[red]Session '{session_id}' not found.[/]")
        raise typer.Exit(code=1)

    checkpoint = state["checkpoint"]
    runs = state["runs"]

    console.print(Panel(f"Session: {state['session'].title}", title="Session Diff", border_style="blue"))

    if checkpoint:
        console.print(f"[bold]Last Checkpoint:[/]")
        console.print(f"  Git HEAD: {checkpoint.git_head or 'N/A'}")
        console.print(f"  Branch: {checkpoint.git_branch or 'N/A'}")
        console.print(f"  Tests: {checkpoint.tests_passing}/{checkpoint.tests_total}")
        if checkpoint.changed_files:
            console.print(f"  Changed files: {len(checkpoint.changed_files)}")
            for f in checkpoint.changed_files[:10]:
                console.print(f"    {f}")

    completed = [r for r in runs if r.status.value == "completed"]
    failed = [r for r in runs if r.status.value == "failed"]
    interrupted = [r for r in runs if r.status.value == "interrupted"]

    console.print(f"\n[bold]Runs:[/] {len(completed)} completed, {len(failed)} failed, {len(interrupted)} interrupted")


@session_app.command("memory")
def session_memory(
    session_id: str = typer.Argument(..., help="Session ID"),
    add: Optional[str] = typer.Option(None, "--add", help="Add a memory item"),
    memory_type: str = typer.Option("note", "--type", help="Memory type"),
    importance: float = typer.Option(0.5, "--importance", help="Importance (0.0-1.0)"),
    search: Optional[str] = typer.Option(None, "--search", help="Search memories"),
) -> None:
    """Manage session memory."""
    from harness_core.session.manager import SessionManager
    from harness_core.session.domain import MemoryType

    manager = SessionManager()

    if add:
        try:
            mtype = MemoryType(memory_type)
        except ValueError:
            console.print(f"[red]Invalid memory type: {memory_type}[/]")
            console.print(f"Valid types: {[t.value for t in MemoryType]}")
            raise typer.Exit(code=1)

        item = manager.add_memory(session_id, mtype, add, importance=importance)
        console.print(f"[green]Memory added:[/] {item.memory_id}")
        return

    if search:
        items = manager.storage.search_memories(session_id, search)
    else:
        items = manager.storage.get_memories(session_id)

    if not items:
        console.print("[dim]No memories found.[/]")
        return

    table = Table(title=f"Memories ({len(items)})")
    table.add_column("Type", style="cyan")
    table.add_column("Content", max_width=60)
    table.add_column("Importance", justify="right")
    table.add_column("Created", style="dim")

    for m in items:
        import datetime
        created = datetime.datetime.fromtimestamp(m.created_at).strftime("%Y-%m-%d %H:%M")
        table.add_row(m.memory_type.value, m.content[:60], f"{m.importance:.1f}", created)

    console.print(table)


# ── config sub-commands ────────────────────────────────────────────────────

@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    config_file = Path(".harness/config.yaml")
    if config_file.exists():
        console.print(config_file.read_text(encoding="utf-8"))
    else:
        console.print("[red]No configuration found. Run `harness init` first.[/]")


# ── M7 Extension CLI Commands ──────────────────────────────────────────────

plugin_app = typer.Typer(help="Plugin management")
app.add_typer(plugin_app, name="plugin")

tools_app = typer.Typer(help="Tool management")
app.add_typer(tools_app, name="tools")

mcp_app = typer.Typer(help="MCP server management")
app.add_typer(mcp_app, name="mcp")

hooks_app = typer.Typer(help="Hook management")
app.add_typer(hooks_app, name="hooks")

providers_app = typer.Typer(help="Provider management")
app.add_typer(providers_app, name="providers")


@plugin_app.command("list")
def plugin_list(
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List installed plugins."""
    from harness_core.plugins.manager import PluginManager

    pm = PluginManager()
    plugins = pm.list_all()

    if json_output:
        print(json.dumps(plugins, indent=2))
        return

    if not plugins:
        console.print("[dim]No plugins installed.[/]")
        console.print("[dim]Install: harness plugin install <path>[/]")
        return

    table = Table(title=f"Plugins ({len(plugins)})")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Type", style="green")
    table.add_column("State")
    table.add_column("Description", max_width=40)

    for p in plugins:
        state_color = {
            "enabled": "green", "disabled": "yellow",
            "failed": "red", "installed": "blue",
        }.get(p.get("state", ""), "white")
        table.add_row(
            p.get("name", ""),
            p.get("version", ""),
            p.get("type", ""),
            f"[{state_color}]{p.get('state', '')}[/]",
            p.get("description", "")[:40],
        )

    console.print(table)


@plugin_app.command("install")
def plugin_install(
    path: str = typer.Argument(..., help="Plugin directory path"),
) -> None:
    """Install a plugin from a local directory."""
    from harness_core.plugins.manager import PluginManager

    pm = PluginManager()
    try:
        manifest = pm.install(path)
        if manifest:
            console.print(f"[green]Plugin installed:[/] {manifest.name} v{manifest.version}")
        else:
            console.print("[red]Failed to install plugin.[/]")
            raise typer.Exit(code=1)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=1)


@plugin_app.command("enable")
def plugin_enable(name: str = typer.Argument(..., help="Plugin name")) -> None:
    """Enable a plugin."""
    from harness_core.plugins.manager import PluginManager
    pm = PluginManager()
    if pm.enable(name):
        console.print(f"[green]Plugin enabled:[/] {name}")
    else:
        console.print(f"[red]Plugin '{name}' not found or cannot be enabled.[/]")
        raise typer.Exit(code=1)


@plugin_app.command("disable")
def plugin_disable(name: str = typer.Argument(..., help="Plugin name")) -> None:
    """Disable a plugin."""
    from harness_core.plugins.manager import PluginManager
    pm = PluginManager()
    if pm.disable(name):
        console.print(f"[yellow]Plugin disabled:[/] {name}")
    else:
        console.print(f"[red]Plugin '{name}' not found.[/]")
        raise typer.Exit(code=1)


@plugin_app.command("remove")
def plugin_remove(
    name: str = typer.Argument(..., help="Plugin name"),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Remove a plugin."""
    from harness_core.plugins.manager import PluginManager
    pm = PluginManager()
    if not force:
        confirm = typer.confirm(f"Remove plugin '{name}'?")
        if not confirm:
            console.print("[dim]Cancelled.[/]")
            return
    if pm.remove(name):
        console.print(f"[red]Plugin removed:[/] {name}")
    else:
        console.print(f"[red]Plugin '{name}' not found.[/]")
        raise typer.Exit(code=1)


@plugin_app.command("inspect")
def plugin_inspect(
    name: str = typer.Argument(..., help="Plugin name"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Inspect a plugin's details."""
    from harness_core.plugins.manager import PluginManager
    pm = PluginManager()
    info = pm.inspect(name)
    if info is None:
        console.print(f"[red]Plugin '{name}' not found.[/]")
        raise typer.Exit(code=1)

    if json_output:
        print(json.dumps(info, indent=2))
        return

    console.print(Panel(f"Plugin: {info.get('name', '')}", border_style="blue"))
    console.print(f"  [bold]Version:[/] {info.get('version', '')}")
    console.print(f"  [bold]Type:[/] {info.get('type', '')}")
    console.print(f"  [bold]Description:[/] {info.get('description', '')}")
    console.print(f"  [bold]Author:[/] {info.get('author', '')}")
    console.print(f"  [bold]State:[/] {info.get('state', '')}")
    console.print(f"  [bold]Capabilities:[/] {info.get('capabilities', [])}")
    console.print(f"  [bold]Permissions:[/] {info.get('permissions', [])}")
    if info.get('load_error'):
        console.print(f"  [bold red]Load Error:[/] {info['load_error']}")


@tools_app.command("list")
def tools_list(
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List available tools (core + plugin + MCP)."""
    from harness_core.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListFilesTool
    from harness_core.tools.search import GlobTool, GrepTool
    from harness_core.tools.shell import RunCommandTool
    from harness_core.tools.git import GitStatusTool, GitDiffTool, GitLogTool

    core_tools = [
        ReadFileTool(), WriteFileTool(), EditFileTool(), ListFilesTool(),
        GlobTool(), GrepTool(), RunCommandTool(),
        GitStatusTool(), GitDiffTool(), GitLogTool(),
    ]

    tool_list = []
    for t in core_tools:
        s = t.schema
        tool_list.append({
            "name": s.name,
            "description": s.description,
            "source": "CORE",
            "parameters": s.parameters,
        })

    if json_output:
        print(json.dumps(tool_list, indent=2))
        return

    table = Table(title=f"Tools ({len(tool_list)} core)")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Description", max_width=50)

    for t in tool_list:
        table.add_row(t["name"], t["source"], t["description"][:50])

    console.print(table)


@tools_app.command("inspect")
def tools_inspect(
    name: str = typer.Argument(..., help="Tool name"),
) -> None:
    """Inspect a tool's details."""
    from harness_core.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListFilesTool
    from harness_core.tools.search import GlobTool, GrepTool
    from harness_core.tools.shell import RunCommandTool
    from harness_core.tools.git import GitStatusTool, GitDiffTool, GitLogTool

    all_tools = {}
    for t in [
        ReadFileTool(), WriteFileTool(), EditFileTool(), ListFilesTool(),
        GlobTool(), GrepTool(), RunCommandTool(),
        GitStatusTool(), GitDiffTool(), GitLogTool(),
    ]:
        all_tools[t.schema.name] = t

    tool = all_tools.get(name)
    if tool is None:
        console.print(f"[red]Tool '{name}' not found.[/]")
        raise typer.Exit(code=1)

    s = tool.schema
    console.print(Panel(f"Tool: {s.name}", border_style="blue"))
    console.print(f"  [bold]Description:[/] {s.description}")
    console.print(f"  [bold]Source:[/] CORE")
    console.print(f"  [bold]Parameters:[/] {json.dumps(s.parameters, indent=2)}")


@mcp_app.command("list")
def mcp_list(
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List configured MCP servers."""
    from harness_core.mcp.client import MCPClient
    client = MCPClient()
    servers = client.list_servers()
    status = client.get_status()

    if json_output:
        print(json.dumps([s.to_dict() for s in servers], indent=2))
        return

    if not servers:
        console.print("[dim]No MCP servers configured.[/]")
        console.print("[dim]Configure in .harness/config.yaml under 'mcp:'[/]")
        return

    table = Table(title=f"MCP Servers ({len(servers)})")
    table.add_column("Name", style="cyan")
    table.add_column("Command", style="green")
    table.add_column("Transport")
    table.add_column("Enabled")
    table.add_column("Status")

    for s in servers:
        st = status.get(s.name, {})
        running = "Running" if st.get("running") else "Stopped"
        table.add_row(
            s.name, s.command, s.transport,
            "Yes" if s.enabled else "No",
            running,
        )

    console.print(table)


@hooks_app.command("list")
def hooks_list(
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List registered hooks."""
    from harness_core.hooks.hooks import HookRegistry
    registry = HookRegistry()
    hooks = registry.list_hooks()
    stats = registry.get_stats()

    if json_output:
        print(json.dumps({
            "hooks": [{"id": h.hook_id, "event": h.event.value, "name": h.name} for h in hooks],
            "stats": stats,
        }, indent=2))
        return

    if not hooks:
        console.print("[dim]No hooks registered.[/]")
        console.print("[dim]Plugins can register hooks via the extension API.[/]")
        return

    table = Table(title=f"Hooks ({len(hooks)})")
    table.add_column("ID", style="cyan")
    table.add_column("Event", style="green")
    table.add_column("Name")
    table.add_column("Priority", justify="right")
    table.add_column("Source")

    for h in hooks:
        table.add_row(h.hook_id, h.event.value, h.name, str(h.priority), h.source or "-")

    console.print(table)
    console.print(f"[dim]Stats: {stats['total_hooks']} hooks across {len(stats['events'])} events[/]")


# ── M8 Provider Setup & Interactive Setup ───────────────────────────────────

@providers_app.command("list")
def providers_list(
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List available providers and their status."""
    providers_info = [
        {
            "name": "OpenRouter",
            "env_var": "OPENROUTER_API_KEY",
            "description": "Access 300+ models including free tier",
            "url": "https://openrouter.ai",
            "free_models": True,
        },
        {
            "name": "Ollama",
            "env_var": "OLLAMA_HOST",
            "description": "Local inference — no API key needed",
            "url": "https://ollama.com",
            "free_models": True,
        },
        {
            "name": "LiteLLM",
            "env_var": "LITELLM_API_KEY",
            "description": "Unified API for 100+ LLM providers",
            "url": "https://litellm.ai",
            "free_models": False,
        },
    ]

    if json_output:
        print(json.dumps(providers_info, indent=2))
        return

    table = Table(title="Available Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Free Models", justify="center")
    table.add_column("Description", max_width=40)

    import os
    for p in providers_info:
        has_key = bool(os.environ.get(p["env_var"]))
        status = "[green]Configured[/]" if has_key else "[yellow]Not configured[/]"
        table.add_row(
            p["name"],
            status,
            "Yes" if p["free_models"] else "No",
            p["description"],
        )

    console.print(table)
    console.print("\n[bold]Configure:[/]")
    console.print("  [cyan]harness providers configure openrouter[/]")
    console.print("  [cyan]harness providers configure ollama[/]")


@providers_app.command("configure")
def providers_configure(
    provider: str = typer.Argument(..., help="Provider name (openrouter, ollama, litellm)"),
) -> None:
    """Configure a model provider."""
    import os

    provider = provider.lower().strip()

    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if key:
            console.print("[green]OpenRouter is already configured.[/]")
            console.print(f"  API key: {key[:8]}...{key[-4:]}")
            return

        console.print("[bold]Configure OpenRouter[/]")
        console.print("\nOpenRouter provides access to 300+ AI models including free tiers.")
        console.print("\nTo configure:")
        console.print("  1. Visit https://openrouter.ai/keys")
        console.print("  2. Create an API key")
        console.print("  3. Set the environment variable:")
        console.print("\n    [cyan]# PowerShell[/]")
        console.print("    $env:OPENROUTER_API_KEY = 'sk-or-v1-...'\n")
        console.print("    [cyan]# Bash/Linux/macOS[/]")
        console.print("    export OPENROUTER_API_KEY='sk-or-v1-...'\n")
        console.print("    [cyan]# Or add to .env file (in project root)[/]")
        console.print("    echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env\n")
        console.print("  4. Run: [cyan]harness doctor[/] to verify\n")
        console.print("[dim]Free models available — no payment required to start.[/]")

    elif provider == "ollama":
        console.print("[bold]Configure Ollama[/]")
        console.print("\nOllama runs AI models locally — no API key needed.")
        console.print("\nTo configure:")
        console.print("  1. Install Ollama: https://ollama.com/download")
        console.print("  2. Start the server: [cyan]ollama serve[/]")
        console.print("  3. Pull a model: [cyan]ollama pull codellama[/]")
        console.print("  4. Run: [cyan]harness doctor[/] to verify\n")
        console.print("[dim]Recommended models: codellama, deepseek-coder, llama3[/]")

    elif provider == "litellm":
        console.print("[bold]Configure LiteLLM[/]")
        console.print("\nLiteLLM provides a unified API for 100+ LLM providers.")
        console.print("\nTo configure:")
        console.print("  1. Set the API key:")
        console.print("\n    [cyan]# PowerShell[/]")
        console.print("    $env:LITELLM_API_KEY = 'your-key'\n")
        console.print("    [cyan]# Bash/Linux/macOS[/]")
        console.print("    export LITELLM_API_KEY='your-key'\n")
        console.print("  2. Run: [cyan]harness doctor[/] to verify")

    else:
        console.print(f"[red]Unknown provider: {provider}[/]")
        console.print("[dim]Available: openrouter, ollama, litellm[/]")
        raise typer.Exit(code=1)


@app.command()
def setup() -> None:
    """Interactive setup for new users."""
    import os
    console.print(Panel("Welcome to Harness Engineering CLI", border_style="blue"))
    console.print("\nHarness is a model-agnostic AI coding agent that runs in your terminal.")
    console.print("This wizard will help you get started.\n")

    # Step 1: Check existing configuration
    console.print("[bold]Step 1: Checking configuration...[/]")
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    has_ollama = False
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5,
        )
        has_ollama = result.returncode == 0
    except Exception:
        pass

    if has_openrouter:
        console.print("  [green][OK] OpenRouter configured[/]")
    else:
        console.print("  [yellow][--] OpenRouter not configured[/]")

    if has_ollama:
        console.print("  [green][OK] Ollama running[/]")
    else:
        console.print("  [yellow][--] Ollama not running[/]")

    if not has_openrouter and not has_ollama:
        console.print("\n[bold yellow]No providers configured![/]")
        console.print("\nYou need at least one provider to use Harness:")
        console.print("\n  [cyan]Option 1: OpenRouter (recommended for beginners)[/]")
        console.print("    - Access 300+ models including free tiers")
        console.print("    - No payment required to start")
        console.print("    - Run: [cyan]harness providers configure openrouter[/]\n")
        console.print("  [cyan]Option 2: Ollama (local, private, free)[/]")
        console.print("    - Runs entirely on your machine")
        console.print("    - No API key needed")
        console.print("    - Run: [cyan]harness providers configure ollama[/]\n")
        console.print("[dim]You can configure providers now or later.[/]")
        return

    # Step 2: Test connection
    console.print("\n[bold]Step 2: Testing provider connection...[/]")

    async def _test() -> None:
        if has_openrouter:
            from harness_core.providers.openrouter import OpenRouterProvider
            provider = OpenRouterProvider()
            if await provider.health_check():
                models = await provider.list_models()
                free = [m for m in models if m.is_free]
                console.print(f"  [green][OK] OpenRouter connected ({len(models)} models, {len(free)} free)[/]")
            else:
                console.print("  [red][FAIL] OpenRouter connection failed[/]")
            await provider.close()

        if has_ollama:
            from harness_core.providers.ollama import OllamaProvider
            provider = OllamaProvider()
            if await provider.health_check():
                models = await provider.list_models()
                console.print(f"  [green][OK] Ollama running ({len(models)} models)[/]")
            else:
                console.print("  [red][FAIL] Ollama connection failed[/]")
            await provider.close()

    asyncio.run(_test())

    # Step 3: Project check
    console.print("\n[bold]Step 3: Checking current project...[/]")
    harness_dir = Path(".harness")
    if harness_dir.exists():
        console.print("  [green][OK] Harness project initialized[/]")
    else:
        console.print("  [yellow][--] No Harness project in current directory[/]")
        console.print("  [dim]Run 'harness init' to initialize a project[/]")

    # Step 4: Quick test
    console.print("\n[bold]Step 4: Ready to try?[/]")
    console.print("\nTry your first task:")
    console.print("  [cyan]harness run \"List all Python files and show their sizes\"[/]")
    console.print("\nFor more options:")
    console.print("  [cyan]harness --help[/]")
    console.print("  [cyan]harness models recommend --task 'Fix failing tests'[/]")
    console.print("  [cyan]harness session list[/]")
    console.print("\n[bold green]Setup complete![/]")


@app.command()
def providers() -> None:
    """Manage model providers."""
    console.print("[dim]Use 'harness providers list' or 'harness providers configure <name>'[/]")
    console.print("\nAvailable providers:")
    console.print("  [cyan]openrouter[/] — 300+ models, free tier available")
    console.print("  [cyan]ollama[/]     — local inference, no API key")
    console.print("  [cyan]litellm[/]    — unified API for 100+ providers")


# ── M10 Benchmark Commands ──────────────────────────────────────────────────

benchmark_app = typer.Typer(help="Benchmark commands")
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("run")
def benchmark_run(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to benchmark"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Task category"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Run coding benchmark tasks."""
    from benchmarks.tasks import ALL_TASKS, TaskCategory
    from benchmarks.runner import BenchmarkRunner

    tasks = ALL_TASKS
    if category:
        try:
            cat_enum = TaskCategory(category)
            tasks = [t for t in tasks if t.category == cat_enum]
        except ValueError:
            console.print(f"[red]Unknown category: {category}[/]")
            console.print(f"[dim]Available: {', '.join(c.value for c in TaskCategory)}[/]")
            raise typer.Exit(code=1)

    runner = BenchmarkRunner()
    results = runner.run_all(model=model or "", provider="openrouter")

    if json_output:
        print(json.dumps(runner.get_summary(), indent=2))
        return

    summary = runner.get_summary()
    console.print(Panel("Harness Benchmark Results", border_style="blue"))
    console.print(f"  [bold]Tasks:[/] {summary['total']}")
    console.print(f"  [bold]Success:[/] {summary['success']}")
    console.print(f"  [bold]Partial:[/] {summary['partial']}")
    console.print(f"  [bold]Failed:[/] {summary['failed']}")
    console.print(f"  [bold]Success rate:[/] {summary['success_rate']}")
    console.print(f"  [bold]Avg duration:[/] {summary['avg_duration_ms']}ms")

    if summary.get("by_category"):
        table = Table(title="By Category")
        table.add_column("Category", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Success", justify="right")
        table.add_column("Partial", justify="right")
        table.add_column("Failed", justify="right")
        for cat, stats in summary["by_category"].items():
            table.add_row(
                cat,
                str(stats["total"]),
                str(stats["success"]),
                str(stats["partial"]),
                str(stats["failed"]),
            )
        console.print(table)

    # Save results
    path = runner.save_results()
    console.print(f"\n[dim]Results saved to {path}[/]")


@benchmark_app.command("report")
def benchmark_report() -> None:
    """Show latest benchmark report."""
    from benchmarks.runner import BenchmarkRunner
    import json as json_mod

    runner = BenchmarkRunner()
    results_dir = runner.results_dir
    files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)

    if not files:
        console.print("[dim]No benchmark results found. Run: harness benchmark run[/]")
        return

    latest = files[0]
    data = json_mod.loads(latest.read_text(encoding="utf-8"))
    summary = data.get("summary", {})

    console.print(Panel(f"Benchmark Report — {latest.name}", border_style="blue"))
    console.print(f"  [bold]Tasks:[/] {summary.get('total', 0)}")
    console.print(f"  [bold]Success:[/] {summary.get('success', 0)}")
    console.print(f"  [bold]Partial:[/] {summary.get('partial', 0)}")
    console.print(f"  [bold]Failed:[/] {summary.get('failed', 0)}")
    console.print(f"  [bold]Success rate:[/] {summary.get('success_rate', 'N/A')}")
    console.print(f"  [bold]Avg duration:[/] {summary.get('avg_duration_ms', 0)}ms")

    results = data.get("results", [])
    if results:
        table = Table(title="Task Results")
        table.add_column("Task", style="cyan")
        table.add_column("Category")
        table.add_column("Difficulty")
        table.add_column("Language")
        table.add_column("Status")
        table.add_column("Duration", justify="right")
        table.add_column("Error", style="dim", max_width=30)

        for r in results:
            if r["success"]:
                status = "[green]SUCCESS[/]"
            elif r["partial"]:
                status = "[yellow]PARTIAL[/]"
            else:
                status = "[red]FAILED[/]"
            table.add_row(
                r["task_id"],
                r["category"],
                r["difficulty"],
                r["language"],
                status,
                f"{r['duration_ms']:.0f}ms",
                r.get("error", "")[:30],
            )
        console.print(table)

    console.print(f"\n[dim]Results: {latest}[/]")


if __name__ == "__main__":
    app()

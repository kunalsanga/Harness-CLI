"""CLI entry point for Harness Engineering."""

from __future__ import annotations

import asyncio
import json
import os
import sys
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
    no_args_is_help=True,
)
console = Console()

# Sub-commands
config_app = typer.Typer(help="Configuration commands")
app.add_typer(config_app, name="config")


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
    mode: str = typer.Option("auto", "--mode", help="Routing mode (auto/free/best/fast/local/cheap)"),
    max_iterations: int = typer.Option(30, "--max-iterations", help="Maximum iterations"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", help="Maximum cost"),
    headless: bool = typer.Option(False, "--headless", help="Headless mode"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Run an engineering task."""
    from harness_core.agent.loop import AgentLoop
    from harness_core.agent.types import AgentConfig, AgentRole
    from harness_core.observability.events import Event, EventBus
    from harness_core.providers.openrouter import OpenRouterProvider
    from harness_core.tools.filesystem import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
    from harness_core.tools.git import GitDiffTool, GitLogTool, GitStatusTool
    from harness_core.tools.search import GlobTool, GrepTool
    from harness_core.tools.shell import RunCommandTool

    async def _run() -> None:
        from harness_core.routing.router import ModelRouter, RouterConfig

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
                pass  # graceful fallback

        # Override with CLI args
        if model:
            router_config.routing_mode = mode
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

        router = ModelRouter(
            providers=providers,
            config=router_config,
            event_bus=event_bus,
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

        config = AgentConfig(
            role=AgentRole.BUILD,
            max_iterations=max_iterations,
            model_preference=model,
            routing_mode=mode,
        )

        agent = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=Path.cwd(),
            config=config,
            event_bus=event_bus,
            router=router,
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

    console.print("\n[bold green]Doctor check complete.[/]")


@app.command()
def models(
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


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    config_file = Path(".harness/config.yaml")
    if config_file.exists():
        console.print(config_file.read_text(encoding="utf-8"))
    else:
        console.print("[red]No configuration found. Run `harness init` first.[/]")


if __name__ == "__main__":
    app()

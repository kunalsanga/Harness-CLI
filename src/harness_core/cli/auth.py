"""Authentication CLI commands for Harness.

Provides:
    harness auth setup   — configure provider credentials
    harness auth status  — show provider connection status
    harness auth logout  — remove stored credentials
"""

from __future__ import annotations

import sys
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from harness_core.config.credentials import (
    CredentialResolver,
    CredentialStore,
    ProviderCredential,
    mask_secret,
)

console = Console()

auth_app = typer.Typer(help="Authentication and provider configuration")


@auth_app.command("setup")
def auth_setup(
    provider: str = typer.Option(
        "openrouter", "--provider", "-p",
        help="Provider to configure (openrouter, openai, anthropic, nvidia)"
    ),
) -> None:
    """Configure provider API credentials."""
    provider = provider.lower().strip()
    valid_providers = ["openrouter", "openai", "anthropic", "nvidia"]
    if provider not in valid_providers:
        console.print(f"[red]Unknown provider: {provider}[/]")
        console.print(f"[dim]Valid providers: {', '.join(valid_providers)}[/]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold]Configure {provider.title()}[/]\n")

    # Prompt for API key (masked input)
    try:
        import getpass
        api_key = getpass.getpass(f"  API Key for {provider.title()}: ")
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Setup cancelled.[/]")
        raise typer.Exit(code=130)

    if not api_key or not api_key.strip():
        console.print("[red]No API key provided.[/]")
        raise typer.Exit(code=1)

    api_key = api_key.strip()

    # Validate the key by making a test request
    console.print(f"\n  [dim]Validating {provider.title()} credentials...[/]")
    valid, message = _validate_provider(provider, api_key)

    if not valid:
        console.print(f"  [red]✗ Validation failed: {message}[/]")
        console.print("  [dim]Credential was NOT saved.[/]")
        raise typer.Exit(code=1)

    # Save the credential
    store = CredentialStore()
    store.save_credential(provider, api_key)
    console.print(f"  [green]✓ Credential saved for {provider.title()}[/]")
    console.print(f"  [dim]Key: {mask_secret(api_key)}[/]")
    console.print(f"  [dim]Stored at: {store.config_dir / 'credentials.json'}[/]\n")


@auth_app.command("status")
def auth_status() -> None:
    """Show provider connection status and configured credentials."""
    resolver = CredentialResolver()

    table = Table(title="Provider Status", border_style="blue")
    table.add_column("Provider", style="bold")
    table.add_column("Status")
    table.add_column("Key", style="dim")
    table.add_column("Source")

    providers = ["openrouter", "openai", "anthropic", "nvidia", "ollama"]
    for prov in providers:
        cred = resolver.resolve(prov)
        if cred and cred.is_valid:
            status = "[green]✓ Configured[/]"
            key_display = mask_secret(cred.api_key)
            source = cred.source
        else:
            status = "[dim]— Not configured[/]"
            key_display = "—"
            source = "—"
        table.add_row(prov.title(), status, key_display, source)

    console.print(table)

    # Show harness free mode status
    console.print("\n[bold]Harness Free Mode[/]")
    console.print("  [dim]Uses OpenRouter's free-model router for zero-cost coding[/]")


@auth_app.command("logout")
def auth_logout(
    provider: str = typer.Argument("openrouter", help="Provider to disconnect"),
) -> None:
    """Remove stored credentials for a provider."""
    provider = provider.lower().strip()
    store = CredentialStore()
    removed = store.remove_credential(provider)
    if removed:
        console.print(f"[green]✓ Credential removed for {provider.title()}[/]")
    else:
        console.print(f"[dim]No credential found for {provider.title()}[/]")


def _validate_provider(provider: str, api_key: str) -> tuple[bool, str]:
    """Validate a provider credential with a minimal API call.

    Returns (is_valid, message).
    """
    import asyncio

    async def _check() -> tuple[bool, str]:
        try:
            import httpx
        except ImportError:
            return True, "httpx not available, skipping validation"

        if provider == "openrouter":
            return await _validate_openrouter(api_key)
        elif provider == "nvidia":
            return await _validate_nvidia(api_key)
        elif provider == "openai":
            return await _validate_openai(api_key)
        elif provider == "anthropic":
            return True, "Anthropic validation not yet implemented (saved anyway)"
        return True, "Validation not implemented"

    try:
        return asyncio.run(_check())
    except Exception as e:
        return False, str(e)


async def _validate_openrouter(api_key: str) -> tuple[bool, str]:
    """Validate an OpenRouter API key."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                free_models = [
                    m for m in models
                    if float(m.get("pricing", {}).get("prompt", "1")) == 0
                ]
                return True, f"Connected — {len(models)} models, {len(free_models)} free"
            elif resp.status_code == 401:
                return False, "Invalid API key (401 Unauthorized)"
            elif resp.status_code == 403:
                return False, "Access denied (403 Forbidden)"
            else:
                return False, f"HTTP {resp.status_code}"
        except httpx.TimeoutException:
            return True, "Timeout during validation (key may still be valid)"
        except httpx.ConnectError:
            return True, "Network unreachable (key saved, will retry on use)"


async def _validate_nvidia(api_key: str) -> tuple[bool, str]:
    """Validate an NVIDIA NIM API key."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                "https://integrate.api.nvidia.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                count = len(data.get("data", []))
                return True, f"Connected — {count} models available"
            elif resp.status_code in (401, 403):
                return False, f"Authentication failed ({resp.status_code})"
            else:
                return False, f"HTTP {resp.status_code}"
        except httpx.TimeoutException:
            return True, "Timeout during validation (key may still be valid)"
        except httpx.ConnectError:
            return True, "Network unreachable (key saved, will retry on use)"


async def _validate_openai(api_key: str) -> tuple[bool, str]:
    """Validate an OpenAI API key."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                count = len(data.get("data", []))
                return True, f"Connected — {count} models available"
            elif resp.status_code in (401, 403):
                return False, f"Authentication failed ({resp.status_code})"
            else:
                return False, f"HTTP {resp.status_code}"
        except httpx.TimeoutException:
            return True, "Timeout during validation (key may still be valid)"
        except httpx.ConnectError:
            return True, "Network unreachable (key saved, will retry on use)"

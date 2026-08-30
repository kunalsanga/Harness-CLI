"""Tests for welcome screen provider state accuracy."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock
from io import StringIO

from harness_core.cli.interactive import InteractiveShell


class TestWelcomeScreenProviderState:
    """Test that welcome screen accurately reports provider state."""

    def _make_shell(self, provider=None, router=None, model="", provider_name="") -> InteractiveShell:
        """Create a shell with mocked state."""
        shell = InteractiveShell.__new__(InteractiveShell)
        shell.model = None
        shell.mode = "auto"
        shell.free = False
        shell.local = False
        shell.plain = False
        shell.max_iterations = 30
        shell.max_cost = None
        shell.workspace = "/tmp/test"
        shell.console = MagicMock()
        shell.session_id = None
        shell.session_manager = None
        shell.current_model = model
        shell.current_provider = provider_name
        shell.total_tool_calls = 0
        shell.total_iterations = 0
        shell.session_start = 0.0
        shell.task_start = 0.0
        shell.running = False
        shell.verbose = False
        shell._event_bus = None
        shell._agent_loop = None
        shell._provider = provider
        shell._router = router
        shell._task_aware = None
        return shell

    def test_welcome_shows_connected_when_provider_exists(self):
        """Welcome screen should show 'connected' when provider is available."""
        shell = self._make_shell(provider=MagicMock(), router=MagicMock())
        shell._print_welcome()

        # Check that console.print was called with connected message
        calls = shell.console.print.call_args_list
        provider_calls = [c for c in calls if "connected" in str(c)]
        assert len(provider_calls) > 0, "Should show 'connected' when provider is available"

    def test_welcome_shows_not_connected_when_no_provider(self):
        """Welcome screen should show 'not connected' when provider is absent."""
        shell = self._make_shell(provider=None, router=None)
        shell._print_welcome()

        calls = shell.console.print.call_args_list
        disconnected_calls = [c for c in calls if "No provider connected" in str(c)]
        assert len(disconnected_calls) > 0, "Should show 'No provider connected' when provider is absent"

    def test_welcome_shows_model_when_routed(self):
        """Welcome screen should show model name when routing has selected one."""
        shell = self._make_shell(
            provider=MagicMock(),
            router=MagicMock(),
            model="glm-5.3-flash",
            provider_name="openrouter",
        )
        shell._print_welcome()

        calls = shell.console.print.call_args_list
        model_calls = [c for c in calls if "glm-5.3-flash" in str(c)]
        assert len(model_calls) > 0, "Should show model name when available"

    def test_welcome_shows_model_routing_ready_when_router_exists(self):
        """Welcome screen should show routing ready when router is initialized."""
        shell = self._make_shell(provider=MagicMock(), router=MagicMock())
        shell._print_welcome()

        calls = shell.console.print.call_args_list
        routing_calls = [c for c in calls if "routing ready" in str(c)]
        assert len(routing_calls) > 0, "Should show 'routing ready' when router exists"

    def test_welcome_shows_routing_not_initialized_when_no_router(self):
        """Welcome screen should show routing not initialized when router is absent."""
        shell = self._make_shell(provider=MagicMock(), router=None)
        shell._print_welcome()

        calls = shell.console.print.call_args_list
        not_init_calls = [c for c in calls if "not initialized" in str(c)]
        assert len(not_init_calls) > 0, "Should show 'not initialized' when router is absent"

    def test_welcome_never_shows_stale_provider_state(self):
        """Welcome screen must not show stale state like 'No provider' when provider is connected."""
        shell = self._make_shell(provider=MagicMock(), router=MagicMock())
        shell._print_welcome()

        calls = shell.console.print.call_args_list
        # Should NOT show "No provider connected" when we have a provider
        stale_calls = [c for c in calls if "No provider connected" in str(c)]
        assert len(stale_calls) == 0, "Should NOT show 'No provider connected' when provider is available"

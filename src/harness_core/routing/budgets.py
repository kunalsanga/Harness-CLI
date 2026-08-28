"""Request budget management.

Enforces limits on iterations, tool calls, tokens, cost, and time.
Prevents runaway spending and protects against infinite loops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BudgetConfig:
    """Configurable budget limits for a task/session."""

    max_iterations: int = 30
    max_tool_calls: int = 100
    max_tokens: int = 500_000
    max_cost: float = 5.0  # dollars
    timeout_seconds: float = 600.0  # 10 minutes
    # Per-model limits
    max_cost_per_model: float | None = None
    max_calls_per_model: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetConfig:
        """Create from a config dict (e.g., from YAML)."""
        return cls(
            max_iterations=data.get("max_iterations", 30),
            max_tool_calls=data.get("max_tool_calls", 100),
            max_tokens=data.get("max_tokens", 500_000),
            max_cost=data.get("max_cost", 5.0),
            timeout_seconds=data.get("timeout_seconds", 600.0),
            max_cost_per_model=data.get("max_cost_per_model"),
            max_calls_per_model=data.get("max_calls_per_model"),
        )


@dataclass
class BudgetState:
    """Current usage against the budget."""

    iterations: int = 0
    tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    start_time: float = field(default_factory=time.time)
    # Per-model tracking
    model_calls: dict[str, int] = field(default_factory=dict)
    model_costs: dict[str, float] = field(default_factory=dict)


class BudgetManager:
    """Tracks and enforces request budgets."""

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self.config = config or BudgetConfig()
        self.state = BudgetState()

    def reset(self) -> None:
        """Reset all budget tracking."""
        self.state = BudgetState()

    def record_iteration(self) -> bool:
        """Record an iteration. Returns False if budget exceeded."""
        self.state.iterations += 1
        return self.state.iterations <= self.config.max_iterations

    def record_tool_call(self) -> bool:
        """Record a tool call. Returns False if budget exceeded."""
        self.state.tool_calls += 1
        return self.state.tool_calls <= self.config.max_tool_calls

    def record_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model_id: str | None = None,
    ) -> bool:
        """Record token usage. Returns False if budget exceeded."""
        self.state.total_input_tokens += input_tokens
        self.state.total_output_tokens += output_tokens
        if model_id:
            self.state.model_calls[model_id] = self.state.model_calls.get(model_id, 0) + 1
        return self.state.total_input_tokens + self.state.total_output_tokens <= self.config.max_tokens

    def record_cost(
        self,
        cost: float,
        model_id: str | None = None,
    ) -> bool:
        """Record cost. Returns False if budget exceeded."""
        self.state.total_cost += cost
        if model_id:
            self.state.model_costs[model_id] = self.state.model_costs.get(model_id, 0.0) + cost
        return self.state.total_cost <= self.config.max_cost

    def is_within_timeout(self) -> bool:
        """Check if we're still within the time budget."""
        elapsed = time.time() - self.state.start_time
        return elapsed < self.config.timeout_seconds

    def check_all(self) -> tuple[bool, str | None]:
        """Check all budgets. Returns (ok, reason_if_exceeded)."""
        if self.state.iterations > self.config.max_iterations:
            return False, f"Iteration limit reached ({self.config.max_iterations})"
        if self.state.tool_calls > self.config.max_tool_calls:
            return False, f"Tool call limit reached ({self.config.max_tool_calls})"
        if self.state.total_input_tokens + self.state.total_output_tokens > self.config.max_tokens:
            return False, f"Token limit reached ({self.config.max_tokens})"
        if self.state.total_cost > self.config.max_cost:
            return False, f"Cost limit reached (${self.config.max_cost:.2f})"
        if not self.is_within_timeout():
            return False, f"Timeout reached ({self.config.timeout_seconds}s)"
        return True, None

    def check_model_limit(self, model_id: str) -> tuple[bool, str | None]:
        """Check per-model limits."""
        if self.config.max_calls_per_model is not None:
            calls = self.state.model_calls.get(model_id, 0)
            if calls >= self.config.max_calls_per_model:
                return False, f"Model call limit reached for {model_id} ({self.config.max_calls_per_model})"
        if self.config.max_cost_per_model is not None:
            cost = self.state.model_costs.get(model_id, 0.0)
            if cost >= self.config.max_cost_per_model:
                return False, f"Model cost limit reached for {model_id} (${self.config.max_cost_per_model:.2f})"
        return True, None

    def estimated_task_cost(self) -> float:
        """Estimate remaining cost based on current burn rate."""
        elapsed = time.time() - self.state.start_time
        if elapsed <= 0:
            return 0.0
        burn_rate = self.state.total_cost / elapsed  # cost per second
        remaining = self.config.timeout_seconds - elapsed
        return max(0.0, burn_rate * remaining)

    def remaining_budget(self) -> dict[str, Any]:
        """Get remaining budget across all dimensions."""
        elapsed = time.time() - self.state.start_time
        return {
            "iterations_remaining": max(0, self.config.max_iterations - self.state.iterations),
            "tool_calls_remaining": max(0, self.config.max_tool_calls - self.state.tool_calls),
            "tokens_remaining": max(
                0,
                self.config.max_tokens - self.state.total_input_tokens - self.state.total_output_tokens,
            ),
            "cost_remaining": max(0.0, self.config.max_cost - self.state.total_cost),
            "time_remaining": max(0.0, self.config.timeout_seconds - elapsed),
            "total_tokens_used": self.state.total_input_tokens + self.state.total_output_tokens,
            "total_cost_used": round(self.state.total_cost, 6),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for observability."""
        return {
            "config": {
                "max_iterations": self.config.max_iterations,
                "max_tool_calls": self.config.max_tool_calls,
                "max_tokens": self.config.max_tokens,
                "max_cost": self.config.max_cost,
                "timeout_seconds": self.config.timeout_seconds,
            },
            "state": {
                "iterations": self.state.iterations,
                "tool_calls": self.state.tool_calls,
                "total_tokens": self.state.total_input_tokens + self.state.total_output_tokens,
                "total_cost": round(self.state.total_cost, 6),
                "elapsed_seconds": round(time.time() - self.state.start_time, 1),
            },
            "remaining": self.remaining_budget(),
        }

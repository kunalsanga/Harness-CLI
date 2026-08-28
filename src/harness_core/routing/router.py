"""Production-grade model router.

Selects the best model for a task using 14-dimension scoring,
health tracking, fallback chains, budget enforcement,
task classification, and model registry capabilities.
Supports multiple providers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
)
from harness_core.routing.budgets import BudgetConfig, BudgetManager
from harness_core.routing.fallback import FallbackConfig, FallbackEngine, FallbackResult
from harness_core.routing.health import ModelHealthTracker
from harness_core.routing.scoring import (
    ScoringContext,
    ScoringWeights,
    rank_models,
)

if TYPE_CHECKING:
    from harness_core.routing.task_aware import TaskAwareRouter


@dataclass
class RoutingDecision:
    """A recorded routing decision for observability."""

    task_description: str = ""
    selected_model: str = ""
    selected_provider: str = ""
    score: float = 0.0
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    routing_mode: str = "auto"
    timestamp: float = field(default_factory=time.time)


@dataclass
class RouterConfig:
    """Configuration for the model router."""

    routing_mode: str = "auto"  # auto, free, best, fast, local, cheap
    prefer_free: bool = False
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    # Maximum models to consider in the fallback chain
    max_fallback_chain: int = 4

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouterConfig:
        """Create from a config dict."""
        weights_data = data.get("scoring_weights", {})
        budget_data = data.get("budget", {})
        fallback_data = data.get("fallback", {})

        return cls(
            routing_mode=data.get("routing_mode", "auto"),
            prefer_free=data.get("prefer_free", False),
            scoring_weights=ScoringWeights(**weights_data) if weights_data else ScoringWeights(),
            budget=BudgetConfig.from_dict(budget_data) if budget_data else BudgetConfig(),
            fallback=FallbackConfig(
                max_fallback_models=fallback_data.get("max_fallback_models", 3),
            ) if fallback_data else FallbackConfig(),
            max_fallback_chain=data.get("max_fallback_chain", 4),
        )


class ModelRouter:
    """Routes requests to the best available model.

    Architecture:
        1. Discover models from all providers
        2. Filter by health and capability
        3. Classify task (via TaskAwareRouter if available)
        4. Score by 14 dimensions (task type, capability, history, etc.)
        5. Build fallback chain
        6. Execute with retry/fallback via FallbackEngine
        7. Track health and budget
    """

    def __init__(
        self,
        providers: list[ModelProvider] | None = None,
        config: RouterConfig | None = None,
        event_bus: EventBus | None = None,
        task_aware: TaskAwareRouter | None = None,
    ) -> None:
        self.providers: dict[str, ModelProvider] = {}
        for p in (providers or []):
            self.providers[p.name] = p
        self.config = config or RouterConfig()
        self.health = ModelHealthTracker()
        self.budget = BudgetManager(self.config.budget)
        self.fallback_engine = FallbackEngine(
            health_tracker=self.health,
            fallback_config=self.config.fallback,
        )
        self.event_bus = event_bus or EventBus()
        self.task_aware = task_aware
        self._model_cache: list[ModelInfo] = []
        self._last_refresh: float = 0.0
        self._routing_decisions: list[RoutingDecision] = []

    async def refresh_models(self, force: bool = False) -> list[ModelInfo]:
        """Discover models from all providers. Caches for 5 minutes."""
        now = time.time()
        if not force and self._model_cache and (now - self._last_refresh) < 300:
            return self._model_cache

        all_models: list[ModelInfo] = []
        for name, provider in self.providers.items():
            try:
                models = await provider.list_models()
                all_models.extend(models)
            except Exception:
                pass  # provider failure doesn't block routing

        # Enrich with health data
        for model in all_models:
            model.reliability = self.health.get_reliability(model.id)

        self._model_cache = all_models
        self._last_refresh = now

        await self.event_bus.emit(Event(
            type="router.models_refreshed",
            source="model_router",
            data={"count": len(all_models), "providers": list(self.providers.keys())},
        ))

        return all_models

    def _filter_models(
        self,
        models: list[ModelInfo],
        ctx: ScoringContext,
    ) -> list[ModelInfo]:
        """Filter models based on hard requirements."""
        filtered = []
        for m in models:
            # Must support tools if required
            if ctx.requires_tools and not m.supports_tools:
                continue
            # Must support vision if required
            if ctx.requires_vision and not m.supports_vision:
                continue
            # Must be healthy
            if not self.health.get_state(m.id).is_healthy:
                continue
            # Free-only mode
            if self.config.routing_mode == "free" and not m.is_free:
                continue
            # Local-only mode
            if self.config.routing_mode == "local" and not m.is_local:
                continue
            filtered.append(m)
        return filtered

    def _build_scoring_context(
        self,
        request: CompletionRequest,
    ) -> ScoringContext:
        """Build scoring context from a completion request.

        Uses TaskAwareRouter for task classification when available.
        Falls back to keyword heuristics otherwise.
        """
        # Extract task description from the last user message
        task_desc = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                task_desc = msg.get("content", "")
                break

        has_tools = bool(request.tools)
        needs_vision = False

        # Determine prefer_free based on routing mode
        prefer_free = self.config.prefer_free or self.config.routing_mode == "free"

        # Use TaskAwareRouter for classification if available
        task_type = ""
        classification_confidence = 0.0
        model_caps: dict[str, float] = {}
        hist_success = 0.0
        hist_latency = 0.0
        hist_efficiency = 0.0

        if self.task_aware is not None:
            task_type_obj, profile, confidence = self.task_aware.classify_task(request)
            task_type = task_type_obj.value if task_type_obj else ""
            classification_confidence = confidence
        else:
            # Fallback: keyword heuristics
            task_lower = task_desc.lower()
            if any(kw in task_lower for kw in ["fix", "bug", "error", "fail"]):
                task_type = "bug_fix"
            elif any(kw in task_lower for kw in ["implement", "create", "add", "build"]):
                task_type = "implementation"
            elif any(kw in task_lower for kw in ["refactor", "restructure", "clean"]):
                task_type = "refactoring"
            elif any(kw in task_lower for kw in ["explain", "research", "analyze"]):
                task_type = "research"
            elif any(kw in task_lower for kw in ["test", "coverage"]):
                task_type = "testing"

        # Detect task tags from content
        task_tags: list[str] = []
        task_lower = task_desc.lower()
        if any(kw in task_lower for kw in ["code", "fix", "bug", "implement", "refactor", "test"]):
            task_tags.append("coding")
        if any(kw in task_lower for kw in ["explain", "research", "analyze", "compare"]):
            task_tags.append("research")
        if any(kw in task_lower for kw in ["design", "architect", "plan", "reason"]):
            task_tags.append("reasoning")

        # Estimate context size from messages
        est_tokens = sum(
            len(str(msg.get("content", ""))) // 4 for msg in request.messages
        ) + 1000

        # User-selected model
        user_model = request.model or ""

        return ScoringContext(
            task_description=task_desc,
            requires_tools=has_tools,
            requires_vision=needs_vision,
            estimated_context_tokens=est_tokens,
            prefer_free=prefer_free,
            routing_mode=self.config.routing_mode,
            task_tags=task_tags,
            # New 14-dimension fields
            task_type=task_type,
            classification_confidence=classification_confidence,
            model_capability_scores=model_caps,
            historical_success_rate=hist_success,
            historical_avg_latency_ms=hist_latency,
            historical_tool_efficiency=hist_efficiency,
            user_selected_model=user_model,
        )

    async def select_models(
        self,
        request: CompletionRequest,
    ) -> list[tuple[str, ModelProvider]]:
        """Select an ordered chain of (model_id, provider) for fallback.

        Returns at least one model if any are available.
        """
        models = await self.refresh_models()
        ctx = self._build_scoring_context(request)

        # Filter
        filtered = self._filter_models(models, ctx)
        if not filtered:
            # Fallback: use any model with tools support
            filtered = [m for m in models if m.supports_tools]

        # Score and rank
        ranked = rank_models(filtered, ctx, self.config.scoring_weights)

        # Build chain
        chain: list[tuple[str, ModelProvider]] = []
        for model, score in ranked[: self.config.max_fallback_chain]:
            provider = self.providers.get(model.provider)
            if provider is None:
                continue

            # Check per-model budget
            ok, _ = self.budget.check_model_limit(model.id)
            if not ok:
                continue

            chain.append((model.id, provider))

            # Record decision
            if len(chain) == 1:
                decision = RoutingDecision(
                    task_description=ctx.task_description[:200],
                    selected_model=model.id,
                    selected_provider=model.provider,
                    score=score,
                    routing_mode=self.config.routing_mode,
                )
                self._routing_decisions.append(decision)

                await self.event_bus.emit(Event(
                    type="routing.decision",
                    source="model_router",
                    data={
                        "model": model.id,
                        "provider": model.provider,
                        "score": round(score, 3),
                        "mode": self.config.routing_mode,
                        "alternatives": [
                            {"model": m.id, "score": round(s, 3)}
                            for m, s in ranked[1:5]
                        ],
                    },
                ))

        if not chain:
            # Absolute fallback: use the first available provider with any model
            for name, provider in self.providers.items():
                try:
                    models_list = await provider.list_models()
                    tool_models = [m for m in models_list if m.supports_tools]
                    if tool_models:
                        chain.append((tool_models[0].id, provider))
                        break
                except Exception:
                    continue

        return chain

    async def execute(
        self,
        request: CompletionRequest,
    ) -> FallbackResult:
        """Route and execute a completion request.

        Selects models, builds fallback chain, and executes with retry/fallback.
        """
        # Check overall budget
        ok, reason = self.budget.check_all()
        if not ok:
            return FallbackResult(final_error=f"Budget exceeded: {reason}")

        chain = await self.select_models(request)
        if not chain:
            return FallbackResult(final_error="No available models")

        result = await self.fallback_engine.execute(request, chain)

        # Record budget usage
        if result.succeeded and result.response:
            usage = result.response.usage
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            self.budget.record_tokens(input_tokens, output_tokens, result.model_used)

        return result

    def get_routing_decisions(self) -> list[RoutingDecision]:
        """Get all recorded routing decisions."""
        return list(self._routing_decisions)

    def get_health_report(self) -> dict[str, Any]:
        """Get a health report for all tracked models."""
        return self.health.to_dict()

    def get_budget_status(self) -> dict[str, Any]:
        """Get current budget status."""
        return self.budget.to_dict()

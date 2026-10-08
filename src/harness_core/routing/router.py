"""Production-grade model router.

Selects the best model for a task using 14-dimension scoring,
health tracking, fallback chains, budget enforcement,
task classification, and model registry capabilities.
Supports multiple providers.
"""

from __future__ import annotations

import inspect
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
    allow_paid_models: bool = False
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
            allow_paid_models=data.get("allow_paid_models", False),
            scoring_weights=ScoringWeights(**weights_data) if weights_data else ScoringWeights(),
            budget=BudgetConfig.from_dict(budget_data) if budget_data else BudgetConfig(),
            fallback=FallbackConfig(
                max_fallback_models=fallback_data.get("max_fallback_models", 3),
                model_attempt_timeout_seconds=fallback_data.get(
                    "model_attempt_timeout_seconds", 90.0
                ),
                total_timeout_seconds=fallback_data.get("total_timeout_seconds", 120.0),
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
        self.event_bus = event_bus or EventBus()
        self.budget = BudgetManager(self.config.budget)
        self.fallback_engine = FallbackEngine(
            health_tracker=self.health,
            fallback_config=self.config.fallback,
            event_bus=self.event_bus,
        )
        self.task_aware = task_aware
        self._model_cache: list[ModelInfo] = []
        self._last_refresh: float = 0.0
        self._routing_decisions: list[RoutingDecision] = []
        self._discovery_errors: list[dict[str, str]] = []
        self.current_model: str = ""
        self.attempt_count: int = 0
        self.failed_models_for_current_request: list[str] = []
        self.failover_reason: str = ""

    async def refresh_models(self, force: bool = False) -> list[ModelInfo]:
        """Discover models from all providers. Caches for 5 minutes.

        Discovery failures are recorded and never silently rewritten as
        "the provider has zero models." Configured static models still merge
        in so providers without live discovery keep working.
        """
        now = time.time()
        if not force and self._model_cache and (now - self._last_refresh) < 300:
            return self._model_cache

        all_models: list[ModelInfo] = []
        discovery_errors: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for name, provider in self.providers.items():
            discovered = 0
            try:
                models = await provider.list_models()
                for model in models:
                    key = (model.provider or name, model.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    all_models.append(model)
                    discovered += 1
            except Exception as exc:
                category = getattr(getattr(exc, "category", None), "value", None) or "unknown"
                detail = str(exc)[:300]
                discovery_errors.append({
                    "provider": name,
                    "category": str(category),
                    "error": detail,
                })
                await self.event_bus.emit(Event(
                    type="router.discovery_failed", source="model_router",
                    data={"provider": name, "category": category, "error": detail},
                ))
            for model in getattr(provider, "configured_models", []) or []:
                key = (model.provider or name, model.id)
                if key in seen:
                    continue
                seen.add(key)
                all_models.append(model)

        # Enrich with health data
        for model in all_models:
            model.reliability = self.health.get_reliability(model.id)

        self._model_cache = all_models
        self._discovery_errors = discovery_errors
        self._last_refresh = now

        await self.event_bus.emit(Event(
            type="router.models_refreshed",
            source="model_router",
            data={
                "count": len(all_models),
                "providers": list(self.providers.keys()),
                "discovery_errors": len(discovery_errors),
            },
        ))

        return all_models

    def _filter_models(
        self,
        models: list[ModelInfo],
        ctx: ScoringContext,
    ) -> list[ModelInfo]:
        """Filter models based on hard requirements.

        Availability/health is a hard constraint — unavailable models
        are excluded before scoring, not merely scored lower.
        """
        filtered = []
        for m in models:
            # Must support tools if required
            if ctx.requires_tools and m.supports_tools is not True:
                continue
            # Must support vision if required
            if ctx.requires_vision and m.supports_vision is not True:
                continue
            if ctx.requires_structured_output and m.supports_structured_output is not True:
                continue
            if ctx.requires_streaming and m.supports_streaming is not True:
                continue
            if ctx.requires_reasoning and m.supports_reasoning is not True:
                continue
            if ctx.estimated_context_tokens > 0 and m.context_window and m.context_window < ctx.estimated_context_tokens:
                continue
            # Health is a hard constraint — skip unavailable models entirely
            health_state = self.health.get_state(m.id)
            if health_state.is_unavailable:
                continue
            if not health_state.is_healthy:
                continue
            # Free-only mode
            if ctx.routing_mode == "free" and not m.is_free:
                continue
            if not self.config.allow_paid_models and not m.is_free and not m.is_local:
                continue
            # Local-only mode
            if ctx.routing_mode == "local" and not m.is_local:
                continue
            filtered.append(m)
        return filtered

    def _build_scoring_context(
        self,
        request: CompletionRequest,
        active_routing_mode: str,
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
        needs_vision = bool(request.metadata.get("requires_vision", False))

        # Determine prefer_free based on routing mode
        prefer_free = self.config.prefer_free or active_routing_mode == "free"

        # Phase 10: honor per-agent model_policy values passed through
        # routing_mode_override (e.g. "reasoning_high", "coding", "fast").
        # Role policies are mapped onto the existing scoring vocabulary so
        # the 14-dimension ranking actually prefers the right model class.
        policy = (active_routing_mode or "auto").lower()
        policy_tags_by_mode = {
            "reasoning_high": ("reasoning", "plan"),
            "reasoning": ("reasoning", "plan"),
            "coding": ("coding",),
            "implementation": ("coding",),
            "fast": ("fast",),
            "cheap": ("fast",),
        }
        policy_tags = policy_tags_by_mode.get(policy, ())
        if policy in ("fast", "cheap") and not request.model:
            # Fast/cheap agents prefer free models unless the user pinned one.
            prefer_free = True

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
        # Policy tags take precedence over content heuristics.
        for tag in policy_tags:
            if tag not in task_tags:
                task_tags.append(tag)

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
            requires_structured_output=bool(request.metadata.get("requires_structured_output", False)),
            requires_streaming=bool(request.metadata.get("requires_streaming", False)),
            requires_reasoning=bool(request.metadata.get("requires_reasoning", False)),
            estimated_context_tokens=est_tokens,
            prefer_free=prefer_free,
            routing_mode=active_routing_mode,
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
        routing_mode_override: str | None = None,
    ) -> list[tuple[str, ModelProvider]]:
        """Select an ordered chain of (model_id, provider) for fallback.

        Returns compatible, healthy candidates from configured providers.
        """
        active_routing_mode = routing_mode_override or self.config.routing_mode
        ctx = self._build_scoring_context(request, active_routing_mode)

        models = list(await self.refresh_models())
        for provider in self.providers.values():
            hints = provider.routing_hints(active_routing_mode)
            if inspect.isawaitable(hints):
                hints = await hints
            if isinstance(hints, list):
                models.extend(hints)
        models = list({(model.provider, model.id): model for model in models}.values())

        # ── Explicit Model Pinning ───────────────────────────────────────────
        if ctx.user_selected_model:
            target_id = ctx.user_selected_model
            p: ModelProvider | None = None
            if "::" in target_id:
                prov_name, target_id = target_id.split("::", 1)
                p = self.providers.get(prov_name.lower())
            else:
                # Prefer an exact catalog match (e.g. openrouter/free) before
                # interpreting provider/model slash forms like groq/compound.
                exact = [m for m in models if m.id == target_id]
                if exact:
                    p = self.providers.get(exact[0].provider)
                elif "/" in target_id:
                    prov_name, maybe_model = target_id.split("/", 1)
                    if prov_name.lower() in self.providers:
                        p = self.providers.get(prov_name.lower())
                        target_id = maybe_model
            if p is None:
                matches = self._filter_models([m for m in models if m.id == target_id], ctx)
                if len(matches) == 1:
                    p = self.providers.get(matches[0].provider)
            
            if p:
                await self.event_bus.emit(Event(
                    type="routing.decision",
                    source="model_router",
                    data={
                        "model": target_id,
                        "provider": p.name,
                        "score": 1.0,
                        "mode": "pinned",
                        "alternatives": [],
                    },
                ))
                return [(target_id, p)]

        # Provider routing hints are normalized ModelInfo candidates. Free
        # routes and ordinary models follow the same health/capability checks.
        if active_routing_mode == "free":
            return await self._build_free_chain(request, self._filter_models(models, ctx))

        # Filter
        filtered = self._filter_models(models, ctx)
        if not filtered:
            # Non-free fallback: use any model with tools support
            filtered = [
                m for m in models
                if m.supports_tools is True
                and (m.is_free or m.is_local or self.config.allow_paid_models)
            ]

        # Score and rank
        ranked = rank_models(filtered, ctx, self.config.scoring_weights)

        # Build chain
        chain: list[tuple[str, ModelProvider]] = []
        chain_models: list[ModelInfo] = []
        chain_ids: set[tuple[str, str]] = set()
        primary_recorded = False

        def _record_primary(model: ModelInfo, score: float) -> None:
            decision = RoutingDecision(
                task_description=ctx.task_description[:200],
                selected_model=model.id,
                selected_provider=model.provider,
                score=score,
                routing_mode=active_routing_mode,
            )
            self._routing_decisions.append(decision)

        async def _emit_primary(model: ModelInfo, score: float) -> None:
            await self.event_bus.emit(Event(
                type="routing.decision",
                source="model_router",
                data={
                    "model": model.id,
                    "provider": model.provider,
                    "score": round(score, 3),
                    "mode": active_routing_mode,
                    "alternatives": [
                        {"model": m.id, "score": round(s, 3)}
                        for m, s in ranked[1:5]
                    ],
                },
            ))

        async def _try_add(model: ModelInfo, score: float) -> bool:
            nonlocal primary_recorded
            key = (model.provider, model.id)
            if key in chain_ids:
                return False
            provider = self.providers.get(model.provider)
            if provider is None:
                return False
            # Check per-model budget
            ok, _ = self.budget.check_model_limit(model.id)
            if not ok:
                return False
            chain_ids.add(key)
            chain.append((model.id, provider))
            chain_models.append(model)
            if not primary_recorded:
                primary_recorded = True
                _record_primary(model, score)
                await _emit_primary(model, score)
            return True

        limit = self.config.max_fallback_chain
        for model, score in ranked[:limit]:
            await _try_add(model, score)

        # Free-model safety net (non-free modes): paid models may require
        # credits this account does not have (402 Payment Required). Ensure
        # top-ranked free models are reachable in the chain so fallback can
        # land on a usable model instead of failing the whole task.
        if active_routing_mode != "free" and ranked:
            min_free_slots = max(2, limit // 2)
            free_in_chain = sum(1 for m in chain_models if m.is_free)
            if free_in_chain < min_free_slots:
                for model, score in ranked:
                    if free_in_chain >= min_free_slots:
                        break
                    if not model.is_free:
                        continue
                    if await _try_add(model, score):
                        free_in_chain += 1

        if not chain:
            if active_routing_mode == "free":
                # In free mode: DO NOT fall back to paid models
                # Return empty chain
                return []
            # Absolute fallback: use the first available provider with any model
            for name, provider in self.providers.items():
                try:
                    models_list = await provider.list_models()
                    tool_models = [m for m in models_list if m.supports_tools is True and (m.is_free or m.is_local or self.config.allow_paid_models)]
                    if tool_models:
                        chain.append((tool_models[0].id, provider))
                        break
                except Exception:
                    continue

        return chain

    async def _build_free_chain(
        self,
        request: CompletionRequest,
        models: list[ModelInfo],
    ) -> list[tuple[str, ModelProvider]]:
        """Build a provider-neutral chain from discovered free model metadata."""
        free_models = [m for m in models if m.is_free and m.supports_tools is True]
        # Concrete free models first; provider dynamic routes (e.g. openrouter/free) last.
        free_models.sort(key=lambda model: (
            "dynamic-route" in model.tags,
            -model.context_window,
            model.provider,
            model.id,
        ))
        chain: list[tuple[str, ModelProvider]] = []
        seen: set[tuple[str, str]] = set()
        for model in free_models[: self.config.max_fallback_chain]:
            provider = self.providers.get(model.provider)
            if provider is None or (model.provider, model.id) in seen:
                continue
            if not self.health.get_state(model.id).is_healthy:
                continue
            seen.add((model.provider, model.id))
            chain.append((model.id, provider))
        if chain:
            first_model, first_provider = chain[0]
            self._routing_decisions.append(RoutingDecision(
                selected_model=first_model, selected_provider=first_provider.name,
                score=1.0, routing_mode="free",
            ))
            await self.event_bus.emit(Event(
                type="routing.decision", source="model_router",
                data={"model": first_model, "provider": first_provider.name,
                      "score": 1.0, "mode": "free",
                      "alternatives": [{"model": model, "provider": provider.name} for model, provider in chain[1:]]},
            ))
        return chain

    async def execute(
        self,
        request: CompletionRequest,
        routing_mode_override: str | None = None,
    ) -> FallbackResult:
        """Route and execute a completion request.

        Args:
            request: The completion request.
            routing_mode_override: Optional per-request routing mode override
                (e.g., from an AgentProfile).
        """
        active_routing_mode = routing_mode_override or self.config.routing_mode

        # Check overall budget
        ok, reason = self.budget.check_all()
        if not ok:
            return FallbackResult(final_error=f"Budget exceeded: {reason}")

        chain = await self.select_models(request, routing_mode_override)
        if not chain:
            if self._discovery_errors and not self._model_cache:
                details = "; ".join(
                    f"{item['provider']}: {item.get('error') or item.get('category', 'discovery failed')}"
                    for item in self._discovery_errors
                )
                return FallbackResult(
                    final_error=(
                        "Model discovery failed and no configured models are available. "
                        f"{details}"
                    ),
                    error_category="provider",
                )
            if active_routing_mode == "free":
                return FallbackResult(
                    final_error=(
                        "No configured free model is currently available. "
                        "No paid model was used. "
                        "Check /models, wait for cooldowns, or disable free mode."
                    )
                )
            return FallbackResult(final_error="No available models")

        self.current_model = chain[0][0]
        self.attempt_count = 0
        self.failed_models_for_current_request = []
        self.failover_reason = ""
        result = await self.fallback_engine.execute(request, chain)
        self.attempt_count = result.attempt_count
        self.failed_models_for_current_request = list(dict.fromkeys(
            attempt["model"] for attempt in result.attempts
            if attempt.get("status") == "error"
        ))
        for attempt in reversed(result.attempts):
            if attempt.get("status") == "error":
                self.failover_reason = str(attempt.get("classification", ""))
                break
        if result.succeeded:
            self.current_model = result.model_used

        # Record budget usage
        if result.succeeded and result.response:
            usage = result.response.token_usage
            if usage is not None:
                self.budget.record_tokens(
                    usage.input_tokens or 0,
                    usage.output_tokens or 0,
                    result.model_used,
                )
                if self.event_bus:
                    from harness_core.observability.events import Event
                    import asyncio
                    asyncio.create_task(self.event_bus.emit(Event(
                        type="model.usage",
                        source="model_router",
                        data={
                            "input_tokens": usage.input_tokens or 0,
                            "output_tokens": usage.output_tokens or 0,
                            "total_tokens": usage.total_tokens or ((usage.input_tokens or 0) + (usage.output_tokens or 0)),
                        }
                    )))

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

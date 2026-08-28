"""Model routing subsystem.

Provides intelligent model selection, fallback, retry, budget enforcement,
and health tracking for multi-provider model routing.
"""

from harness_core.routing.budgets import BudgetConfig, BudgetManager
from harness_core.routing.fallback import (
    FallbackConfig,
    FallbackEngine,
    FallbackResult,
    classify_error,
)
from harness_core.routing.health import (
    HealthEvent,
    ModelHealthState,
    ModelHealthTracker,
)
from harness_core.routing.router import (
    RouterConfig,
    ModelRouter,
    RoutingDecision,
)
from harness_core.routing.scoring import (
    ScoringContext,
    ScoringWeights,
    compute_model_score,
    rank_models,
)

__all__ = [
    "BudgetConfig",
    "BudgetManager",
    "FallbackConfig",
    "FallbackEngine",
    "FallbackResult",
    "HealthEvent",
    "ModelHealthState",
    "ModelHealthTracker",
    "RouterConfig",
    "ModelRouter",
    "RoutingDecision",
    "ScoringContext",
    "ScoringWeights",
    "classify_error",
    "compute_model_score",
    "rank_models",
]

"""M5+ Advanced Agent Engine — multi-agent orchestration system."""

from .domain import (
    AgentMessage,
    AgentResult,
    AgentRole,
    AgentStatus,
    MessageType,
    ReviewVerdict,
    SubTask,
    TaskGraph,
    TaskStatus,
)
from .registry import AgentConfig, AgentRegistry
from .executor import AgentExecutor
from .orchestrator import AgentBudget, ExecutionMode, Orchestrator, OrchestratorResult
from .parallel import FileOwnershipTracker, FileLockState, ParallelScheduler
from .cancellation import CancellationHandler, GracefulShutdown, OperationTimeout

__all__ = [
    # Domain
    "AgentRole",
    "AgentStatus",
    "TaskStatus",
    "MessageType",
    "ReviewVerdict",
    "SubTask",
    "TaskGraph",
    "AgentResult",
    "AgentMessage",
    # Registry
    "AgentConfig",
    "AgentRegistry",
    # Executor
    "AgentExecutor",
    # Orchestrator
    "ExecutionMode",
    "AgentBudget",
    "Orchestrator",
    "OrchestratorResult",
    # Parallel
    "FileOwnershipTracker",
    "FileLockState",
    "ParallelScheduler",
    # Cancellation
    "CancellationHandler",
    "GracefulShutdown",
    "OperationTimeout",
]

"""Built-in benchmark tasks."""

from harness_core.benchmarks.tasks.tool_use import TOOL_USE_TASKS
from harness_core.benchmarks.tasks.navigation import NAVIGATION_TASKS
from harness_core.benchmarks.tasks.coding import CODING_TASKS
from harness_core.benchmarks.tasks.debugging import DEBUGGING_TASKS

ALL_BENCHMARK_TASKS = TOOL_USE_TASKS + NAVIGATION_TASKS + CODING_TASKS + DEBUGGING_TASKS

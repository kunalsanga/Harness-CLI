"""
Agent benchmark engine — runs controlled benchmarks in isolated workspaces.

Every benchmark executes in a temporary directory. Never touches the user's project.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from harness_core.benchmarks.scoring import BenchmarkScoringWeights, aggregate_results
from harness_core.benchmarks.types import (
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkSuiteResult,
    BenchmarkTask,
)
from harness_core.providers.base import CompletionRequest, ModelProvider


class AgentBenchmarkEngine:
    """Runs agent benchmarks in isolated temporary workspaces.

    Safety: Every benchmark runs in a temp directory that is cleaned up after.
    """

    def __init__(
        self,
        providers: dict[str, ModelProvider] | None = None,
        scoring_weights: BenchmarkScoringWeights | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self.providers = providers or {}
        self.scoring_weights = scoring_weights or BenchmarkScoringWeights()
        self.event_bus = event_bus

    async def run_task(
        self,
        task: BenchmarkTask,
        model_id: str,
        provider_name: str,
    ) -> BenchmarkResult:
        """Run a single benchmark task in an isolated workspace."""
        provider = self.providers.get(provider_name)
        if provider is None:
            return BenchmarkResult(
                task_name=task.name,
                category=task.category.value,
                model_id=model_id,
                provider=provider_name,
                error=f"Provider '{provider_name}' not available",
            )

        # Create isolated workspace
        workspace = Path(tempfile.mkdtemp(prefix="harness_bench_"))
        start_time = time.monotonic()

        try:
            # Set up workspace files
            for rel_path, content in task.setup_files.items():
                file_path = workspace / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            # Build the task prompt
            task_prompt = self._build_task_prompt(task, workspace)

            # Run the agent loop
            result = await self._run_agent_loop(
                prompt=task_prompt,
                workspace=workspace,
                model_id=model_id,
                provider=provider,
                max_iterations=task.max_iterations,
                timeout_seconds=task.timeout_seconds,
            )

            # Score the result
            elapsed_ms = (time.monotonic() - start_time) * 1000
            result.task_name = task.name
            result.category = task.category.value
            result.model_id = model_id
            result.provider = provider_name
            result.latency_ms = elapsed_ms

            # Check success criteria
            result = self._evaluate_result(result, task, workspace)

            return result

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return BenchmarkResult(
                task_name=task.name,
                category=task.category.value,
                model_id=model_id,
                provider=provider_name,
                latency_ms=elapsed_ms,
                error="Timeout",
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return BenchmarkResult(
                task_name=task.name,
                category=task.category.value,
                model_id=model_id,
                provider=provider_name,
                latency_ms=elapsed_ms,
                error=str(e),
            )
        finally:
            # Always clean up the workspace
            shutil.rmtree(workspace, ignore_errors=True)

    async def run_suite(
        self,
        tasks: list[BenchmarkTask],
        model_id: str,
        provider_name: str,
    ) -> BenchmarkSuiteResult:
        """Run a full benchmark suite and aggregate results."""
        results = []
        for task in tasks:
            result = await self.run_task(task, model_id, provider_name)
            results.append(result)

        return aggregate_results(results, self.scoring_weights)

    def _build_task_prompt(self, task: BenchmarkTask, workspace: Path) -> str:
        """Build the prompt for a benchmark task."""
        parts = [task.description]
        if task.instructions:
            parts.append(f"\nInstructions:\n{task.instructions}")
        parts.append(f"\nWorkspace: {workspace}")
        return "\n".join(parts)

    async def _run_agent_loop(
        self,
        prompt: str,
        workspace: Path,
        model_id: str,
        provider: ModelProvider,
        max_iterations: int = 15,
        timeout_seconds: int = 120,
    ) -> BenchmarkResult:
        """Run a simplified agent loop for benchmarking."""
        tool_calls_count = 0
        iterations = 0
        total_tokens = 0
        files_modified = 0

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]

        for i in range(max_iterations):
            iterations = i + 1

            request = CompletionRequest(
                messages=messages,
                model=model_id,
                tools=self._get_tool_schemas(),
            )

            try:
                response = await asyncio.wait_for(
                    provider.generate(request),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise
            except Exception:
                break

            # Track usage
            usage = response.usage
            total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

            # Process tool calls
            if response.tool_calls:
                tool_calls_count += len(response.tool_calls)

                # Execute tools
                tool_results = []
                for tc in response.tool_calls:
                    result = self._execute_tool(tc, workspace)
                    tool_results.append(result)
                    if tc.get("function", {}).get("name", "").startswith("write_"):
                        files_modified += 1

                # Build next message
                messages.append({"role": "assistant", "content": response.content or "", "tool_calls": response.tool_calls})
                for tc, tr in zip(response.tool_calls, tool_results):
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tr})
            else:
                # No tool calls — model is done
                break

        return BenchmarkResult(
            tokens_used=total_tokens,
            tool_calls=tool_calls_count,
            iterations=iterations,
            files_modified=files_modified,
        )

    def _execute_tool(self, tool_call: dict, workspace: Path) -> str:
        """Execute a tool call in the benchmark workspace."""
        func_name = tool_call.get("function", {}).get("name", "")
        try:
            args = __import__("json").loads(tool_call.get("function", {}).get("arguments", "{}"))
        except Exception:
            args = {}

        try:
            if func_name == "read_file":
                path = workspace / args.get("path", "")
                if path.exists():
                    return path.read_text(encoding="utf-8", errors="replace")[:10000]
                return f"Error: File not found: {args.get('path')}"
            elif func_name == "write_file":
                path = workspace / args.get("path", "")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(args.get("content", ""), encoding="utf-8")
                return f"File written: {args.get('path')}"
            elif func_name == "list_files":
                path = workspace / args.get("path", ".")
                if path.is_dir():
                    entries = sorted(p.name for p in path.iterdir())
                    return "\n".join(entries[:200])
                return f"Error: Not a directory: {args.get('path')}"
            elif func_name == "run_command":
                return "Benchmark: command execution not allowed in isolated workspace"
            else:
                return f"Tool '{func_name}' executed"
        except Exception as e:
            return f"Error: {e}"

    def _get_tool_schemas(self) -> list[dict]:
        """Get minimal tool schemas for benchmarking."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                    },
                },
            },
        ]

    def _evaluate_result(
        self,
        result: BenchmarkResult,
        task: BenchmarkTask,
        workspace: Path,
    ) -> BenchmarkResult:
        """Evaluate a benchmark result against success criteria."""
        # Check if expected files exist
        files_present = True
        for expected_file in task.expected_files:
            if not (workspace / expected_file).exists():
                files_present = False
                break

        # Basic scoring based on category
        if task.category == BenchmarkCategory.TOOL_USE:
            result.tool_use_score = 1.0 if result.tool_calls > 0 else 0.0
            result.success = result.tool_calls > 0 and files_present
        elif task.category == BenchmarkCategory.REPOSITORY_NAVIGATION:
            result.navigation_score = 1.0 if result.tool_calls > 0 else 0.0
            result.success = result.tool_calls > 0
        elif task.category == BenchmarkCategory.CODE_GENERATION:
            result.coding_score = 1.0 if files_present else 0.3
            result.success = files_present
        elif task.category == BenchmarkCategory.BUG_FIXING:
            result.coding_score = 1.0 if files_present else 0.3
            result.recovery_score = 1.0 if result.iterations > 1 else 0.5
            result.success = files_present
        elif task.category == BenchmarkCategory.DEBUGGING:
            result.recovery_score = 1.0 if result.tool_calls > 2 else 0.5
            result.success = result.tool_calls > 0
        elif task.category == BenchmarkCategory.ERROR_RECOVERY:
            result.recovery_score = 1.0 if result.success else 0.0
        elif task.category == BenchmarkCategory.CONTEXT_HANDLING:
            result.context_score = 1.0 if result.tool_calls > 0 else 0.0
            result.success = result.tool_calls > 0
        elif task.category == BenchmarkCategory.PLANNING:
            result.planning_score = 1.0 if result.iterations >= 2 else 0.5
            result.success = result.iterations >= 1
        elif task.category == BenchmarkCategory.VERIFICATION:
            result.verification_score = 1.0 if result.tool_calls > 0 else 0.0
            result.success = result.tool_calls > 0

        # Overall score
        scores = [
            s for s in [
                result.coding_score,
                result.tool_use_score,
                result.navigation_score,
                result.recovery_score,
                result.context_score,
                result.verification_score,
                result.planning_score,
            ]
            if s is not None
        ]
        result.score = sum(scores) / len(scores) if scores else 0.0

        return result

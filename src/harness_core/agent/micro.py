"""Programmatic bounded micro-workers (milestone Part 12).

A clean, explicit registry for cheap deterministic work and cheap model
work, so trivial operations (path validation, candidate filtering, token
counting, summarization of bounded text) never hit the strongest model:

    cheap deterministic work  → registered python callables (no eval)
    cheap model work          → optional small-model adapter
    strong model work         → the main agent loop (unchanged)

Not a general code-execution mechanism: handlers are explicit registered
callables with declared names, and nothing executes strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from harness_core.context.pack import estimate_tokens

MicroHandler = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass
class MicroWorkerSpec:
    """A registered bounded operation."""

    name: str
    handler: MicroHandler
    description: str = ""
    max_output_tokens: int = 2_000


class MicroWorkerRegistry:
    """Registry of explicit bounded operations."""

    def __init__(self) -> None:
        self._handlers: dict[str, MicroWorkerSpec] = {}

    def register(self, spec: MicroWorkerSpec) -> None:
        """Register a handler. Re-registering a name replaces it."""
        self._handlers[spec.name] = spec

    def has(self, name: str) -> bool:
        return name in self._handlers

    def names(self) -> list[str]:
        return sorted(self._handlers)

    async def run(self, name: str, payload: dict[str, Any]) -> Any:
        """Execute a registered handler by name.

        Output is clamped to the spec's token bound (cheap work stays
        cheap). Unknown names raise KeyError — this is a programming error,
        not a runtime failure to swallow.
        """
        spec = self._handlers.get(name)
        if spec is None:
            raise KeyError(f"unknown micro-worker: {name}")
        result = await spec.handler(payload)
        out = spec.max_output_tokens
        if isinstance(result, str) and estimate_tokens(result) > out:
            result = result[: out * 4]
        return result


# ── Built-in handlers ────────────────────────────────────────────────
# Registered by default; deterministic, pure, and bounded.


async def _path_validate(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate paths against a workspace root — no I/O beyond stat."""
    from pathlib import Path

    root = Path(payload.get("workspace_root", "."))
    paths = payload.get("paths", [])
    valid: list[str] = []
    rejected: list[dict[str, str]] = []
    for raw in paths[:100]:  # bounded
        p = str(raw).replace("\\", "/")
        if not p or ".." in Path(p).parts:
            rejected.append({"path": str(raw)[:200], "reason": "traversal_or_empty"})
            continue
        try:
            resolved = (root / p).resolve()
            resolved.relative_to(root.resolve())
        except (OSError, RuntimeError, ValueError):
            rejected.append({"path": str(raw)[:200], "reason": "outside_workspace"})
            continue
        if not resolved.is_file():
            rejected.append({"path": str(raw)[:200], "reason": "not_found"})
            continue
        valid.append(p)
    return {"valid": valid, "rejected": rejected}


async def _token_count(payload: dict[str, Any]) -> dict[str, Any]:
    """Estimate token counts for texts using the shared estimator."""
    texts = payload.get("texts", [])
    counts = [estimate_tokens(t) for t in texts[:200]]
    return {"counts": counts, "total": sum(counts)}


async def _dedupe_paths(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize + dedupe candidate paths, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in payload.get("paths", [])[:200]:
        p = str(raw).replace("\\", "/").strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return {"paths": out, "duplicates_removed": len(payload.get("paths", [])) - len(out)}


def default_micro_registry() -> MicroWorkerRegistry:
    """Registry preloaded with the standard deterministic handlers."""
    reg = MicroWorkerRegistry()
    reg.register(MicroWorkerSpec(
        name="path.validate",
        handler=_path_validate,
        description="Validate paths exist inside the workspace (traversal-safe)",
        max_output_tokens=1_000,
    ))
    reg.register(MicroWorkerSpec(
        name="tokens.count",
        handler=_token_count,
        description="Estimate token counts for bounded text lists",
        max_output_tokens=500,
    ))
    reg.register(MicroWorkerSpec(
        name="paths.dedupe",
        handler=_dedupe_paths,
        description="Normalize and dedupe path lists, preserving order",
        max_output_tokens=500,
    ))
    return reg

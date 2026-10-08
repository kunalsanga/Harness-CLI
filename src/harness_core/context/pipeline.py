"""Context intelligence pipeline.

Wires the existing Harness pieces into one discover→rank→select flow:

    USER TASK
       ↓
    ContextPipeline.discover(ContextRequest)
       ├── candidate discovery (deterministic)
       │     • ContextReuseManager — cached/unchanged files (reuse, Part 6)
       │     • ContextEngine.discover_project — repository index
       │     • RelevanceRanker — path/name scoring
       │     • optional grep hits (passed in by the caller)
       │     • optional memory provider (injected)
       │     • recently modified files (dependency/relationship signal)
       ├── candidate ranking (deterministic)
       ├── AI relevance re-ranking (optional, fallback-safe, Part 3)
       └── budget fit + content loading (Parts 4/5)

The pipeline integrates ContextReuseManager rather than duplicating it:
candidates already snapshotted and unchanged are marked CACHED, changed
files INVALIDATED, and only fresh/invalidated files are read from disk.
Compaction marks entries SUMMARIZED via the reuse manager contract.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

from harness_core.context.models import (
    CandidateSource,
    ContextCandidate,
    ContextEvidence,
    ContextRequest,
    ContextSelection,
    ContextSnapshot,
    Freshness,
)
from harness_core.context.relevance import AIRelevanceRanker, AIRelevanceConfig
from harness_core.context.reuse import ContextReuseManager

if TYPE_CHECKING:
    from harness_core.intelligence import ProjectIntelligence

# Safety cap on how many file contents get loaded per discovery cycle —
# bounds both memory and the token cost of file content pieces.
_MAX_FILE_CONTENTS = 12
_MAX_FILE_BYTES = 200_000


class _MemoryProvider(Protocol):
    """Minimal memory surface usable by the pipeline (MemoryManager subset)."""

    async def retrieve(self, query: str, limit: int) -> list[Any]: ...


class ContextPipeline:
    """Bounded, deterministic-first context discovery for the agent loop."""

    def __init__(
        self,
        workspace_root: Path,
        reuse: ContextReuseManager | None = None,
        ai_ranker: AIRelevanceRanker | None = None,
        memory: _MemoryProvider | None = None,
        intelligence: "ProjectIntelligence | None" = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.reuse = reuse or ContextReuseManager()
        self.ai_ranker = ai_ranker
        self.memory = memory
        self.intelligence = intelligence

    # ── candidate discovery ───────────────────────────────────────────

    def _recent_files(self, limit: int = 6) -> list[str]:
        """Files most recently observed by the reuse manager.

        Uses snapshot insertion order (ContextReuseManager preserves
        insertion order), newest last.
        """
        snaps = getattr(self.reuse, "_snapshots", {})
        return [s.path for s in list(snaps.values())[-limit:]][::-1]

    async def _discover_candidates(
        self,
        request: ContextRequest,
        project_files: list[str],
        search_matches: dict[str, list[str]] | None,
    ) -> list[ContextCandidate]:
        """Gather candidates from all deterministic sources.

        Aggregation adds each source as evidence; candidates seen by
        multiple sources rise naturally through the evidence fold.
        """
        candidates: dict[str, ContextCandidate] = {}

        def _add(path: str, source: CandidateSource, score: float, detail: str) -> None:
            path = path.replace("\\", "/")
            if not path or path.startswith(".git/") or ".." in Path(path).parts:
                return
            cand = candidates.get(path)
            if cand is None:
                cand = ContextCandidate(path=path)
                candidates[path] = cand
            cand.add_evidence(ContextEvidence(source=source, score=score, detail=detail))

        # 1. Deterministic relevance over the repository index.
        ranker = self.ai_ranker._deterministic if self.ai_ranker else None
        if ranker is None:
            from harness_core.analysis.relevance import RelevanceRanker

            ranker = RelevanceRanker()
        # Repository index: docs (.md) stay in the candidate pool — the
        # ranker weights importance, so README/docs surface when relevant.
        for s in ranker.rank_files(
            project_files, request.task, search_matches=search_matches, max_results=request.max_candidates
        ):
            _add(s.path, CandidateSource.PATH_MATCH, s.total_score, "repo index rank")

        # 2. Grep evidence boosts files with content matches.
        for path in (search_matches or {}):
            _add(path, CandidateSource.GREP, 0.7, "content match")

        # 2b. ProjectIntelligence: symbol hits + dependency expansion.
        # Optional injected facade — when absent (or when anything fails)
        # discovery degrades to the deterministic sources above.
        intel = self.intelligence
        if intel is not None:
            try:
                root = Path(intel.root)
                if intel.file_list(limit=1):
                    # Symbol hits: task terms that match indexed symbol names.
                    seen_syms: set[str] = set()
                    for token in request.task.replace("(", " ").replace(")", " ").split():
                        token = token.strip(".,:;_-")
                        if len(token) < 4:
                            continue
                        for sym in intel.find_symbols(token, limit=5):
                            if sym.name in seen_syms:
                                continue
                            seen_syms.add(sym.name)
                            rel = str(Path(sym.file_path).relative_to(root))
                            _add(
                                rel,
                                CandidateSource.SYMBOL,
                                0.6,
                                f"symbol '{sym.name}' ({sym.kind})",
                            )
                    # Dependency expansion: one hop from grep/symbol seeds.
                    seeds = list((search_matches or {}).keys())
                    seeds += [
                        str(Path(s.file_path).relative_to(root)) for s in intel.find_symbols(
                            " ".join(request.task.split()[:4]), limit=3
                        )
                    ]
                    for seed in seeds[:10]:
                        try:
                            related = intel.dependencies.get_related_files(
                                str(root / seed), max_depth=1
                            )
                        except (ValueError, OSError):
                            continue
                        for rel_path in related[:8]:
                            _add(
                                str(Path(rel_path).relative_to(root)),
                                CandidateSource.DEPENDENCY,
                                0.5,
                                f"linked to {seed}",
                            )
            except Exception:
                pass  # intelligence is advisory — never blocks discovery

        # 3. Recently-touched files (reuse manager order).
        for i, path in enumerate(self._recent_files()):
            _add(path, CandidateSource.RECENT, max(0.2, 0.5 - i * 0.05), "recently read")

        # 4. Memory hits (optional, non-fatal).
        if self.memory is not None:
            try:
                hits = await self.memory.retrieve(request.task, limit=5)
                for hit in hits:
                    path = getattr(hit, "file_path", "") or (
                        hit.get("file_path", "") if isinstance(hit, dict) else ""
                    )
                    if path:
                        _add(path, CandidateSource.MEMORY, 0.5, "memory association")
            except Exception:
                pass  # memory is advisory — never blocks discovery

        return list(candidates.values())

    # ── validation & freshness ────────────────────────────────────────

    def _validate_candidate(
        self, cand: ContextCandidate
    ) -> tuple[bool, Freshness]:
        """Check the file exists inside the workspace and classify freshness.

        Returns (valid, freshness). Path traversal attempts are rejected
        here as defense in depth (candidates originate from our own index,
        but memory/other future sources may be less trusted).
        """
        rel = cand.path.replace("\\", "/")
        if rel.startswith("/") or ".." in Path(rel).parts:
            return False, Freshness.STALE
        try:
            p = (self.workspace_root / rel).resolve()
            p.relative_to(self.workspace_root.resolve())
        except (OSError, RuntimeError, ValueError):
            return False, Freshness.STALE
        if not p.is_file():
            return False, Freshness.STALE
        # Freshness vs the reuse manager snapshot.
        if self.reuse.has_snapshot(cand.path):
            try:
                stat = p.stat()
                unchanged = self.reuse.is_unchanged(
                    cand.path, size=stat.st_size, mtime_ns=int(stat.st_mtime * 1e9)
                )
                return True, Freshness.CACHED if unchanged else Freshness.INVALIDATED
            except OSError:
                return True, Freshness.INVALIDATED
        return True, Freshness.FRESH

    def _read_content(self, rel: str, max_tokens: int) -> tuple[str, int] | None:
        """Read a file bounded by max_tokens; records the reuse snapshot."""
        try:
            p = (self.workspace_root / rel).resolve()
            raw = p.read_text(encoding="utf-8", errors="replace")[:_MAX_FILE_BYTES]
            stat = p.stat()
            self.reuse.record_read(
                rel, content=raw, size=stat.st_size, mtime_ns=int(stat.st_mtime * 1e9)
            )
            max_chars = max(0, max_tokens * 4)
            if len(raw) > max_chars:
                raw = raw[:max_chars] + "\n... [truncated by context pipeline]"
            return raw, min(max_tokens, len(raw) // 4)
        except OSError:
            return None

    # ── main entry ────────────────────────────────────────────────────

    async def discover(
        self,
        request: ContextRequest,
        project_files: list[str] | None = None,
        search_matches: dict[str, list[str]] | None = None,
    ) -> ContextSelection:
        """Run the full pipeline for one task.

        Emits nothing itself — the caller (agent loop) owns event emission
        with run/task identity. Returns a ContextSelection carrying the
        snapshot for observability.
        """
        started = time.monotonic()
        project_files = project_files or []

        # 1. Deterministic candidate discovery.
        candidates = await self._discover_candidates(request, project_files, search_matches)
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[: request.max_candidates]
        candidates_considered = len(candidates)

        # 2. AI-assisted re-ranking (optional, fallback-safe).
        ai_used = False
        ai_failed = False
        if request.include_ai_ranking and self.ai_ranker is not None:
            result = await self.ai_ranker.rank(request.task, candidates, request)
            candidates = result.ranked
            ai_used = result.used
            ai_failed = result.failed

        # 3. Validate + classify freshness; drop invalid candidates.
        valid: list[ContextCandidate] = []
        for cand in candidates:
            ok, freshness = self._validate_candidate(cand)
            if not ok:
                cand.exists = False
                continue
            cand.freshness = freshness
            valid.append(cand)

        # 4. Budget fit: load content only for files that earn their tokens.
        selected: list[ContextCandidate] = []
        fresh_contents: dict[str, str] = {}
        tokens_used = 0
        for cand in valid:
            if len(selected) >= request.max_files or len(selected) >= _MAX_FILE_CONTENTS:
                break
            # Cached (unchanged) files skip the disk read — reuse contract.
            if cand.freshness is Freshness.CACHED:
                # No fresh content to inject; the model already has this file
                # in history. Keep it in the snapshot as a selection record.
                cand.tokens_estimate = 0
                selected.append(cand)
                continue
            read = self._read_content(cand.path, request.max_file_tokens)
            if read is None:
                continue
            content, tokens = read
            if tokens_used + tokens > max(0, request.max_file_tokens * request.max_files):
                continue  # file-content budget exhausted
            cand.tokens_estimate = tokens
            selected.append(cand)
            fresh_contents[cand.path] = content
            tokens_used += tokens

        snapshot = ContextSnapshot(
            task=request.task,
            candidates_considered=candidates_considered,
            selected=selected,
            tokens_used=tokens_used,
            tokens_budget=request.max_file_tokens * request.max_files,
            ai_ranking_used=ai_used,
            ai_ranking_failed=ai_failed,
        )
        snapshot.duration_ms = round((time.monotonic() - started) * 1000, 1)  # type: ignore[attr-defined]
        return ContextSelection(
            snapshot=snapshot,
            files=[c.path for c in selected],
            absolute_paths=[
                str((self.workspace_root / c.path).resolve()) for c in selected
            ],
            fresh_contents=fresh_contents,
        )

"""AI-assisted file relevance with deterministic fallback.

Architecture (per milestone Part 3):

    deterministic candidate generation (RelevanceRanker + index + grep)
        ↓ bounded candidates
    AI ranking stage (cheap/fast model, optional)
        ↓ validate → dedupe → intersect with candidates
    final ranking

The AI stage is *advisory only*: it can reorder and drop candidates, never
add unvalidated paths. Every guarantee is enforced:

- bounded candidate count
- bounded output
- validate returned paths
- reject nonexistent paths
- prevent path traversal
- deduplicate paths
- preserve ranking/evidence
- deterministic fallback if model ranking fails
- never make the entire agent unusable because the relevance model is unavailable

If no suitable model is available the AI stage is skipped entirely and the
deterministic ranking is returned unchanged (with the skip recorded).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from harness_core.analysis.relevance import RelevanceRanker
from harness_core.context.models import (
    ContextCandidate,
    ContextEvidence,
    ContextRequest,
    CandidateSource,
)

# The AI stage is only worth its latency on real decisions; below this many
# candidates the deterministic order is already good enough.
_MIN_CANDIDATES_FOR_AI = 4


class RankingModel(Protocol):
    """Minimal model surface needed for relevance ranking.

    Satisfied by ModelProvider (generate) and by ModelRouter.execute via a
    tiny adapter, so tests can pass a stub.
    """

    async def generate(self, request: Any) -> Any: ...


@dataclass
class AIRelevanceConfig:
    """Configurable limits for the AI relevance stage."""

    model_preference: str = ""          # empty = let provider default decide
    max_candidates: int = 40            # bounded prompt candidate list
    max_returned_paths: int = 12        # bounded output
    max_response_chars: int = 4000      # bounded response size
    timeout_seconds: float = 12.0
    temperature: float = 0.0


@dataclass
class AIRelevanceResult:
    """Outcome of the AI relevance stage."""

    used: bool = False                  # AI stage ran and produced an order
    failed: bool = False                # AI stage ran but failed (fallback applied)
    skipped: bool = False               # AI stage not attempted (disabled/too few candidates)
    reason: str = ""
    ranked: list[ContextCandidate] = field(default_factory=list)


def _norm_path(p: str) -> str:
    """Normalize to forward-slash, workspace-relative form."""
    return p.strip().replace("\\", "/").lstrip("./")


class AIRelevanceRanker:
    """Model-assisted re-ranking of deterministic context candidates.

    The deterministic ranker always produces the baseline ordering. When a
    model is supplied and the candidate list is large enough to matter, the
    model may reorder the candidates. Model output is untrusted input:

    1. parsed leniently (JSON list, or lines of paths)
    2. normalized and deduplicated
    3. rejected unless the path is inside the candidate set (prevents the
       model from injecting arbitrary files) and resolves inside the
       workspace (prevents traversal)
    4. truncated to ``max_returned_paths``

    Any failure — missing model, exception, timeout, unparsable output —
    degrades to the deterministic ordering. This class never raises.
    """

    def __init__(
        self,
        model: RankingModel | None = None,
        config: AIRelevanceConfig | None = None,
        workspace_root: Path | None = None,
        deterministic_ranker: RelevanceRanker | None = None,
    ) -> None:
        self._model = model
        self.config = config or AIRelevanceConfig()
        self._workspace_root = workspace_root
        self._deterministic = deterministic_ranker or RelevanceRanker()

    # ── deterministic baseline ────────────────────────────────────────

    def deterministic_candidates(
        self,
        task: str,
        candidate_paths: list[str],
        search_matches: dict[str, list[str]] | None = None,
        max_candidates: int | None = None,
    ) -> list[ContextCandidate]:
        """Rank raw paths deterministically into bounded ContextCandidates.

        This is the fallback path and the candidate source for the AI stage.
        It never touches a model and never touches the filesystem — callers
        validate existence separately so this stays pure.
        """
        limit = max_candidates or self.config.max_candidates
        scored = self._deterministic.rank_files(
            candidate_paths, task, search_matches=search_matches,
            max_results=limit,
        )
        out: list[ContextCandidate] = []
        for s in scored:
            cand = ContextCandidate(path=s.path, score=0.0)
            cand.add_evidence(
                ContextEvidence(
                    source=CandidateSource.PATH_MATCH,
                    score=s.total_score,
                    detail="deterministic relevance",
                )
            )
            out.append(cand)
        return out

    # ── AI-assisted stage ─────────────────────────────────────────────

    def _build_prompt(self, task: str, candidates: list[ContextCandidate]) -> str:
        """Build the bounded ranking prompt. Paths only — never file contents."""
        path_lines = "\n".join(f"{i + 1}. {c.path}" for i, c in enumerate(candidates))
        return (
            "You are a code-relevance judge. Given an engineering task and a list of "
            "candidate files, return the files most relevant to the task.\n\n"
            f"Task: {task}\n\n"
            "Candidate files:\n"
            f"{path_lines}\n\n"
            "Respond with ONLY a JSON array of the most relevant paths, most relevant "
            "first, e.g. [\"src/app.py\",\"src/utils.py\"]. "
            f"Return at most {self.config.max_returned_paths} paths. Use paths exactly as listed."
        )

    def _parse_response(self, text: str) -> list[str]:
        """Leniently extract paths from a model response.

        Accepts a JSON array of strings, a JSON object with a 'paths' or
        'files' key, or a plain line/list format. Anything else yields [].
        """
        text = (text or "").strip()
        if not text:
            return []
        # Direct JSON array (possibly fenced)
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    return [str(x) for x in data if isinstance(x, str)]
            except (json.JSONDecodeError, ValueError):
                pass
        # JSON object wrapper
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict):
                    for key in ("paths", "files", "ranking", "result"):
                        val = data.get(key)
                        if isinstance(val, list):
                            return [str(x) for x in val if isinstance(x, str)]
            except (json.JSONDecodeError, ValueError):
                pass
        # Line fallback: strip numbering/bullets
        paths: list[str] = []
        for line in text.splitlines():
            cleaned = re.sub(r"^[\s\d\.\)\-\*•]+", "", line).strip().strip("`'\"")
            if cleaned and "/" in cleaned or (cleaned and "." in Path(cleaned).name):
                paths.append(cleaned)
        return paths

    def _validate_path(self, raw: str, allowed: dict[str, ContextCandidate]) -> str | None:
        """Validate one model-returned path against the candidate set.

        Rules:
        - normalized into workspace-relative forward-slash form
        - must be one of the deterministic candidates (the model cannot
          introduce files the deterministic pass never saw)
        - must not escape the workspace (defense in depth against traversal
          even for candidate paths)
        """
        if not raw or len(raw) > 512:
            return None
        norm = _norm_path(raw)
        if not norm or ".." in Path(norm).parts:
            return None
        if self._workspace_root is not None:
            try:
                resolved = (self._workspace_root / norm).resolve()
                resolved.relative_to(self._workspace_root.resolve())
            except (OSError, RuntimeError, ValueError):
                return None
        if norm in allowed:
            return norm
        # Windows/POSIX casing drift tolerance: exact-normalized match first
        for cand_path in allowed:
            if cand_path.lower() == norm.lower():
                return cand_path
        return None

    async def rank(
        self,
        task: str,
        candidates: list[ContextCandidate],
        request: ContextRequest | None = None,
    ) -> AIRelevanceResult:
        """Optionally re-rank candidates with the model, else deterministic.

        Contract: never raises; on any problem returns the deterministic
        baseline with ``failed`` or ``skipped`` recorded so observability can
        count AI-relevance failures without breaking the agent.
        """
        baseline = list(candidates)
        if not self.config.model_preference and self._model is None:
            return AIRelevanceResult(
                skipped=True, reason="no relevance model configured", ranked=baseline
            )
        if len(baseline) < _MIN_CANDIDATES_FOR_AI:
            return AIRelevanceResult(
                skipped=True, reason="too few candidates to warrant AI ranking",
                ranked=baseline,
            )
        if self._model is None:
            return AIRelevanceResult(
                skipped=True, reason="no relevance model available", ranked=baseline
            )

        bounded = baseline[: self.config.max_candidates]
        prompt = self._build_prompt(task, bounded)
        try:
            from harness_core.providers.base import CompletionRequest

            req = CompletionRequest(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.model_preference or None,
                temperature=self.config.temperature,
                max_tokens=800,
            )
            response = await self._model.generate(req)
            text = (getattr(response, "content", "") or "")[: self.config.max_response_chars]
            returned = self._parse_response(text)
        except Exception as exc:  # noqa: BLE001 — any model failure must degrade
            return AIRelevanceResult(
                failed=True, reason=f"relevance model failed: {type(exc).__name__}",
                ranked=baseline,
            )

        if not returned:
            return AIRelevanceResult(
                failed=True, reason="relevance model returned no parsable paths",
                ranked=baseline,
            )

        allowed = {c.path: c for c in baseline}
        seen: set[str] = set()
        ranked: list[ContextCandidate] = []
        for raw in returned[: self.config.max_returned_paths * 2]:  # bounded scan
            if len(ranked) >= self.config.max_returned_paths:
                break
            valid = self._validate_path(raw, allowed)
            if valid is None or valid in seen:
                continue
            seen.add(valid)
            cand = allowed[valid]
            cand.add_evidence(
                ContextEvidence(
                    source=CandidateSource.AI_RANKING,
                    score=1.0 - (len(ranked) / max(1, self.config.max_returned_paths)) * 0.5,
                    detail=f"model rank position {len(ranked) + 1}",
                )
            )
            ranked.append(cand)

        if not ranked:
            return AIRelevanceResult(
                failed=True,
                reason="all model-returned paths rejected by validation",
                ranked=baseline,
            )

        # Candidates the model dropped keep their deterministic rank, after
        # the model-endorsed ones — reordering only, never data loss.
        rest = [c for c in baseline if c.path not in seen]
        return AIRelevanceResult(used=True, ranked=ranked + rest)

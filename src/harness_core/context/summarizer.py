"""Model-assisted summarization for context compaction (milestone Part 4).

The existing ``ContextCompactor`` is deterministic and stays authoritative.
This module adds an *optional* cheap-model summarization stage in front of
it: when a model is available, older tool results are condensed into a
compact summary; when it is not (or it fails), compaction falls back to
the deterministic truncation the ContextCompactor already performs.

Contract:
- bounded input (older messages capped) and bounded output
- never destroys the current task intent — the latest user request and
  current objective are supplied as protected context and echoed into the
  summary header deterministically
- any failure degrades to deterministic behavior; never raises
"""

from __future__ import annotations

from typing import Any, Protocol

from harness_core.context.compaction import AgentMessage

# Bound the text we send for summarization — summarization of a bounded
# input is bounded work; a full unbounded dump is neither cheap nor safe.
_MAX_JOINED_CHARS = 12_000
_MAX_OUTPUT_TOKENS = 400


class SummarizerModel(Protocol):
    async def generate(self, request: Any) -> Any: ...


class ModelSummarizer:
    """Optional cheap-model summarizer with deterministic fallback."""

    def __init__(self, model: SummarizerModel | None = None, model_preference: str = "") -> None:
        self._model = model
        self._model_preference = model_preference

    @property
    def available(self) -> bool:
        return self._model is not None

    @staticmethod
    def _render_messages(messages: list[AgentMessage]) -> str:
        """Bounded plain-text rendering of older messages."""
        parts: list[str] = []
        for m in messages:
            kind = m.kind or "normal"
            body = (m.content or "").strip()
            if not body:
                continue
            parts.append(f"[{kind}] {body[:600]}")
        text = "\n".join(parts)
        if len(text) > _MAX_JOINED_CHARS:
            text = text[:_MAX_JOINED_CHARS] + "\n... [bounded by summarizer]"
        return text

    def _prompt(self, joined: str, protected: str) -> str:
        return (
            "Summarize the following earlier agent work into a compact state note "
            "for continuing an engineering task. Keep: what was attempted, which "
            "files were touched, what failed and why, what remains. Drop redundant "
            "tool output. Maximum 150 words. Plain text only.\n\n"
            f"Current objective (must not be lost): {protected}\n\n"
            f"Earlier work:\n{joined}"
        )

    async def summarize(
        self,
        messages: list[AgentMessage],
        protected_intent: str = "",
    ) -> str | None:
        """Summarize older messages, or None when summarization is not possible.

        Never raises. Deterministic truncation fallback is the caller's job
        (ContextCompactor already implements it); returning None simply
        means "use the deterministic path".
        """
        if self._model is None:
            return None
        joined = self._render_messages(messages)
        if not joined.strip():
            return None
        try:
            from harness_core.providers.base import CompletionRequest

            req = CompletionRequest(
                messages=[{"role": "user", "content": self._prompt(joined, protected_intent)}],
                model=self._model_preference or None,
                temperature=0.0,
                max_tokens=_MAX_OUTPUT_TOKENS,
            )
            resp = await self._model.generate(req)
            text = (getattr(resp, "content", "") or "").strip()
            if not text:
                return None
            # Bounded, single-paragraph summary; strip markdown fences if the
            # model added them.
            text = text.strip("`").strip()
            if text.startswith("summary"):
                text = text[7:].lstrip(" :\n")
            return text[:2000]
        except Exception:
            # Model unavailable / failed — deterministic fallback applies.
            return None

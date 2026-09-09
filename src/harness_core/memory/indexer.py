"""TF-IDF + token-hash vectoriser for memory embedding.

Provides a self-contained embedding implementation — no external model calls
or library dependencies required.  Uses a two-stage approach:

1. Tokenise + lowercase + strip punctuation.
2. Hash each token into a fixed-dimension vector using term-frequency hashing.
3. Compute TF-IDF-like weights with smoothed IDF
   (``term_freq * (log((1 + N) / (1 + doc_freq)) + 1)``) so vectors are
   never all-zero, even for a single-document corpus.
4. L2-normalise to unit length so cosine similarity = dot product.

This is sufficient for "did I build JWT auth before?" retrieval without
requiring sentence-transformers, OpenAI ada, or any external service.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness_core.memory.domain import MemoryEntry


# Default embedding dimension for the hash-vector approach.
# 256 is small enough for fast in-memory operations yet large enough
# to give distinct buckets for typical project vocabulary.
_EMBEDDING_DIM = 256


class MemoryIndexer:
    """Builds and queries TF-IDF embeddings for memory entries."""

    def __init__(self, dimension: int = _EMBEDDING_DIM) -> None:
        self.dimension = dimension
        # Global document frequency map: token -> number of entries indexed
        self._doc_freq: Counter[str] = Counter()
        # Number of documents indexed
        self._n_docs: int = 0
        # In-memory cache of entry embeddings (id -> vector)
        self._entry_embeddings: dict[str, list[float]] = {}

    # ── Tokenisation ──────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Return lower-case alphanumeric tokens from text."""
        # Normalise unicode (e.g. smart quotes → straight quotes)
        text = unicodedata.normalize("NFKD", text)
        # Strip markdown links and code fences
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
        # Keep only alphanumeric + hyphens + underscores
        tokens = re.findall(r"[a-z0-9][a-z0-9\-_]*[a-z0-9]|[a-z0-9]", text.lower())
        return tokens

    # ── TF-IDF vectorisation ──────────────────────────────────────────────

    def _term_freq(self, tokens: list[str]) -> dict[str, float]:
        """Raw term frequency as a fraction of total tokens (1.0 max)."""
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = len(tokens)
        return {tok: cnt / total for tok, cnt in counts.items()}

    def _hash_vector(self, tf: dict[str, float]) -> list[float]:
        """Hash each term into a fixed-dim vector, weighted by TF-IDF.

        Uses *smoothed* inverse document frequency
        (``idf = log((1 + n) / (1 + df)) + 1``, the scikit-learn convention)
        so that terms present in every document — or a corpus of a single
        document, or a freshly restarted indexer with no statistics — still
        produce a non-zero, unit-normalisable vector.  The raw
        ``log(n / df)`` formulation yields 0 for ``df == n`` which made every
        embedding all-zero and made single-entry / post-restart stores
        unsearchable.
        """
        vec = [0.0] * self.dimension
        n = self._n_docs if self._n_docs > 0 else 1  # avoid div/0 on empty corpus
        for token, freq in tf.items():
            idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimension
            df = self._doc_freq.get(token, 0)
            idf = math.log((1.0 + n) / (1.0 + df)) + 1.0
            vec[idx] += freq * idf
        return vec

    def _normalise(self, vec: list[float]) -> list[float]:
        """L2-normalise a vector to unit length."""
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    # ── Public API ────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for arbitrary text."""
        tokens = self._tokenise(text)
        tf = self._term_freq(tokens)
        vec = self._hash_vector(tf)
        return self._normalise(vec)

    def index(self, entry: MemoryEntry) -> list[float]:
        """Index an entry and cache its embedding for later search."""
        tokens = self._tokenise(entry.content)
        # Update document frequency
        for tok in set(tokens):
            self._doc_freq[tok] += 1
        self._n_docs += 1
        # Build and cache the embedding
        tf = self._term_freq(tokens)
        vec = self._hash_vector(tf)
        normed = self._normalise(vec)
        self._entry_embeddings[entry.id] = normed
        return normed

    def get_embedding(self, entry_id: str) -> list[float] | None:
        """Return a cached embedding by entry ID."""
        return self._entry_embeddings.get(entry_id)

    # ── Similarity ────────────────────────────────────────────────────────

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        """Return cosine similarity between two normalised vectors."""
        return sum(av * bv for av, bv in zip(a, b))

    # ── Persistence helpers ────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "dimension": self.dimension,
            "doc_freq": dict(self._doc_freq),
            "n_docs": self._n_docs,
            "entry_embeddings": self._entry_embeddings,
        }

    def load_state(self, state: dict) -> None:
        self.dimension = state.get("dimension", _EMBEDDING_DIM)
        self._doc_freq = Counter(state.get("doc_freq", {}))
        self._n_docs = state.get("n_docs", 0)
        self._entry_embeddings = state.get("entry_embeddings", {})

"""
Vector stores and retrieval indexes for Chapter 13: RAG.

Provides three pure-NumPy / scikit-learn / rank-bm25 indexes that work
without FAISS or Chroma — making them ideal for CI, learning, and small to
medium corpora:

    InMemoryVectorStore : dense cosine-similarity store with save/load
    BM25Index           : sparse BM25 retriever wrapping rank_bm25
    HybridIndex         : combines dense + sparse via reciprocal rank fusion

For real production at scale, swap `InMemoryVectorStore` for FAISS / Chroma /
PGVector — the public API (`add`, `search`) is intentionally compatible.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single hit returned by a retriever."""

    chunk_id: str
    score: float
    text: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# Dense in-memory store
# ---------------------------------------------------------------------------

class InMemoryVectorStore:
    """
    A minimal NumPy-backed dense vector index.

    Stores L2-normalized embeddings so cosine similarity reduces to a single
    matrix multiply. Suitable for thousands of chunks; for millions, swap
    in FAISS — the API is the same.
    """

    def __init__(self, dim: int, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize
        self.embeddings: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self.chunk_ids: List[str] = []
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []

    # ---- mutation -------------------------------------------------------

    def add(
        self,
        embeddings: np.ndarray,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict]] = None,
    ) -> None:
        """Add a batch of embeddings and their associated metadata."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"embeddings must be (n, {self.dim}); got {embeddings.shape}"
            )
        if not (len(embeddings) == len(chunk_ids) == len(texts)):
            raise ValueError("embeddings, chunk_ids, texts must have same length")
        if self.normalize:
            embeddings = _l2_normalize(embeddings)
        self.embeddings = np.vstack([self.embeddings, embeddings])
        self.chunk_ids.extend(chunk_ids)
        self.texts.extend(texts)
        if metadatas is None:
            metadatas = [{} for _ in chunk_ids]
        self.metadatas.extend(metadatas)

    # ---- query ----------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Return the top-k most similar chunks by cosine similarity."""
        if len(self.embeddings) == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query dim {q.shape[0]} != index dim {self.dim}")
        if self.normalize:
            q = _l2_normalize(q.reshape(1, -1))[0]
        scores = self.embeddings @ q
        top_k = min(top_k, len(scores))
        # argpartition is O(n); then sort just the top-k.
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [
            SearchResult(
                chunk_id=self.chunk_ids[i],
                score=float(scores[i]),
                text=self.texts[i],
                metadata=self.metadatas[i],
            )
            for i in idx
        ]

    # ---- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the store as a single pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "dim": self.dim,
                    "normalize": self.normalize,
                    "embeddings": self.embeddings,
                    "chunk_ids": self.chunk_ids,
                    "texts": self.texts,
                    "metadatas": self.metadatas,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryVectorStore":
        """Load a previously saved store."""
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        store = cls(dim=state["dim"], normalize=state["normalize"])
        store.embeddings = state["embeddings"]
        store.chunk_ids = state["chunk_ids"]
        store.texts = state["texts"]
        store.metadatas = state["metadatas"]
        return store

    def __len__(self) -> int:
        return len(self.chunk_ids)


# ---------------------------------------------------------------------------
# Sparse BM25 index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Wraps rank_bm25's BM25Okapi with the same `add` / `search` surface as
    `InMemoryVectorStore`. Tokenizes by lowercase whitespace + punctuation
    stripping — fine for English educational examples.
    """

    _TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

    def __init__(self):
        self.chunk_ids: List[str] = []
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []
        self._tokenized: List[List[str]] = []
        self._bm25 = None  # built lazily / on add

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        return [t.lower() for t in cls._TOKEN_RE.findall(text or "")]

    def add(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict]] = None,
    ) -> None:
        if not (len(chunk_ids) == len(texts)):
            raise ValueError("chunk_ids and texts must have same length")
        if metadatas is None:
            metadatas = [{} for _ in chunk_ids]
        self.chunk_ids.extend(chunk_ids)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self._tokenized.extend(self.tokenize(t) for t in texts)
        self._bm25 = None  # invalidate

    def _ensure_built(self) -> None:
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank-bm25 is required for BM25Index. "
                "Install with: pip install rank-bm25"
            ) from e
        if not self._tokenized:
            return
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if not self._tokenized:
            return []
        self._ensure_built()
        scores = self._bm25.get_scores(self.tokenize(query))
        top_k = min(top_k, len(scores))
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [
            SearchResult(
                chunk_id=self.chunk_ids[i],
                score=float(scores[i]),
                text=self.texts[i],
                metadata=self.metadatas[i],
            )
            for i in idx
        ]

    def __len__(self) -> int:
        return len(self.chunk_ids)


# ---------------------------------------------------------------------------
# Hybrid (dense + sparse) index with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class HybridIndex:
    """
    Combines a dense `InMemoryVectorStore` with a sparse `BM25Index` and
    fuses their rankings via Reciprocal Rank Fusion (RRF):

        score(d) = sum over rankers r:  1 / (k + rank_r(d))

    RRF needs no score calibration between rankers and works very well
    in practice for hybrid retrieval.
    """

    def __init__(
        self,
        dense: InMemoryVectorStore,
        sparse: BM25Index,
        rrf_k: int = 60,
    ):
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> List[SearchResult]:
        """Pull `candidate_k` from each retriever, fuse, return top_k."""
        dense_hits = self.dense.search(query_embedding, top_k=candidate_k)
        sparse_hits = self.sparse.search(query_text, top_k=candidate_k)
        return reciprocal_rank_fusion(
            [dense_hits, sparse_hits], k=self.rrf_k
        )[:top_k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def reciprocal_rank_fusion(
    rankings: List[List[SearchResult]],
    k: int = 60,
) -> List[SearchResult]:
    """
    Combine multiple ranked lists into a single list using RRF.
    The first occurrence of each chunk_id wins for the returned text/metadata.
    """
    fused: Dict[str, float] = {}
    keep: Dict[str, SearchResult] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            keep.setdefault(hit.chunk_id, hit)
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    out: List[SearchResult] = []
    for chunk_id, score in ordered:
        h = keep[chunk_id]
        out.append(
            SearchResult(
                chunk_id=h.chunk_id,
                score=float(score),
                text=h.text,
                metadata=h.metadata,
            )
        )
    return out


def save_jsonl(records: List[Dict], path: str | Path) -> None:
    """Convenience writer for evaluation runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

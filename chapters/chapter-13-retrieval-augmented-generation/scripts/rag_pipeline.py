"""
End-to-end RAG pipeline for Chapter 13.

`RAGPipeline` wires together:
    - a chunker             (any callable text -> List[Chunk])
    - an embedder           (any callable List[str] -> ndarray)
    - a vector store        (InMemoryVectorStore or compatible)
    - an optional reranker  (callable (query, hits) -> hits)
    - an LLM client         (defaults to MockLLM — no API key needed)

It also ships an `evaluate(...)` method that computes hit@k and Mean
Reciprocal Rank against gold (query -> relevant chunk_ids) labels so the
notebooks can score retrieval quality offline.

The whole thing runs without `openai`, `anthropic`, `faiss`, or
`sentence-transformers`. Real clients can be plugged in via duck typing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chunking import Chunk, Chunker
from vectorstore import InMemoryVectorStore, SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TF-IDF embedder — default, no extra deps
# ---------------------------------------------------------------------------

class TfidfEmbedder:
    """
    TF-IDF + truncated-SVD as a tiny dense embedder. Quality is far below
    sentence-transformers but it has zero extra dependencies and is fully
    deterministic, which is ideal for CI and teaching.

    Call `fit(corpus)` once on your full corpus, then `encode(...)` is a
    dense projection of the TF-IDF vector. `encode_query(...)` uses the
    same fitted transformers.
    """

    def __init__(self, dim: int = 128, max_features: int = 4096, random_state: int = 42):
        self.dim = dim
        self.max_features = max_features
        self.random_state = random_state
        self._tfidf = None
        self._svd = None

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(max_features=self.max_features)
        X = self._tfidf.fit_transform(texts)
        n_components = max(2, min(self.dim, X.shape[1] - 1, X.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self._svd.fit(X)
        # Make sure self.dim matches what SVD actually produced.
        self.dim = n_components
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._tfidf is None or self._svd is None:
            raise RuntimeError("TfidfEmbedder.fit must be called first.")
        X = self._tfidf.transform(texts)
        return self._svd.transform(X).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockLLM:
    """
    A deterministic 'LLM' that templates retrieved chunks into an answer.

    It does NOT generate fluent prose — it produces a structured, citation-
    rich response so notebooks can run end-to-end without any API access.
    Replace with an OpenAI / Anthropic client to get real generation; the
    pipeline only requires a `.complete(prompt: str) -> str` method.
    """

    def __init__(self, max_chars: int = 280):
        self.max_chars = max_chars

    def complete(self, prompt: str) -> str:
        # Extract the user question and supporting context from the prompt
        # using simple string conventions used by `_build_prompt`.
        question = ""
        context = ""
        if "Question:" in prompt:
            question = prompt.split("Question:", 1)[1].split("\n", 1)[0].strip()
        if "Context:" in prompt:
            context = prompt.split("Context:", 1)[1].split("Question:", 1)[0].strip()

        # Pull the first sentence-ish span from each cited chunk and stitch.
        snippets: List[str] = []
        for line in context.splitlines():
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            # Format: "[chunk_id] text..."
            close = line.find("]")
            if close == -1:
                continue
            chunk_id = line[1:close]
            body = line[close + 1 :].strip()
            first = body.split(". ", 1)[0]
            snippets.append(f"{first.strip().rstrip('.')} [{chunk_id}].")
            if sum(len(s) for s in snippets) > self.max_chars:
                break

        if not snippets:
            return f"I do not have enough context to answer: {question}".strip()
        head = f"Q: {question}\nA: " if question else "A: "
        return head + " ".join(snippets)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """The output of `RAGPipeline.answer`."""

    answer: str
    contexts: List[SearchResult]
    prompt: str
    latency_seconds: float = 0.0
    metadata: Dict = field(default_factory=dict)


class RAGPipeline:
    """
    Orchestrates: chunk -> embed -> index -> retrieve -> (rerank) -> generate.

    Example
    -------
    >>> pipe = RAGPipeline()
    >>> pipe.index_documents({"d1": "RAG retrieves relevant text. Then LLMs answer."})
    >>> r = pipe.answer("What does RAG do?")
    >>> print(r.answer)
    """

    def __init__(
        self,
        chunker: Optional[Chunker] = None,
        embedder=None,
        vector_store: Optional[InMemoryVectorStore] = None,
        reranker: Optional[Callable[[str, List[SearchResult]], List[SearchResult]]] = None,
        llm_client=None,
        top_k: int = 5,
        candidate_k: int = 20,
    ):
        self.chunker = chunker or Chunker(strategy="sentence", max_tokens=80)
        self.embedder = embedder or TfidfEmbedder(dim=128)
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm_client or MockLLM()
        self.top_k = top_k
        self.candidate_k = candidate_k
        self._fitted = False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, documents: Dict[str, str]) -> int:
        """Chunk, embed, and index a {doc_id: text} mapping. Returns chunk count."""
        chunks: List[Chunk] = self.chunker.chunk_documents(documents)
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        # Fit the embedder on the chunk corpus the first time.
        if hasattr(self.embedder, "fit") and not self._fitted:
            self.embedder.fit(texts)
            self._fitted = True
        embeddings = self.embedder.encode(texts)
        if self.vector_store is None:
            self.vector_store = InMemoryVectorStore(dim=embeddings.shape[1])
        self.vector_store.add(
            embeddings=embeddings,
            chunk_ids=[c.chunk_id for c in chunks],
            texts=texts,
            metadatas=[{"doc_id": c.doc_id, **c.metadata} for c in chunks],
        )
        return len(chunks)

    # ------------------------------------------------------------------
    # Query path
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        if self.vector_store is None or len(self.vector_store) == 0:
            return []
        k = top_k or self.top_k
        cand_k = max(self.candidate_k, k)
        if hasattr(self.embedder, "encode_query"):
            q_emb = self.embedder.encode_query(query)
        else:
            q_emb = self.embedder.encode([query])[0]
        hits = self.vector_store.search(q_emb, top_k=cand_k)
        if self.reranker is not None:
            hits = self.reranker(query, hits)
        return hits[:k]

    def answer(self, query: str, top_k: Optional[int] = None) -> RAGResponse:
        t0 = time.time()
        contexts = self.retrieve(query, top_k=top_k)
        prompt = self._build_prompt(query, contexts)
        text = self.llm.complete(prompt)
        return RAGResponse(
            answer=text,
            contexts=contexts,
            prompt=prompt,
            latency_seconds=time.time() - t0,
            metadata={"n_contexts": len(contexts)},
        )

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(query: str, contexts: List[SearchResult]) -> str:
        """
        Standard grounded-answer prompt. Each context is prefixed with its
        chunk_id in brackets so the model can cite sources.
        """
        ctx_lines = [f"[{c.chunk_id}] {c.text}" for c in contexts]
        ctx_block = "\n".join(ctx_lines) if ctx_lines else "(no context found)"
        return (
            "You are a helpful assistant. Answer the question using ONLY the\n"
            "context below. Cite sources using their bracketed chunk_id. If\n"
            "the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{ctx_block}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        queries: Sequence[str],
        relevant_chunk_ids: Sequence[Sequence[str]],
        top_ks: Sequence[int] = (1, 3, 5, 10),
    ) -> Dict[str, float]:
        """
        Compute hit@k and Mean Reciprocal Rank for a list of queries.

        Args:
            queries:               list of query strings
            relevant_chunk_ids:    parallel list — for each query, the set
                                   of chunk_ids that count as relevant
            top_ks:                cutoffs to report hit@k at

        Returns: {"hit@1": .., "hit@3": .., ..., "mrr": .., "latency_p50": ..}
        """
        if len(queries) != len(relevant_chunk_ids):
            raise ValueError("queries and relevant_chunk_ids must align")
        if not queries:
            return {}

        max_k = max(top_ks)
        hits = {k: 0 for k in top_ks}
        rr_sum = 0.0
        latencies: List[float] = []

        for q, gold in zip(queries, relevant_chunk_ids):
            t0 = time.time()
            results = self.retrieve(q, top_k=max_k)
            latencies.append(time.time() - t0)
            gold_set = set(gold)
            ranked_ids = [r.chunk_id for r in results]
            # Hit@k
            for k in top_ks:
                if any(cid in gold_set for cid in ranked_ids[:k]):
                    hits[k] += 1
            # Reciprocal rank
            rr = 0.0
            for rank, cid in enumerate(ranked_ids, start=1):
                if cid in gold_set:
                    rr = 1.0 / rank
                    break
            rr_sum += rr

        n = len(queries)
        out: Dict[str, float] = {f"hit@{k}": hits[k] / n for k in top_ks}
        out["mrr"] = rr_sum / n
        if latencies:
            lat = sorted(latencies)
            out["latency_p50"] = lat[len(lat) // 2]
            out["latency_p95"] = lat[max(0, int(len(lat) * 0.95) - 1)]
        return out

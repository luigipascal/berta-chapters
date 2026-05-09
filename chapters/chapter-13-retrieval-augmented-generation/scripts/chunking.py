"""
Document chunking strategies for Chapter 13: Retrieval-Augmented Generation.

Provides four chunking approaches and a unified `Chunker` facade:

    - fixed_size_chunks       : split by character count, hard boundaries
    - sliding_window_chunks   : overlapping windows of tokens
    - sentence_chunks         : group sentences up to a target token budget
    - semantic_chunks         : merge adjacent sentences with high TF-IDF cosine
                                similarity into the same chunk

All functions return `List[Chunk]` so downstream embedding and indexing code
sees a uniform interface.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single retrievable unit of text plus provenance metadata."""

    text: str
    chunk_id: str
    doc_id: str = ""
    start: int = 0
    end: int = 0
    metadata: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)

    def token_count(self) -> int:
        # Whitespace-based token approximation (good enough without tiktoken).
        return len(self.text.split())


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _whitespace_tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter — no NLTK dependency required."""
    text = text.strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------

def fixed_size_chunks(
    text: str,
    chunk_size: int = 500,
    doc_id: str = "doc",
) -> List[Chunk]:
    """
    Split `text` into non-overlapping character windows of `chunk_size`.

    Simple, deterministic, and fast — but breaks across word and sentence
    boundaries. Use as a baseline.
    """
    if not text:
        return []
    chunks: List[Chunk] = []
    for i, start in enumerate(range(0, len(text), chunk_size)):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if not piece:
            continue
        chunks.append(
            Chunk(
                text=piece,
                chunk_id=f"{doc_id}::fixed::{i}",
                doc_id=doc_id,
                start=start,
                end=end,
                metadata={"strategy": "fixed", "chunk_size": chunk_size},
            )
        )
    return chunks


def sliding_window_chunks(
    text: str,
    window_tokens: int = 80,
    overlap_tokens: int = 16,
    doc_id: str = "doc",
) -> List[Chunk]:
    """
    Split into overlapping word windows. Overlap lets a chunk straddle
    information that crosses boundaries.
    """
    if not text:
        return []
    if overlap_tokens >= window_tokens:
        raise ValueError("overlap_tokens must be smaller than window_tokens")
    tokens = _whitespace_tokenize(text)
    if not tokens:
        return []
    step = max(1, window_tokens - overlap_tokens)
    chunks: List[Chunk] = []
    for i, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start : start + window_tokens]
        if not window:
            continue
        piece = " ".join(window)
        chunks.append(
            Chunk(
                text=piece,
                chunk_id=f"{doc_id}::slide::{i}",
                doc_id=doc_id,
                start=start,
                end=start + len(window),
                metadata={
                    "strategy": "sliding",
                    "window_tokens": window_tokens,
                    "overlap_tokens": overlap_tokens,
                },
            )
        )
        if start + window_tokens >= len(tokens):
            break
    return chunks


def sentence_chunks(
    text: str,
    max_tokens: int = 120,
    doc_id: str = "doc",
) -> List[Chunk]:
    """
    Greedy sentence packing: append sentences to the current chunk until
    adding the next one would exceed `max_tokens`. Then start a new chunk.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_tokens = 0
    idx = 0
    for sent in sentences:
        n = len(sent.split())
        if buf and buf_tokens + n > max_tokens:
            joined = " ".join(buf).strip()
            chunks.append(
                Chunk(
                    text=joined,
                    chunk_id=f"{doc_id}::sent::{idx}",
                    doc_id=doc_id,
                    metadata={"strategy": "sentence", "max_tokens": max_tokens},
                )
            )
            idx += 1
            buf = [sent]
            buf_tokens = n
        else:
            buf.append(sent)
            buf_tokens += n
    if buf:
        joined = " ".join(buf).strip()
        chunks.append(
            Chunk(
                text=joined,
                chunk_id=f"{doc_id}::sent::{idx}",
                doc_id=doc_id,
                metadata={"strategy": "sentence", "max_tokens": max_tokens},
            )
        )
    return chunks


def semantic_chunks(
    text: str,
    similarity_threshold: float = 0.3,
    max_tokens: int = 200,
    doc_id: str = "doc",
) -> List[Chunk]:
    """
    Semantic chunking: split into sentences, then start a new chunk whenever
    the cosine similarity between consecutive sentence TF-IDF vectors drops
    below `similarity_threshold` (or the max-token budget is reached).

    Uses scikit-learn's TfidfVectorizer for a no-extra-deps similarity proxy.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentence_chunks(text, max_tokens=max_tokens, doc_id=doc_id)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        logger.warning("scikit-learn not installed; falling back to sentence_chunks")
        return sentence_chunks(text, max_tokens=max_tokens, doc_id=doc_id)

    vec = TfidfVectorizer().fit(sentences)
    mat = vec.transform(sentences)

    chunks: List[Chunk] = []
    buf: List[str] = [sentences[0]]
    buf_tokens = len(sentences[0].split())
    idx = 0
    for i in range(1, len(sentences)):
        sim = float(cosine_similarity(mat[i - 1], mat[i])[0, 0])
        n = len(sentences[i].split())
        too_big = buf_tokens + n > max_tokens
        if sim < similarity_threshold or too_big:
            chunks.append(
                Chunk(
                    text=" ".join(buf).strip(),
                    chunk_id=f"{doc_id}::sem::{idx}",
                    doc_id=doc_id,
                    metadata={
                        "strategy": "semantic",
                        "similarity_threshold": similarity_threshold,
                    },
                )
            )
            idx += 1
            buf = [sentences[i]]
            buf_tokens = n
        else:
            buf.append(sentences[i])
            buf_tokens += n
    if buf:
        chunks.append(
            Chunk(
                text=" ".join(buf).strip(),
                chunk_id=f"{doc_id}::sem::{idx}",
                doc_id=doc_id,
                metadata={"strategy": "semantic"},
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Unified facade
# ---------------------------------------------------------------------------

class Chunker:
    """
    Unified chunker that delegates to one of the strategies above.

    >>> ch = Chunker(strategy="sentence", max_tokens=80)
    >>> chunks = ch.chunk("Some text here. Another sentence.", doc_id="d1")
    """

    STRATEGIES: Dict[str, Callable] = {
        "fixed": fixed_size_chunks,
        "sliding": sliding_window_chunks,
        "sentence": sentence_chunks,
        "semantic": semantic_chunks,
    }

    def __init__(self, strategy: str = "sentence", **kwargs):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from {list(self.STRATEGIES)}"
            )
        self.strategy = strategy
        self.kwargs = kwargs

    def chunk(self, text: str, doc_id: str = "doc") -> List[Chunk]:
        fn = self.STRATEGIES[self.strategy]
        return fn(text, doc_id=doc_id, **self.kwargs)

    def chunk_documents(
        self, documents: Dict[str, str]
    ) -> List[Chunk]:
        """Chunk a mapping of {doc_id: text} into a flat list of Chunk objects."""
        out: List[Chunk] = []
        for doc_id, text in documents.items():
            out.extend(self.chunk(text, doc_id=doc_id))
        return out

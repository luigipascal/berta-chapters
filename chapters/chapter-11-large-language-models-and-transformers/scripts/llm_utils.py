"""
LLM helpers for Chapter 11 — tokenization and embedding utilities with a
graceful fallback when ``transformers`` / ``torch`` are not installed.

The fallback path lets the chapter run end-to-end on a vanilla numpy/sklearn
environment (e.g. in CI) while still demonstrating the right shapes and APIs.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Tokenizer wrapper
# ============================================================================


class LLMTokenizerWrapper:
    """
    Wrap a Hugging Face ``AutoTokenizer``; fall back to a deterministic
    whitespace + hashing tokenizer if ``transformers`` is unavailable.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 vocab_size: int = 30522) -> None:
        self.model_name = model_name
        self.vocab_size = vocab_size
        self._tokenizer = None
        self._fallback = False
        try:
            from transformers import AutoTokenizer  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Loaded HF tokenizer for %s", model_name)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "transformers not available (%s); using whitespace fallback. "
                "Run `pip install transformers` for the real tokenizer.", e,
            )
            self._fallback = True

    # -- fallback impl --------------------------------------------------------

    def _fallback_encode(self, text: str, max_length: Optional[int]) -> List[int]:
        toks = re.findall(r"\w+|[^\w\s]", text.lower())
        ids = [(hash(t) % (self.vocab_size - 2)) + 2 for t in toks]  # 0=PAD, 1=UNK
        if max_length is not None:
            ids = ids[:max_length]
            ids += [0] * (max_length - len(ids))
        return ids

    # -- public API -----------------------------------------------------------

    def encode(self, text: str, max_length: Optional[int] = None,
               padding: bool = False) -> List[int]:
        """Encode a single string to a list of token ids."""
        if self._fallback:
            return self._fallback_encode(text, max_length if padding else None)
        kwargs = {"truncation": True}
        if max_length is not None:
            kwargs["max_length"] = max_length
        if padding and max_length is not None:
            kwargs["padding"] = "max_length"
        return self._tokenizer.encode(text, **kwargs)

    def encode_batch(self, texts: Sequence[str], max_length: int = 64) -> np.ndarray:
        """Encode a batch to a (batch, max_length) int array (padded)."""
        rows = [self.encode(t, max_length=max_length, padding=True) for t in texts]
        return np.asarray(rows, dtype=np.int64)

    def tokenize(self, text: str) -> List[str]:
        """Return the surface tokens (strings) for inspection."""
        if self._fallback:
            return re.findall(r"\w+|[^\w\s]", text.lower())
        return self._tokenizer.tokenize(text)

    @property
    def is_fallback(self) -> bool:
        return self._fallback


# ============================================================================
# Embedding extractor
# ============================================================================


def mean_pool(token_embeds: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Mean-pool token embeddings into a single sentence vector.

    token_embeds: (batch, seq, dim)
    mask:         (batch, seq) of 0/1 (1 = real token); optional
    Returns:      (batch, dim)
    """
    if mask is None:
        return token_embeds.mean(axis=1)
    m = mask.astype(token_embeds.dtype)[:, :, None]
    summed = (token_embeds * m).sum(axis=1)
    denom = np.clip(m.sum(axis=1), 1e-9, None)
    return summed / denom


class EmbeddingExtractor:
    """
    Compute sentence-level embeddings.

    If ``transformers`` + ``torch`` are installed, runs ``AutoModel`` and
    mean-pools the last hidden state. Otherwise falls back to a deterministic
    hashing-trick embedding so notebooks still produce a meaningful vector.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 dim: int = 384, max_length: int = 64) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.dim = dim
        self._model = None
        self._tokenizer = LLMTokenizerWrapper(model_name)
        self._fallback = self._tokenizer.is_fallback
        if not self._fallback:
            try:
                import torch  # noqa: F401
                from transformers import AutoModel  # type: ignore
                self._model = AutoModel.from_pretrained(model_name)
                self._model.eval()
                self.dim = int(self._model.config.hidden_size)
            except Exception as e:  # noqa: BLE001
                logger.warning("Falling back to hashing embedder (%s).", e)
                self._fallback = True
                self._model = None

    # -- fallback hashing embedding ------------------------------------------

    def _fallback_embed(self, texts: Sequence[str]) -> np.ndarray:
        rng_seed = 11  # deterministic
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = re.findall(r"\w+", t.lower()) or ["<empty>"]
            for tok in toks:
                rng = np.random.default_rng(abs(hash(tok)) % (2 ** 32) + rng_seed)
                out[i] += rng.standard_normal(self.dim).astype(np.float32)
            out[i] /= max(len(toks), 1)
        # L2 normalise so cosine sim behaves sensibly
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)

    # -- public API -----------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a list of strings; returns (n, dim) float32 array."""
        if isinstance(texts, str):
            texts = [texts]
        if self._fallback or self._model is None:
            return self._fallback_embed(texts)
        import torch  # local import for environments without torch
        ids = self._tokenizer.encode_batch(texts, max_length=self.max_length)
        mask = (ids != 0).astype(np.int64)
        with torch.no_grad():
            out = self._model(
                input_ids=torch.tensor(ids),
                attention_mask=torch.tensor(mask),
            ).last_hidden_state.cpu().numpy()
        pooled = mean_pool(out, mask=mask)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-9, None)).astype(np.float32)


# ============================================================================
# Similarity helpers
# ============================================================================


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between rows of ``a`` and rows of ``b``.

    a: (n, d), b: (m, d) -> (n, m)
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
    return an @ bn.T


def top_k_similar(query: np.ndarray, corpus: np.ndarray, k: int = 5
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (indices, scores) of the ``k`` rows of ``corpus`` most similar to ``query``.
    """
    sims = cosine_sim(query, corpus)[0]
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


__all__ = [
    "LLMTokenizerWrapper",
    "EmbeddingExtractor",
    "mean_pool",
    "cosine_sim",
    "top_k_similar",
]

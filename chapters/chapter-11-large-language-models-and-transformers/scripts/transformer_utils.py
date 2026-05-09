"""
Pure-NumPy transformer building blocks for Chapter 11.

Implements the math from "Attention Is All You Need" (Vaswani et al., 2017)
in a way that runs without PyTorch / TensorFlow. These are *forward-only*
demonstration classes — they are useful for understanding shapes, masks, and
information flow, not for training real models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ----------------------------- core math helpers -----------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalisation along the last axis (no learnable affine)."""
    mu = x.mean(axis=-1, keepdims=True)
    sigma = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sigma + eps)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (tanh approximation)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


# --------------------------- scaled dot-product attn -------------------------

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ``softmax(QK^T / sqrt(d_k)) V``.

    Q: (..., seq_q, d_k)
    K: (..., seq_k, d_k)
    V: (..., seq_k, d_v)
    mask: optional (..., seq_q, seq_k) of 0/1 (1 = keep). Positions with 0 get -inf.

    Returns (output, attention_weights).
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask.astype(bool), scores, -1e9)
    weights = softmax(scores, axis=-1)
    out = np.matmul(weights, V)
    return out, weights


def causal_mask(seq_len: int) -> np.ndarray:
    """Lower-triangular (1 = keep) mask for autoregressive attention."""
    return np.tril(np.ones((seq_len, seq_len), dtype=np.int32))


# ------------------------------- multi-head -----------------------------------

class MultiHeadAttention:
    """
    Forward-only multi-head self-attention.

    Splits ``d_model`` into ``num_heads`` heads of size ``d_model // num_heads``,
    runs scaled dot-product attention per head in parallel, then concatenates
    and projects back to ``d_model``.
    """

    def __init__(self, d_model: int, num_heads: int, seed: int = 42) -> None:
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.Wq = rng.standard_normal((d_model, d_model)) * scale
        self.Wk = rng.standard_normal((d_model, d_model)) * scale
        self.Wv = rng.standard_normal((d_model, d_model)) * scale
        self.Wo = rng.standard_normal((d_model, d_model)) * scale
        self.last_attn_weights: Optional[np.ndarray] = None

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, seq, d_model) -> (batch, num_heads, seq, d_head)."""
        b, s, _ = x.shape
        x = x.reshape(b, s, self.num_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, num_heads, seq, d_head) -> (batch, seq, d_model)."""
        b, _, s, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, s, self.d_model)

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Self-attention: keys, queries and values all come from ``x``."""
        if x.ndim == 2:
            x = x[None, :, :]  # promote to batch=1
        Q = self._split_heads(x @ self.Wq)
        K = self._split_heads(x @ self.Wk)
        V = self._split_heads(x @ self.Wv)
        if mask is not None and mask.ndim == 2:
            mask = mask[None, None, :, :]
        out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        self.last_attn_weights = weights
        return self._combine_heads(out) @ self.Wo


# ----------------------------- positional encoding ---------------------------

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    Returns an array of shape (seq_len, d_model) where even dimensions use
    ``sin`` and odd dimensions use ``cos`` at geometrically spaced wavelengths.
    """
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / d_model)
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


# ------------------------------ encoder block --------------------------------

class TransformerBlock:
    """
    Single transformer encoder block:

        x = LayerNorm(x + MHA(x))
        x = LayerNorm(x + FFN(x))

    Uses a 2-layer MLP with GELU as the position-wise feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_hidden: int,
        seed: int = 42,
    ) -> None:
        self.attn = MultiHeadAttention(d_model, num_heads, seed=seed)
        rng = np.random.default_rng(seed + 1)
        scale = 1.0 / np.sqrt(d_model)
        self.W1 = rng.standard_normal((d_model, ffn_hidden)) * scale
        self.b1 = np.zeros(ffn_hidden)
        self.W2 = rng.standard_normal((ffn_hidden, d_model)) * scale
        self.b2 = np.zeros(d_model)

    def _ffn(self, x: np.ndarray) -> np.ndarray:
        return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        x = layer_norm(x + self.attn(x, mask=mask))
        x = layer_norm(x + self._ffn(x))
        return x


# ----------------------------- plotting helper --------------------------------

def plot_attention(
    weights: np.ndarray,
    tokens: Optional[list] = None,
    head: int = 0,
    title: str = "Attention weights",
):
    """
    Visualise an attention matrix as a heatmap.

    ``weights`` may be (seq, seq), (heads, seq, seq) or (batch, heads, seq, seq).
    Returns the matplotlib axes (or ``None`` if matplotlib is unavailable).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; cannot plot.")
        return None
    w = np.asarray(weights)
    if w.ndim == 4:
        w = w[0, head]
    elif w.ndim == 3:
        w = w[head]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(w, cmap="viridis", aspect="auto")
    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticklabels(tokens)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return ax


__all__ = [
    "softmax",
    "layer_norm",
    "gelu",
    "scaled_dot_product_attention",
    "causal_mask",
    "MultiHeadAttention",
    "positional_encoding",
    "TransformerBlock",
    "plot_attention",
]

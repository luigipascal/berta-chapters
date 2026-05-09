"""
Decoding / generation utilities for Chapter 11.

All functions operate on raw NumPy logit arrays so they can be exercised on
toy distributions without needing PyTorch or a real LM. The same algorithms
underpin ``model.generate`` in libraries such as Hugging Face ``transformers``.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np


# ----------------------------- helpers ---------------------------------------

def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits)
    return e / np.sum(e, axis=axis, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Sharpen (T < 1) or flatten (T > 1) a logit distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def apply_repetition_penalty(
    logits: np.ndarray,
    generated: Sequence[int],
    penalty: float = 1.1,
) -> np.ndarray:
    """
    Discourage previously generated tokens by dividing positive logits and
    multiplying negative logits by ``penalty`` (CTRL paper, Keskar et al. 2019).
    """
    if penalty == 1.0 or not generated:
        return logits
    out = logits.copy()
    for tok in set(generated):
        if 0 <= tok < out.shape[-1]:
            v = out[..., tok]
            out[..., tok] = np.where(v > 0, v / penalty, v * penalty)
    return out


# ----------------------------- single-step samplers --------------------------

def greedy_step(logits: np.ndarray) -> int:
    """Pick the argmax of a 1-D logit vector."""
    return int(np.argmax(logits))


def sample_with_temperature(
    logits: np.ndarray,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Categorical sample from temperature-scaled logits."""
    rng = rng or np.random.default_rng()
    probs = _softmax(apply_temperature(logits, temperature))
    return int(rng.choice(len(probs), p=probs))


def top_k_sample(
    logits: np.ndarray,
    k: int = 50,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Restrict sampling to the top-``k`` highest-probability tokens
    (Fan et al. 2018, "Hierarchical Neural Story Generation").
    """
    rng = rng or np.random.default_rng()
    logits = apply_temperature(logits, temperature)
    if k >= logits.shape[-1]:
        probs = _softmax(logits)
        return int(rng.choice(len(probs), p=probs))
    top_idx = np.argpartition(-logits, k - 1)[:k]
    top_logits = logits[top_idx]
    probs = _softmax(top_logits)
    return int(top_idx[rng.choice(len(top_idx), p=probs)])


def top_p_sample(
    logits: np.ndarray,
    p: float = 0.9,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability exceeds ``p`` (Holtzman et al. 2019).
    """
    rng = rng or np.random.default_rng()
    if not 0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    logits = apply_temperature(logits, temperature)
    probs = _softmax(logits)
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cum = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cum, p) + 1)
    cutoff = max(cutoff, 1)
    keep = order[:cutoff]
    kp = probs[keep] / probs[keep].sum()
    return int(keep[rng.choice(len(keep), p=kp)])


# ----------------------------- decoding loops --------------------------------

LogitsFn = Callable[[List[int]], np.ndarray]


def greedy_decode(
    logits_fn: LogitsFn,
    prompt: Sequence[int],
    max_new_tokens: int = 32,
    eos_token_id: Optional[int] = None,
) -> List[int]:
    """
    Iteratively call ``logits_fn(generated)`` and append the argmax.

    ``logits_fn`` should return a 1-D logit array over the vocabulary given
    the current prefix.
    """
    out = list(prompt)
    for _ in range(max_new_tokens):
        nxt = greedy_step(logits_fn(out))
        out.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
    return out


def sample_decode(
    logits_fn: LogitsFn,
    prompt: Sequence[int],
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Generic sampling loop combining temperature, top-k/top-p and repetition
    penalty. Pass ``temperature=0`` to fall back to greedy decoding.
    """
    rng = rng or np.random.default_rng()
    out = list(prompt)
    for _ in range(max_new_tokens):
        logits = logits_fn(out)
        logits = apply_repetition_penalty(logits, out, penalty=repetition_penalty)
        if temperature == 0:
            nxt = greedy_step(logits)
        elif top_p is not None:
            nxt = top_p_sample(logits, p=top_p, temperature=temperature, rng=rng)
        elif top_k is not None:
            nxt = top_k_sample(logits, k=top_k, temperature=temperature, rng=rng)
        else:
            nxt = sample_with_temperature(logits, temperature=temperature, rng=rng)
        out.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
    return out


# ----------------------------- evaluation ------------------------------------

def perplexity(log_probs: Sequence[float]) -> float:
    """
    Perplexity = exp(- mean log p(token)). Lower is better.

    ``log_probs`` should be the natural log-probabilities the model assigns
    to each ground-truth token.
    """
    if len(log_probs) == 0:
        raise ValueError("log_probs must be non-empty")
    return float(np.exp(-np.mean(log_probs)))


__all__ = [
    "apply_temperature",
    "apply_repetition_penalty",
    "greedy_step",
    "sample_with_temperature",
    "top_k_sample",
    "top_p_sample",
    "greedy_decode",
    "sample_decode",
    "perplexity",
]

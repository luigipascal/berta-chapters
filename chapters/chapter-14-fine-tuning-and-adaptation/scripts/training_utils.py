"""
Training utilities for Chapter 14: Fine-tuning & Adaptation Techniques.

CPU-only NumPy / scikit-learn building blocks that mirror the pieces of a
real SFT training loop: forward pass, masked loss, gradient step, learning
rate schedule, early stopping, and an evaluation harness with exact match,
F1, and a held-out win-rate stub.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Learning-rate schedules
# --------------------------------------------------------------------------- #


class LRScheduler:
    """
    Linear warmup followed by cosine or linear decay.

    `step()` returns the LR for the current step and advances the counter.
    """

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        kind: str = "cosine",
        min_lr: float = 0.0,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if warmup_steps < 0 or warmup_steps > total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps].")
        if kind not in {"cosine", "linear"}:
            raise ValueError("kind must be 'cosine' or 'linear'.")
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.kind = kind
        self.min_lr = min_lr
        self._step = 0

    def lr_at(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        if self.kind == "linear":
            return self.min_lr + (self.base_lr - self.min_lr) * (1.0 - progress)
        # cosine
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self) -> float:
        lr = self.lr_at(self._step)
        self._step += 1
        return lr


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #


@dataclass
class EarlyStopping:
    """Stop when validation metric has not improved for `patience` checks."""

    patience: int = 3
    min_delta: float = 0.0
    mode: str = "min"  # "min" for loss, "max" for accuracy
    best: float = field(init=False, default=math.inf)
    bad_epochs: int = field(init=False, default=0)
    stopped: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        if self.mode == "max":
            self.best = -math.inf

    def update(self, metric: float) -> bool:
        improved = (
            metric < self.best - self.min_delta if self.mode == "min" else metric > self.best + self.min_delta
        )
        if improved:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True
        return self.stopped


# --------------------------------------------------------------------------- #
# Tiny SFT loop on a linear "head" — analog of fine-tuning a classifier head
# --------------------------------------------------------------------------- #


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def masked_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray]:
    """
    Cross-entropy averaged over `mask == 1` positions only.

    logits: (N, V), targets: (N,), mask: (N,) of {0, 1}.
    Returns (loss, dlogits) so a caller can backprop.
    """
    if logits.ndim != 2 or targets.ndim != 1 or mask.ndim != 1:
        raise ValueError("Expected logits (N,V), targets (N,), mask (N,).")
    n, v = logits.shape
    probs = softmax(logits, axis=-1)
    correct_logp = -np.log(probs[np.arange(n), targets] + eps)
    masked = correct_logp * mask
    denom = mask.sum() if mask.sum() > 0 else 1.0
    loss = float(masked.sum() / denom)
    # gradient wrt logits (only masked positions contribute)
    grad = probs.copy()
    grad[np.arange(n), targets] -= 1.0
    grad *= (mask / denom)[:, None]
    return loss, grad


def simple_sft_loop(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    n_classes: int,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-2,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    seed: int = 42,
    val_split: float = 0.2,
    early_stop_patience: int = 0,
    verbose: bool = False,
) -> Dict:
    """
    Tiny SFT analog: train a linear "head" W, b on (X, y) with a per-token
    loss mask. Mirrors the shape of a real SFT loop.

    Returns history of train / val loss and the final parameters.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    W = rng.normal(scale=0.01, size=(d, n_classes))
    b = np.zeros(n_classes)

    total_steps = max(1, (len(train_idx) // batch_size) * epochs)
    sched = LRScheduler(lr, total_steps, warmup_steps=int(warmup_ratio * total_steps))
    stopper = EarlyStopping(patience=early_stop_patience, mode="min") if early_stop_patience > 0 else None
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}

    for ep in range(epochs):
        rng.shuffle(train_idx)
        epoch_loss = 0.0
        nb = 0
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start : start + batch_size]
            logits = X[idx] @ W + b
            loss, dlogits = masked_cross_entropy(logits, y[idx], mask[idx], )
            grad_W = X[idx].T @ dlogits + weight_decay * W
            grad_b = dlogits.sum(axis=0)
            # gradient clipping (global L2 norm)
            gnorm = math.sqrt(float((grad_W ** 2).sum() + (grad_b ** 2).sum()))
            if gnorm > grad_clip and gnorm > 0:
                grad_W *= grad_clip / gnorm
                grad_b *= grad_clip / gnorm
            cur_lr = sched.step()
            W -= cur_lr * grad_W
            b -= cur_lr * grad_b
            epoch_loss += loss
            nb += 1
        train_loss = epoch_loss / max(1, nb)
        val_logits = X[val_idx] @ W + b
        val_loss, _ = masked_cross_entropy(val_logits, y[val_idx], mask[val_idx])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(cur_lr)
        if verbose:
            logger.info("epoch=%d train=%.4f val=%.4f lr=%.5f", ep, train_loss, val_loss, cur_lr)
        if stopper and stopper.update(val_loss):
            break

    return {"W": W, "b": b, "history": history, "val_idx": val_idx, "train_idx": train_idx}


# --------------------------------------------------------------------------- #
# Evaluation harness
# --------------------------------------------------------------------------- #


def normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1, the SQuAD-style metric, on whitespace tokens."""
    p_toks = normalize_text(pred).split()
    g_toks = normalize_text(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common: Dict[str, int] = {}
    for t in p_toks:
        if t in g_toks:
            common[t] = min(p_toks.count(t), g_toks.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def win_rate_stub(
    preds_a: Sequence[str],
    preds_b: Sequence[str],
    references: Sequence[str],
    judge: Optional[Callable[[str, str, str], int]] = None,
) -> Dict[str, float]:
    """
    Held-out win-rate of model A vs B against `references`.

    `judge(pred_a, pred_b, ref) -> {1, 0, -1}` (A wins / tie / B wins).
    Default judge prefers higher token-F1 against the reference, with ties
    broken to "tie".
    """
    if not (len(preds_a) == len(preds_b) == len(references)):
        raise ValueError("preds_a, preds_b, references must be equal length.")

    def default_judge(a: str, b: str, ref: str) -> int:
        fa, fb = token_f1(a, ref), token_f1(b, ref)
        if abs(fa - fb) < 1e-9:
            return 0
        return 1 if fa > fb else -1

    j = judge or default_judge
    a_wins = ties = b_wins = 0
    for a, b, ref in zip(preds_a, preds_b, references):
        v = j(a, b, ref)
        if v > 0:
            a_wins += 1
        elif v < 0:
            b_wins += 1
        else:
            ties += 1
    n = len(references)
    return {
        "n": n,
        "a_win_rate": a_wins / n if n else 0.0,
        "b_win_rate": b_wins / n if n else 0.0,
        "tie_rate": ties / n if n else 0.0,
    }


@dataclass
class EvalHarness:
    """Aggregate exact-match, F1, and a win-rate stub on a held-out set."""

    references: Sequence[str]

    def score(self, predictions: Sequence[str]) -> Dict[str, float]:
        if len(predictions) != len(self.references):
            raise ValueError("predictions / references length mismatch.")
        n = len(predictions)
        if n == 0:
            return {"n": 0, "exact_match": 0.0, "f1": 0.0}
        em = sum(exact_match(p, g) for p, g in zip(predictions, self.references)) / n
        f1 = sum(token_f1(p, g) for p, g in zip(predictions, self.references)) / n
        return {"n": n, "exact_match": em, "f1": f1}

    def compare(self, preds_a: Sequence[str], preds_b: Sequence[str]) -> Dict[str, float]:
        return win_rate_stub(preds_a, preds_b, self.references)


if __name__ == "__main__":
    # Smoke tests
    sched = LRScheduler(1e-3, total_steps=10, warmup_steps=2, kind="cosine")
    lrs = [sched.step() for _ in range(10)]
    assert lrs[0] < lrs[1] <= 1e-3 and lrs[-1] >= 0.0
    es = EarlyStopping(patience=2, mode="min")
    es.update(1.0); es.update(0.9); es.update(1.1)
    assert es.update(1.2) is True

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = (X[:, 0] > 0).astype(int)
    mask = np.ones(40, dtype=int)
    out = simple_sft_loop(X, y, mask, n_classes=2, epochs=3, batch_size=8, lr=0.05, val_split=0.25)
    assert out["history"]["train_loss"][-1] <= out["history"]["train_loss"][0] + 1e-6

    h = EvalHarness(references=["a b c", "d e"])
    s = h.score(["a b c", "d f"])
    assert 0.0 <= s["f1"] <= 1.0
    cmp = h.compare(["a b c", "d e"], ["a b", "x y"])
    assert math.isclose(cmp["a_win_rate"] + cmp["b_win_rate"] + cmp["tie_rate"], 1.0)
    print("training_utils self-tests passed.")

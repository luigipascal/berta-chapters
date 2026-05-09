"""
Parameter-efficient fine-tuning (PEFT) utilities for Chapter 14.

A self-contained NumPy implementation of LoRA: low-rank adapters that wrap a
frozen linear layer. The forward pass mirrors the math in the LoRA paper
(Hu et al., 2021):

    y = x W^T + x A^T B^T * (alpha / r)

where W (out, in) is frozen, A (r, in) and B (out, r) are trainable, `alpha`
is a scaling factor, and `r` is the rank of the update.

This module is intentionally framework-free so it runs on a CPU laptop.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Frozen linear "base" layer
# --------------------------------------------------------------------------- #


@dataclass
class LinearLayer:
    """Minimal frozen linear layer: y = x W^T + b."""

    W: np.ndarray  # shape (out_features, in_features)
    b: Optional[np.ndarray] = None  # shape (out_features,)
    frozen: bool = True

    @classmethod
    def random(cls, in_features: int, out_features: int, seed: int = 0) -> "LinearLayer":
        rng = np.random.default_rng(seed)
        W = rng.normal(scale=1.0 / math.sqrt(in_features), size=(out_features, in_features))
        b = np.zeros(out_features)
        return cls(W=W, b=b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W.T
        if self.b is not None:
            out = out + self.b
        return out

    def num_params(self) -> int:
        n = self.W.size
        if self.b is not None:
            n += self.b.size
        return n


# --------------------------------------------------------------------------- #
# LoRA adapter
# --------------------------------------------------------------------------- #


@dataclass
class LoRALayer:
    """
    Low-rank adapter for a linear layer.

    Forward:
        delta = (x @ A.T) @ B.T * (alpha / r)
        y     = base.forward(x) + dropout(delta)

    A is initialized to small random values and B to zeros so the adapter is
    a no-op at initialization.
    """

    in_features: int
    out_features: int
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    A: np.ndarray = field(init=False)
    B: np.ndarray = field(init=False)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be > 0.")
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be > 0.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        rng = np.random.default_rng(self.seed)
        # Kaiming-ish init for A, zeros for B (so initial delta == 0).
        self.A = rng.normal(scale=1.0 / math.sqrt(self.in_features), size=(self.rank, self.in_features))
        self.B = np.zeros((self.out_features, self.rank))

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def delta(self, x: np.ndarray) -> np.ndarray:
        return (x @ self.A.T) @ self.B.T * self.scaling

    def forward(self, x: np.ndarray, base: LinearLayer, training: bool = False) -> np.ndarray:
        out = base.forward(x)
        d = self.delta(x)
        if training and self.dropout > 0.0:
            rng = np.random.default_rng()
            keep = 1.0 - self.dropout
            mask = (rng.random(d.shape) < keep).astype(d.dtype)
            d = d * mask / keep
        return out + d

    def num_trainable_params(self) -> int:
        return self.A.size + self.B.size


def apply_lora_to(
    base: LinearLayer,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    seed: int = 0,
) -> LoRALayer:
    """Construct a LoRA adapter shaped to a given frozen linear layer."""
    out_f, in_f = base.W.shape
    return LoRALayer(in_features=in_f, out_features=out_f, rank=rank, alpha=alpha, dropout=dropout, seed=seed)


def merge_lora(base: LinearLayer, lora: LoRALayer) -> LinearLayer:
    """
    Fold the LoRA update back into the base weights:

        W_merged = W + (B @ A) * (alpha / r)

    Returns a new LinearLayer; the original is not mutated.
    """
    if base.W.shape[0] != lora.out_features or base.W.shape[1] != lora.in_features:
        raise ValueError("LoRA shape does not match base layer.")
    delta_W = lora.B @ lora.A * lora.scaling
    return LinearLayer(W=base.W + delta_W, b=None if base.b is None else base.b.copy(), frozen=base.frozen)


def count_trainable_params(base: LinearLayer, lora: Optional[LoRALayer] = None) -> Dict[str, int]:
    """Report parameter counts and the parameter-efficiency ratio."""
    base_params = base.num_params()
    lora_params = lora.num_trainable_params() if lora is not None else 0
    total = base_params + lora_params
    trainable = lora_params if lora is not None and base.frozen else total
    return {
        "base_params": base_params,
        "lora_params": lora_params,
        "trainable": trainable,
        "total": total,
        "trainable_fraction": trainable / total if total else 0.0,
    }


# --------------------------------------------------------------------------- #
# Multi-adapter registry — serve several LoRA adapters from one base
# --------------------------------------------------------------------------- #


@dataclass
class AdapterRegistry:
    """Hold multiple named LoRA adapters that share a frozen base."""

    base: LinearLayer
    adapters: Dict[str, LoRALayer] = field(default_factory=dict)

    def add(self, name: str, lora: LoRALayer) -> None:
        if name in self.adapters:
            raise KeyError(f"Adapter '{name}' already registered.")
        if lora.in_features != self.base.W.shape[1] or lora.out_features != self.base.W.shape[0]:
            raise ValueError("Adapter shape does not match base layer.")
        self.adapters[name] = lora

    def list(self) -> List[str]:
        return list(self.adapters.keys())

    def forward(self, x: np.ndarray, name: Optional[str] = None) -> np.ndarray:
        if name is None:
            return self.base.forward(x)
        if name not in self.adapters:
            raise KeyError(f"Unknown adapter '{name}'.")
        return self.adapters[name].forward(x, self.base)


if __name__ == "__main__":
    # Self-tests: forward shape, no-op at init, merge equivalence, params.
    rng = np.random.default_rng(0)
    base = LinearLayer.random(in_features=16, out_features=8, seed=1)
    x = rng.normal(size=(4, 16))

    lora = apply_lora_to(base, rank=4, alpha=8.0)
    out_init = lora.forward(x, base)
    assert np.allclose(out_init, base.forward(x)), "LoRA must be a no-op at init."

    # Set B != 0 and check that merge produces equivalent outputs.
    lora.B = rng.normal(scale=0.1, size=lora.B.shape)
    out_with_adapter = lora.forward(x, base)
    merged = merge_lora(base, lora)
    out_merged = merged.forward(x)
    assert np.allclose(out_with_adapter, out_merged, atol=1e-6), "Merged forward must match adapter forward."

    counts = count_trainable_params(base, lora)
    assert counts["lora_params"] == 4 * 16 + 8 * 4
    assert counts["trainable_fraction"] < 1.0

    reg = AdapterRegistry(base=base)
    reg.add("task_a", lora)
    assert "task_a" in reg.list()
    out_named = reg.forward(x, "task_a")
    assert out_named.shape == (4, 8)
    print("peft_utils self-tests passed.")

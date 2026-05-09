"""
Monitoring utilities for Chapter 15: MLOps & Model Deployment.

Provides:
    - psi(reference, current, bins=10): Population Stability Index
    - ks_stat(reference, current): two-sample KS statistic + p-value
    - LatencyTracker: rolling latency percentiles
    - DriftDetector: orchestrates feature-by-feature drift checks
    - Logger: structured JSON-line logger for predictions and events
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Drift statistics
# ---------------------------------------------------------------------------

def psi(
    reference: Sequence[float],
    current: Sequence[float],
    bins: int = 10,
) -> float:
    """
    Population Stability Index.

    PSI = sum( (p_curr - p_ref) * ln(p_curr / p_ref) ) over equal-width bins.

    Rule of thumb:
        < 0.10  -> no significant change
        0.10–0.25 -> moderate shift, monitor
        > 0.25  -> significant shift, alert
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        return 0.0
    edges = np.linspace(
        min(ref.min(), cur.min()),
        max(ref.max(), cur.max()),
        bins + 1,
    )
    # Make sure final edge captures the max
    edges[-1] = edges[-1] + _EPS
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    p_ref = (ref_counts / max(ref.size, 1)) + _EPS
    p_cur = (cur_counts / max(cur.size, 1)) + _EPS
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def ks_stat(
    reference: Sequence[float],
    current: Sequence[float],
) -> Dict[str, float]:
    """
    Two-sample Kolmogorov-Smirnov statistic with an asymptotic p-value
    approximation. NumPy-only; avoids a SciPy dependency.

    Returns:
        {'statistic': D, 'pvalue': p_approx}
    """
    a = np.sort(np.asarray(reference, dtype=float))
    b = np.sort(np.asarray(current, dtype=float))
    if a.size == 0 or b.size == 0:
        return {"statistic": 0.0, "pvalue": 1.0}
    data_all = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, data_all, side="right") / a.size
    cdf_b = np.searchsorted(b, data_all, side="right") / b.size
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n = a.size * b.size / (a.size + b.size)
    # Marsaglia-style asymptotic approximation
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    # Series expansion of Q(lam)
    p = 0.0
    for j in range(1, 101):
        term = ((-1) ** (j - 1)) * math.exp(-2 * (lam ** 2) * (j ** 2))
        p += term
    p = max(0.0, min(1.0, 2 * p))
    return {"statistic": d, "pvalue": p}


# ---------------------------------------------------------------------------
# Latency tracking
# ---------------------------------------------------------------------------

@dataclass
class LatencyTracker:
    """Rolling-window latency tracker producing percentile reports."""
    window: int = 1000
    samples: List[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        self.samples.append(float(latency_ms))
        if len(self.samples) > self.window:
            self.samples = self.samples[-self.window:]

    def percentile(self, q: float) -> float:
        if not self.samples:
            return 0.0
        return float(np.percentile(self.samples, q))

    def report(self) -> Dict[str, float]:
        if not self.samples:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        a = np.asarray(self.samples)
        return {
            "count": int(a.size),
            "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
        }


# ---------------------------------------------------------------------------
# Drift orchestrator
# ---------------------------------------------------------------------------

@dataclass
class DriftDetector:
    """
    Computes per-feature drift between a reference window and a current
    window, flagging features whose PSI or KS p-value exceed thresholds.
    """
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    ks_pvalue: float = 0.05
    bins: int = 10

    def detect(
        self,
        reference: Dict[str, Sequence[float]],
        current: Dict[str, Sequence[float]],
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {"features": {}, "alerts": [], "warnings": []}
        for name in reference:
            if name not in current:
                continue
            score = psi(reference[name], current[name], bins=self.bins)
            ks = ks_stat(reference[name], current[name])
            level = "ok"
            if score >= self.psi_alert or ks["pvalue"] < self.ks_pvalue:
                level = "alert"
                report["alerts"].append(name)
            elif score >= self.psi_warn:
                level = "warning"
                report["warnings"].append(name)
            report["features"][name] = {
                "psi": score,
                "ks_statistic": ks["statistic"],
                "ks_pvalue": ks["pvalue"],
                "level": level,
            }
        report["overall_level"] = (
            "alert" if report["alerts"]
            else ("warning" if report["warnings"] else "ok")
        )
        return report


# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------

class Logger:
    """Append-only JSON-lines logger for predictions and operational events."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> Dict[str, Any]:
        """Write one JSON object per line. Returns the record for inspection."""
        record: Dict[str, Any] = {
            "timestamp": time.time(),
            "event": event,
        }
        record.update(fields)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return record

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all logged records (small files / tests only)."""
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


__all__ = ["psi", "ks_stat", "LatencyTracker", "DriftDetector", "Logger"]

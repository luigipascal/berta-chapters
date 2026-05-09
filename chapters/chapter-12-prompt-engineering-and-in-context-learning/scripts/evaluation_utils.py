"""
Evaluation utilities for Chapter 12.

Includes:
- Pure-function graders (`exact_match`, `regex_match`, `cosine_match`).
- A `RubricGrader` that aggregates multiple boolean/scalar checks.
- A `PromptABTester` with a small bootstrap CI for win-rate differences.
- A `PromptEvalHarness` orchestrator that runs prompts over a labeled
  dataset and computes summary metrics.

Everything runs offline against a `BaseLLMClient` (typically `MockLLMClient`).
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-function graders
# ---------------------------------------------------------------------------


def exact_match(prediction: str, reference: str, case_sensitive: bool = False) -> float:
    """1.0 iff strings match exactly (after optional case fold + whitespace strip)."""
    p = prediction.strip()
    r = reference.strip()
    if not case_sensitive:
        p, r = p.lower(), r.lower()
    return 1.0 if p == r else 0.0


def regex_match(prediction: str, pattern: str, flags: int = re.IGNORECASE) -> float:
    """1.0 iff `pattern` matches anywhere in the prediction."""
    return 1.0 if re.search(pattern, prediction, flags=flags) else 0.0


def cosine_match(prediction: str, reference: str, threshold: float = 0.0) -> float:
    """
    Embedding-style similarity using TF-IDF + cosine. Returns the raw similarity
    (0.0–1.0). If `threshold > 0`, returns 1.0/0.0 instead of the score.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        raise ImportError("scikit-learn required: pip install scikit-learn") from e

    if not prediction.strip() or not reference.strip():
        return 0.0
    try:
        vec = TfidfVectorizer().fit([prediction, reference])
        m = vec.transform([prediction, reference])
        sim = float(cosine_similarity(m[0:1], m[1:2])[0, 0])
    except ValueError:
        # Empty vocabulary after stopword removal etc.
        sim = 0.0
    if threshold > 0:
        return 1.0 if sim >= threshold else 0.0
    return sim


# ---------------------------------------------------------------------------
# Rubric grader
# ---------------------------------------------------------------------------


GraderFn = Callable[[str, Any], float]


@dataclass
class RubricItem:
    """A single rubric line: a name + a callable that returns a 0..1 score."""

    name: str
    grader: GraderFn
    weight: float = 1.0


@dataclass
class RubricGrader:
    """
    Aggregate multiple criteria into a weighted average score.

    Each item's grader is called with `(prediction, reference)` and must
    return a float in [0, 1]. Final score is the weighted mean.
    """

    items: List[RubricItem]

    def grade(self, prediction: str, reference: Any) -> Dict[str, float]:
        if not self.items:
            return {"score": 0.0}
        scores: Dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0
        for item in self.items:
            try:
                s = float(item.grader(prediction, reference))
            except Exception as e:
                logger.warning("Rubric '%s' failed: %s", item.name, e)
                s = 0.0
            s = max(0.0, min(1.0, s))
            scores[item.name] = s
            weighted_sum += s * item.weight
            total_weight += item.weight
        scores["score"] = weighted_sum / total_weight if total_weight else 0.0
        return scores


# ---------------------------------------------------------------------------
# A/B tester
# ---------------------------------------------------------------------------


@dataclass
class ABTestResult:
    """Container returned by `PromptABTester.run`."""

    mean_a: float
    mean_b: float
    diff: float
    diff_ci_low: float
    diff_ci_high: float
    n: int
    significant: bool


class PromptABTester:
    """
    Compare two scalar score arrays from prompts A and B.

    Uses non-parametric bootstrap to produce a CI for the mean difference.
    A/B is considered significant if the CI excludes zero.
    """

    def __init__(self, n_iterations: int = 1000, ci: float = 0.95, seed: int = 42):
        self.n_iterations = n_iterations
        self.ci = ci
        self.seed = seed

    def run(self, scores_a: Sequence[float], scores_b: Sequence[float]) -> ABTestResult:
        if len(scores_a) != len(scores_b) or not scores_a:
            raise ValueError("scores_a and scores_b must be non-empty and equal length")
        rng = random.Random(self.seed)
        n = len(scores_a)
        mean_a = sum(scores_a) / n
        mean_b = sum(scores_b) / n
        diffs: List[float] = []
        for _ in range(self.n_iterations):
            idxs = [rng.randrange(n) for _ in range(n)]
            ma = sum(scores_a[i] for i in idxs) / n
            mb = sum(scores_b[i] for i in idxs) / n
            diffs.append(mb - ma)
        diffs.sort()
        alpha = (1 - self.ci) / 2
        lo = diffs[int(alpha * self.n_iterations)]
        hi = diffs[int((1 - alpha) * self.n_iterations) - 1]
        return ABTestResult(
            mean_a=mean_a,
            mean_b=mean_b,
            diff=mean_b - mean_a,
            diff_ci_low=lo,
            diff_ci_high=hi,
            n=n,
            significant=(lo > 0 or hi < 0),
        )


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------


@dataclass
class EvalRecord:
    """One row of an evaluation: input, expected, prediction, scores, latency."""

    task_id: str
    input: str
    reference: str
    prediction: str
    scores: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0


class PromptEvalHarness:
    """
    Orchestrate prompt evaluation over a labeled task set.

    Usage:
        harness = PromptEvalHarness(client, render_fn, grader_fn)
        report = harness.run(rows)  # rows = list of dicts with 'input'/'reference'

    `render_fn(input_text)` returns the rendered prompt string.
    `grader_fn(prediction, reference)` returns a dict including a 'score' key.
    """

    def __init__(
        self,
        client: Any,
        render_fn: Callable[[str], str],
        grader_fn: Callable[[str, str], Dict[str, float]],
    ):
        self.client = client
        self.render_fn = render_fn
        self.grader_fn = grader_fn

    def run(self, rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        import time

        records: List[EvalRecord] = []
        for row in rows:
            task_id = str(row.get("task_id", len(records)))
            inp = row["input"]
            ref = row.get("reference_output", row.get("reference", ""))
            prompt = self.render_fn(inp)
            t0 = time.perf_counter()
            response = self.client.complete(prompt)
            t1 = time.perf_counter()
            pred = getattr(response, "text", str(response))
            scores = self.grader_fn(pred, ref)
            records.append(
                EvalRecord(
                    task_id=task_id,
                    input=inp,
                    reference=ref,
                    prediction=pred,
                    scores=scores,
                    latency_ms=(t1 - t0) * 1000,
                )
            )
        if not records:
            return {"records": [], "metrics": {}}
        agg: Dict[str, float] = {}
        keys = set()
        for r in records:
            keys.update(r.scores.keys())
        for k in keys:
            vals = [r.scores.get(k, 0.0) for r in records]
            agg[k] = sum(vals) / len(vals)
        agg["latency_ms_mean"] = sum(r.latency_ms for r in records) / len(records)
        return {"records": records, "metrics": agg}


# ---------------------------------------------------------------------------
# Prompt-injection detection helpers
# ---------------------------------------------------------------------------


def detect_injection(text: str, patterns: Optional[Sequence[str]] = None) -> List[str]:
    """
    Return the list of pattern names that match `text`.

    Patterns default to a small starter set covering common injection phrases.
    """
    default_patterns = [
        r"ignore (all )?(previous|prior|above) instructions",
        r"disregard (the )?(system|previous)",
        r"reveal (your )?(system )?prompt",
        r"forget (everything|all)",
        r"act as .*(?:dan|jailbreak|developer mode)",
        r"override.*safety",
        r"print.*system prompt",
    ]
    used = list(patterns) if patterns is not None else default_patterns
    hits: List[str] = []
    for p in used:
        if re.search(p, text, flags=re.IGNORECASE):
            hits.append(p)
    return hits


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON extraction: try to load `text`; if that fails, find the
    first {...} block and try again. Returns None on total failure.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


__all__ = [
    "exact_match",
    "regex_match",
    "cosine_match",
    "RubricItem",
    "RubricGrader",
    "PromptABTester",
    "ABTestResult",
    "PromptEvalHarness",
    "EvalRecord",
    "detect_injection",
    "safe_json_parse",
]

"""
LLM client abstractions for Chapter 12.

All notebooks and exercises run **offline**: the default `MockLLMClient`
returns deterministic, rule-based responses so prompt-engineering patterns
can be exercised without API keys, network access, or cost.

Real-provider stubs (`OpenAIClient`, `AnthropicClient`) are included to show
the wiring; they raise NotImplementedError unless the corresponding SDK is
installed. They are intentionally never called by the chapter's notebooks.
"""

from __future__ import annotations

import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response & base client
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Uniform LLM response across providers and the mock client."""

    text: str
    model: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseLLMClient(ABC):
    """Abstract base for all LLM clients used in this chapter."""

    def __init__(self, model: str = "base-llm", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Plain-text completion."""

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        """
        Default chat impl: flatten messages and call `complete`. Subclasses
        with native chat APIs (OpenAI, Anthropic) override this.
        """
        flat = "\n\n".join(
            f"[{m.get('role', 'user').upper()}] {m.get('content', '')}"
            for m in messages
        )
        return self.complete(flat, **kwargs)


# ---------------------------------------------------------------------------
# Mock client (deterministic, rule-based)
# ---------------------------------------------------------------------------


class MockLLMClient(BaseLLMClient):
    """
    Deterministic, offline LLM stand-in.

    Pattern matching is intentionally simple so notebook readers can predict
    outputs and reason about prompt sensitivity. Hooks for `temperature` and
    `seed` allow self-consistency demonstrations to produce varied (but
    reproducible) samples.
    """

    POSITIVE_WORDS = {"good", "great", "love", "loved", "excellent", "amazing", "fantastic", "wonderful", "awesome", "best"}
    NEGATIVE_WORDS = {"bad", "terrible", "hate", "hated", "awful", "worst", "horrible", "poor", "disappointing", "broken"}

    def __init__(self, model: str = "mock-llm-v1", temperature: float = 0.0, seed: int = 42):
        super().__init__(model=model, temperature=temperature)
        self._seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, prompt: str, temperature: Optional[float] = None, seed: Optional[int] = None, **kwargs: Any) -> LLMResponse:
        t = self.temperature if temperature is None else temperature
        s = self._seed if seed is None else seed
        text = self._route(prompt, temperature=t, seed=s)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=self._approx_tokens(prompt),
            completion_tokens=self._approx_tokens(text),
        )

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _route(self, prompt: str, temperature: float, seed: int) -> str:
        p = prompt.lower()

        # ReAct: if last non-empty line is "Thought:" we generate a step.
        if "thought:" in p and "action:" in p:
            return self._react_step(prompt, seed=seed)

        # Chain-of-thought trigger.
        if "let's think step by step" in p or "step-by-step" in p:
            return self._chain_of_thought(prompt, temperature=temperature, seed=seed)

        # Sentiment classification.
        if "sentiment" in p and ("positive" in p or "negative" in p):
            return self._classify_sentiment(prompt)

        # JSON / structured output requests.
        if "json" in p or "schema" in p:
            return self._json_extract(prompt)

        # Email extraction.
        if "email" in p:
            return self._extract_email(prompt)

        # Math word problems (numbers + operations / "how many").
        if any(k in p for k in ["how many", "total", "sum", "+", "plus"]):
            return self._math(prompt, temperature=temperature, seed=seed)

        # Summarization.
        if "summar" in p:
            return self._summarize(prompt)

        # QA fallback.
        return self._qa(prompt, temperature=temperature, seed=seed)

    # ------------------------------------------------------------------
    # Specialist sub-routines
    # ------------------------------------------------------------------

    def _classify_sentiment(self, prompt: str) -> str:
        # Use the snippet AFTER "text:" if present, else the whole prompt.
        m = re.search(r"text\s*:\s*(.+)", prompt, flags=re.IGNORECASE | re.DOTALL)
        snippet = (m.group(1) if m else prompt).lower()
        pos = sum(1 for w in self.POSITIVE_WORDS if w in snippet)
        neg = sum(1 for w in self.NEGATIVE_WORDS if w in snippet)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    def _extract_email(self, prompt: str) -> str:
        match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", prompt)
        return match.group(0) if match else "NONE"

    def _math(self, prompt: str, temperature: float, seed: int) -> str:
        nums = [int(x) for x in re.findall(r"\b\d+\b", prompt)]
        if not nums:
            return "I cannot find numbers."
        # Naive: sum the numbers — works for many "how many in total" prompts.
        total = sum(nums)
        if temperature > 0:
            rng = random.Random(seed)
            jitter = rng.choice([-1, 0, 0, 0, 1])
            total += jitter
        return f"The answer is {total}."

    def _chain_of_thought(self, prompt: str, temperature: float, seed: int) -> str:
        nums = [int(x) for x in re.findall(r"\b\d+\b", prompt)]
        if nums:
            steps = " + ".join(str(n) for n in nums)
            total = sum(nums)
            if temperature > 0:
                rng = random.Random(seed)
                total += rng.choice([-1, 0, 0, 0, 1])
            return (
                f"Step 1: Identify the numbers: {nums}.\n"
                f"Step 2: Add them: {steps} = {total}.\n"
                f"Answer: {total}"
            )
        return "Step 1: Restate the problem.\nStep 2: Apply the rule.\nAnswer: unknown"

    def _summarize(self, prompt: str) -> str:
        body = prompt.split(":", 1)[-1].strip()
        sentences = re.split(r"(?<=[.!?])\s+", body)
        return sentences[0][:160] if sentences else "(empty)"

    def _json_extract(self, prompt: str) -> str:
        # Look for an embedded JSON-ish object first.
        m = re.search(r"\{[^{}]*\}", prompt)
        if m:
            return m.group(0)
        # Otherwise, build a tiny canonical payload from the prompt text.
        snippet = prompt.strip().splitlines()[-1][:80]
        return '{"label": "neutral", "confidence": 0.5, "snippet": "' + snippet.replace('"', "'") + '"}'

    def _react_step(self, prompt: str, seed: int) -> str:
        # Count *actual* observations: lines like "Observation: <something>" with content
        # AFTER the colon. The template's format-description "Observation: <result>" is
        # an instruction placeholder and should not advance the loop.
        observations = re.findall(r"^Observation:\s*[A-Za-z0-9]", prompt, flags=re.MULTILINE)
        # Only consider observations in the scratchpad (after the literal "Question:" line).
        scratchpad_match = re.search(r"Question:.*", prompt, flags=re.DOTALL)
        scratchpad = scratchpad_match.group(0) if scratchpad_match else ""
        real_obs = re.findall(r"^Observation:\s*\S", scratchpad, flags=re.MULTILINE)
        nums_in_question = re.findall(r"\b\d+\b", scratchpad.split("\n", 1)[0]) if scratchpad else []
        nums_q = [int(x) for x in nums_in_question]
        if real_obs:
            # We already executed at least one tool call; finish using the latest observation.
            obs_values = re.findall(r"^Observation:\s*(.+)$", scratchpad, flags=re.MULTILINE)
            answer = obs_values[-1].strip() if obs_values else (str(sum(nums_q)) if nums_q else "0")
            return f" I now know the answer.\nAction: Finish[{answer}]"
        # First step: ask the calculator tool.
        if len(nums_q) >= 2:
            return f" I will add the numbers.\nAction: Calculator[{nums_q[0]} + {nums_q[1]}]"
        return " I should look this up.\nAction: Search[query]"

    def _qa(self, prompt: str, temperature: float, seed: int) -> str:
        # Echo a short, plausible answer derived from the prompt.
        words = re.findall(r"[A-Za-z]+", prompt)
        keyword = next((w for w in reversed(words) if len(w) > 4), "answer")
        if temperature > 0:
            rng = random.Random(seed)
            suffix = rng.choice(["", " (per the context)", " — based on the prompt", "."])
        else:
            suffix = "."
        return f"{keyword.capitalize()}{suffix}"

    @staticmethod
    def _approx_tokens(text: str) -> int:
        # ~4 chars per token heuristic.
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Echo client (debugging)
# ---------------------------------------------------------------------------


class EchoLLMClient(BaseLLMClient):
    """Echoes the prompt verbatim. Useful for inspecting rendered prompts."""

    def __init__(self, model: str = "echo-llm"):
        super().__init__(model=model, temperature=0.0)

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            text=prompt,
            model=self.model,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(prompt) // 4),
        )


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


class RetryClient(BaseLLMClient):
    """
    Wraps another client and retries with exponential backoff.

    Useful in production for transient errors. Retries are no-ops for the
    deterministic mock client, but the wiring is shown for completeness.
    """

    def __init__(
        self,
        inner: BaseLLMClient,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        retry_on: Callable[[Exception], bool] = lambda e: True,
    ):
        super().__init__(model=inner.model, temperature=inner.temperature)
        self.inner = inner
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.retry_on = retry_on

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.inner.complete(prompt, **kwargs)
            except Exception as e:  # pragma: no cover - exercised by tests only
                last_exc = e
                if not self.retry_on(e) or attempt == self.max_attempts:
                    raise
                wait = self.backoff_seconds * (2 ** (attempt - 1))
                logger.warning("LLM call failed (%s). Retrying in %.2fs.", e, wait)
                time.sleep(wait)
        # Unreachable, but mypy-friendly.
        raise RuntimeError("RetryClient: exhausted attempts") from last_exc


# ---------------------------------------------------------------------------
# Optional real-provider stubs (never called by chapter notebooks)
# ---------------------------------------------------------------------------


class OpenAIClient(BaseLLMClient):
    """Stub for OpenAI; raises NotImplementedError unless the SDK is wired in."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, api_key: Optional[str] = None):
        super().__init__(model=model, temperature=temperature)
        try:
            import openai  # noqa: F401
            self._sdk_available = True
        except ImportError:
            self._sdk_available = False
        self.api_key = api_key

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        if not self._sdk_available:
            raise NotImplementedError(
                "openai SDK not installed. Run `pip install openai` and supply an API key."
            )
        raise NotImplementedError(
            "OpenAIClient.complete is intentionally a stub in this chapter. "
            "Implement against your account if you want to use a real model."
        )


class AnthropicClient(BaseLLMClient):
    """Stub for Anthropic; raises NotImplementedError unless the SDK is wired in."""

    def __init__(self, model: str = "claude-3-5-sonnet-latest", temperature: float = 0.0, api_key: Optional[str] = None):
        super().__init__(model=model, temperature=temperature)
        try:
            import anthropic  # noqa: F401
            self._sdk_available = True
        except ImportError:
            self._sdk_available = False
        self.api_key = api_key

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        if not self._sdk_available:
            raise NotImplementedError(
                "anthropic SDK not installed. Run `pip install anthropic` and supply an API key."
            )
        raise NotImplementedError(
            "AnthropicClient.complete is intentionally a stub in this chapter. "
            "Implement against your account if you want to use a real model."
        )


__all__ = [
    "BaseLLMClient",
    "LLMResponse",
    "MockLLMClient",
    "EchoLLMClient",
    "RetryClient",
    "OpenAIClient",
    "AnthropicClient",
]

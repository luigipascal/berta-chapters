"""
Prompt template module for Chapter 12: Prompt Engineering & In-Context Learning.
Provides Jinja-style render templates for zero-shot, few-shot, chain-of-thought,
and ReAct prompts, plus a small in-memory registry of named prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_jinja():
    """Lazy import of jinja2 with a clear install hint."""
    try:
        import jinja2
        return jinja2
    except ImportError:  # pragma: no cover - surfaced at import time
        raise ImportError(
            "jinja2 is required. Install with: pip install jinja2>=3"
        ) from None


# ---------------------------------------------------------------------------
# Core template
# ---------------------------------------------------------------------------


@dataclass
class PromptTemplate:
    """
    A reusable Jinja template with a stable name and version.

    The template renders into either a single string (`render`) or a
    {role: str, content: str} list (`render_messages`) for chat-style APIs.
    """

    name: str
    template: str
    version: str = "v1"
    system: Optional[str] = None
    description: str = ""
    input_variables: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.input_variables:
            # Naive variable detection: look for {{ var }} occurrences.
            import re
            self.input_variables = sorted(
                set(re.findall(r"{{\s*(\w+)\s*}}", self.template))
            )

    def render(self, **kwargs: Any) -> str:
        """Render the template body with the provided variables."""
        jinja2 = _get_jinja()
        env = jinja2.Environment(
            undefined=jinja2.StrictUndefined,
            keep_trailing_newline=True,
        )
        try:
            return env.from_string(self.template).render(**kwargs)
        except jinja2.UndefinedError as e:
            missing = sorted(set(self.input_variables) - set(kwargs))
            raise ValueError(
                f"Missing variables for template '{self.name}': {missing}"
            ) from e

    def render_messages(self, **kwargs: Any) -> List[Dict[str, str]]:
        """Render as a list of chat messages (system + user)."""
        messages: List[Dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.render(**kwargs)})
        return messages

    def fingerprint(self) -> str:
        """Short hash uniquely identifying this prompt's text + version."""
        import hashlib
        h = hashlib.sha256(
            f"{self.name}:{self.version}:{self.system or ''}:{self.template}".encode("utf-8")
        )
        return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Few-shot
# ---------------------------------------------------------------------------


@dataclass
class FewShotExample:
    """A single (input, output) pair used as an in-context example."""

    input: str
    output: str
    label: Optional[str] = None


@dataclass
class FewShotTemplate(PromptTemplate):
    """
    Few-shot prompt: instruction + N labeled examples + the new input.

    The `template` field should reference `{{ examples }}` and `{{ input }}`.
    If empty, a default layout is used.
    """

    examples: List[FewShotExample] = field(default_factory=list)
    example_separator: str = "\n\n"

    def __post_init__(self) -> None:
        if not self.template:
            self.template = (
                "{{ instruction }}\n\n"
                "{% for ex in examples %}"
                "Input: {{ ex.input }}\nOutput: {{ ex.output }}{{ sep }}"
                "{% endfor %}"
                "Input: {{ input }}\nOutput:"
            )
        super().__post_init__()

    def render(self, input: str, instruction: str = "", **kwargs: Any) -> str:
        return super().render(
            input=input,
            instruction=instruction,
            examples=self.examples,
            sep=self.example_separator,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Chain of thought
# ---------------------------------------------------------------------------


@dataclass
class ChainOfThoughtTemplate(PromptTemplate):
    """
    CoT prompt: ask the model to reason step-by-step before answering.

    The default template emits a 'Let's think step by step.' suffix, which is
    the canonical zero-shot CoT trigger from Kojima et al. (2022).
    """

    cot_trigger: str = "Let's think step by step."

    def __post_init__(self) -> None:
        if not self.template:
            self.template = (
                "{{ instruction }}\n\n"
                "Question: {{ input }}\n"
                "{{ cot_trigger }}"
            )
        super().__post_init__()

    def render(self, input: str, instruction: str = "Answer the following.", **kwargs: Any) -> str:
        return super().render(
            input=input,
            instruction=instruction,
            cot_trigger=self.cot_trigger,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# ReAct
# ---------------------------------------------------------------------------


@dataclass
class ReActTemplate(PromptTemplate):
    """
    ReAct prompt: interleaves Thought / Action / Observation lines.

    The model emits Thought + Action; an external runtime executes the Action
    against a tool and appends an Observation. The loop continues until the
    model emits `Action: Finish[answer]`.
    """

    tools_description: str = ""

    def __post_init__(self) -> None:
        if not self.template:
            self.template = (
                "{{ instruction }}\n\n"
                "You may use the following tools:\n{{ tools_description }}\n\n"
                "Use this format strictly:\n"
                "Thought: <your reasoning>\n"
                "Action: <tool_name>[<input>]\n"
                "Observation: <result>\n"
                "... (repeat) ...\n"
                "Thought: I now know the answer.\n"
                "Action: Finish[<answer>]\n\n"
                "Question: {{ input }}\n"
                "{% if scratchpad %}{{ scratchpad }}\n{% endif %}"
                "Thought:"
            )
        super().__post_init__()

    def render(
        self,
        input: str,
        instruction: str = "Solve the question by reasoning and using tools.",
        scratchpad: str = "",
        **kwargs: Any,
    ) -> str:
        return super().render(
            input=input,
            instruction=instruction,
            tools_description=self.tools_description or "(no tools)",
            scratchpad=scratchpad,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PromptRegistry:
    """
    In-memory registry of named, versioned prompt templates.

    Supports `register`, `get` (by name and optional version), and `list`.
    Persistence helpers (`to_yaml` / `from_yaml`) round-trip via PyYAML.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, PromptTemplate]] = {}

    def register(self, template: PromptTemplate, overwrite: bool = False) -> None:
        bucket = self._store.setdefault(template.name, {})
        if template.version in bucket and not overwrite:
            raise ValueError(
                f"Prompt '{template.name}' version '{template.version}' already registered."
            )
        bucket[template.version] = template

    def get(self, name: str, version: Optional[str] = None) -> PromptTemplate:
        if name not in self._store:
            raise KeyError(f"Unknown prompt: '{name}'")
        bucket = self._store[name]
        if version is None:
            # Return the highest-sorted version (simple semantic-ish ordering).
            version = sorted(bucket.keys())[-1]
        if version not in bucket:
            raise KeyError(
                f"Prompt '{name}' has no version '{version}'. "
                f"Available: {sorted(bucket)}"
            )
        return bucket[version]

    def list(self) -> List[str]:
        return [
            f"{name}@{ver}"
            for name, vers in sorted(self._store.items())
            for ver in sorted(vers)
        ]

    def to_yaml(self, path: str) -> None:
        try:
            import yaml
        except ImportError as e:
            raise ImportError("pyyaml required: pip install pyyaml") from e
        payload = {
            name: {
                ver: {
                    "name": tmpl.name,
                    "version": tmpl.version,
                    "system": tmpl.system,
                    "description": tmpl.description,
                    "template": tmpl.template,
                }
                for ver, tmpl in vers.items()
            }
            for name, vers in self._store.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=True)

    @classmethod
    def from_yaml(cls, path: str) -> "PromptRegistry":
        try:
            import yaml
        except ImportError as e:
            raise ImportError("pyyaml required: pip install pyyaml") from e
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        reg = cls()
        for name, vers in payload.items():
            for ver, body in vers.items():
                reg.register(
                    PromptTemplate(
                        name=body["name"],
                        version=body["version"],
                        system=body.get("system"),
                        description=body.get("description", ""),
                        template=body["template"],
                    )
                )
        return reg


# ---------------------------------------------------------------------------
# Built-in named prompts (small starter set used by notebooks/exercises)
# ---------------------------------------------------------------------------


def default_registry() -> PromptRegistry:
    """Return a registry pre-populated with a handful of useful templates."""
    reg = PromptRegistry()
    reg.register(
        PromptTemplate(
            name="qa_zero_shot",
            template="Answer the question concisely.\n\nQuestion: {{ question }}\nAnswer:",
            system="You are a careful, concise assistant.",
            description="Zero-shot question answering.",
        )
    )
    reg.register(
        PromptTemplate(
            name="classify_sentiment",
            template=(
                "Classify the sentiment as 'positive', 'negative', or 'neutral'.\n"
                "Return only the label.\n\n"
                "Text: {{ text }}\nLabel:"
            ),
            system="You are a sentiment classifier.",
            description="Single-label sentiment classification.",
        )
    )
    reg.register(
        FewShotTemplate(
            name="extract_email",
            template="",
            system="You extract email addresses.",
            description="Few-shot email extraction.",
            examples=[
                FewShotExample("Contact me at foo@bar.com.", "foo@bar.com"),
                FewShotExample("No emails here.", "NONE"),
            ],
        )
    )
    reg.register(
        ChainOfThoughtTemplate(
            name="math_word_problem",
            template="",
            system="You solve grade-school math problems.",
            description="Chain-of-thought math.",
        )
    )
    return reg


__all__ = [
    "PromptTemplate",
    "FewShotExample",
    "FewShotTemplate",
    "ChainOfThoughtTemplate",
    "ReActTemplate",
    "PromptRegistry",
    "default_registry",
]

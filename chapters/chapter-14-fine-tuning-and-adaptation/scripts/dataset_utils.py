"""
Dataset utilities for Chapter 14: Fine-tuning & Adaptation Techniques.

Pure-Python / NumPy helpers for instruction-tuning data: formatting, train/val
splitting, token budgeting, packing, and a minimal in-memory dataset class that
works without Hugging Face `datasets` installed.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
DEFAULT_RESPONSE_TEMPLATE = "{output}"


def format_instruction(
    instruction: str,
    input: str = "",
    output: str = "",
    template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    response_template: str = DEFAULT_RESPONSE_TEMPLATE,
) -> Dict[str, str]:
    """
    Format an Alpaca-style instruction example.

    Returns a dict with `prompt`, `response`, and `text` (their concatenation).
    The `prompt` portion is what the model conditions on; the `response`
    portion is what gets supervised by the loss (everything else is masked).
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("`instruction` must be a non-empty string.")
    prompt = template.format(instruction=instruction.strip(), input=(input or "").strip())
    response = response_template.format(output=(output or "").strip())
    return {"prompt": prompt, "response": response, "text": prompt + response}


def load_jsonl(path: str | Path) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    rows: List[Dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {i} of {p}: {exc}") from exc
    return rows


def save_jsonl(rows: Iterable[Dict], path: str | Path) -> int:
    """Write an iterable of dicts to a JSONL file. Returns the count written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def train_val_split(
    rows: Sequence[Dict],
    train_fraction: float = 0.9,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Deterministic shuffle then split into train / val.

    Raises ValueError if train_fraction is not strictly in (0, 1).
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not rows:
        return [], []
    rng = random.Random(seed)
    indexed = list(rows)
    rng.shuffle(indexed)
    cut = max(1, int(len(indexed) * train_fraction))
    return indexed[:cut], indexed[cut:]


def whitespace_tokenize(text: str) -> List[str]:
    """A tiny stand-in tokenizer when `transformers` is unavailable."""
    return text.split()


def tokenize_budget(
    rows: Sequence[Dict],
    max_seq_len: int = 512,
    tokenizer: Optional[Callable[[str], List]] = None,
    text_key: str = "text",
) -> Dict[str, float]:
    """
    Estimate token usage and how many examples fit under `max_seq_len`.

    Returns a summary with mean / p95 / max token counts and a `fit_fraction`
    of examples that would not need truncation.
    """
    tok = tokenizer or whitespace_tokenize
    counts: List[int] = []
    for row in rows:
        text = row.get(text_key) or row.get("prompt", "") + row.get("response", "")
        counts.append(len(tok(text)))
    if not counts:
        return {"n": 0, "mean": 0.0, "p95": 0.0, "max": 0, "fit_fraction": 1.0}
    counts_sorted = sorted(counts)
    p95_idx = max(0, int(0.95 * (len(counts_sorted) - 1)))
    fit = sum(1 for c in counts if c <= max_seq_len) / len(counts)
    return {
        "n": len(counts),
        "mean": sum(counts) / len(counts),
        "p95": counts_sorted[p95_idx],
        "max": max(counts),
        "fit_fraction": fit,
    }


def pack_examples(
    rows: Sequence[Dict],
    max_seq_len: int = 512,
    tokenizer: Optional[Callable[[str], List]] = None,
    text_key: str = "text",
    separator: str = "\n\n",
) -> List[str]:
    """
    Concatenate short examples up to `max_seq_len` tokens to reduce padding.

    Greedy first-fit packing on whitespace-token counts. Returns the packed
    text strings; downstream tokenization should be done with the real
    tokenizer that will be used during training.
    """
    tok = tokenizer or whitespace_tokenize
    sep_len = len(tok(separator))
    packs: List[List[str]] = []
    pack_lens: List[int] = []
    for row in rows:
        text = row.get(text_key) or (row.get("prompt", "") + row.get("response", ""))
        n = len(tok(text))
        if n > max_seq_len:
            packs.append([text])  # oversized example becomes its own pack (truncate later)
            pack_lens.append(n)
            continue
        placed = False
        for i, length in enumerate(pack_lens):
            if length + sep_len + n <= max_seq_len:
                packs[i].append(text)
                pack_lens[i] = length + sep_len + n
                placed = True
                break
        if not placed:
            packs.append([text])
            pack_lens.append(n)
    return [separator.join(p) for p in packs]


def build_loss_mask(
    prompt_len: int,
    total_len: int,
    label_value: int = 1,
    ignore_value: int = 0,
) -> List[int]:
    """
    Mask of length `total_len` where prompt tokens are `ignore_value` and
    response tokens are `label_value`. Used by SFT loss to supervise only
    the response portion.
    """
    if prompt_len < 0 or total_len < prompt_len:
        raise ValueError("Require 0 <= prompt_len <= total_len.")
    return [ignore_value] * prompt_len + [label_value] * (total_len - prompt_len)


@dataclass
class InstructionDataset:
    """Minimal in-memory dataset that mimics the slice / len / iter API."""

    rows: List[Dict] = field(default_factory=list)
    template: str = DEFAULT_INSTRUCTION_TEMPLATE
    response_template: str = DEFAULT_RESPONSE_TEMPLATE

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs) -> "InstructionDataset":
        return cls(rows=load_jsonl(path), **kwargs)

    def formatted(self) -> List[Dict[str, str]]:
        return [
            format_instruction(
                r.get("instruction", ""),
                r.get("input", ""),
                r.get("output", ""),
                template=self.template,
                response_template=self.response_template,
            )
            for r in self.rows
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Dict]:
        return iter(self.formatted())

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.formatted()[idx]


if __name__ == "__main__":
    sample = [
        {"instruction": "Translate to French", "input": "hello", "output": "bonjour"},
        {"instruction": "Sum", "input": "2 and 3", "output": "5"},
    ]
    ds = InstructionDataset(rows=sample)
    assert len(ds) == 2
    formatted = ds.formatted()
    assert "Response" in formatted[0]["prompt"]
    train, val = train_val_split(sample, train_fraction=0.5, seed=0)
    assert len(train) + len(val) == len(sample)
    summary = tokenize_budget(formatted, max_seq_len=8)
    assert summary["n"] == 2
    mask = build_loss_mask(prompt_len=3, total_len=7)
    assert mask == [0, 0, 0, 1, 1, 1, 1]
    print("dataset_utils self-tests passed.")

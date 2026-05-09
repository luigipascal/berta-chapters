# Chapter 12: Prompt Engineering & In-Context Learning

**Track**: Practitioner | **Time**: 6 hours | **Prerequisites**: [Chapter 11: Large Language Models & Transformers](../chapter-11-large-language-models-and-transformers/)

---

Prompt engineering is the practice of designing inputs that get reliable, useful behavior from large language models. **In-context learning** is the surprising ability of modern LLMs to learn a new task from a handful of examples placed in the prompt — no gradient updates required. Together they form the primary interface for working with LLMs in production.

This chapter takes you from prompt anatomy and zero/few-shot patterns through chain-of-thought, ReAct, structured outputs, and prompt-injection defenses, ending with a full evaluation harness and a versioned prompt registry. All notebooks run **offline** with a deterministic mock LLM client, so you can develop and test prompt systems without API keys, network, or cost.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Decompose a prompt** — separate instruction, context, input, and output spec
2. **Apply zero-shot, few-shot, and in-context learning** — choose the right pattern per task
3. **Use chain-of-thought and self-consistency** — improve reasoning quality with structured prompts
4. **Design ReAct and tool-use prompts** — combine reasoning with function/tool calls
5. **Produce structured outputs** — JSON schemas with Pydantic, parsing, and validation
6. **Evaluate prompts systematically** — golden datasets, graders, A/B tests, statistical CIs
7. **Defend against prompt injection** — detect, sandwich, hierarchy, and output filtering
8. **Ship a prompt to production** — versioning, registry, caching, fallbacks, and observability

---

## Prerequisites

- **Chapter 11: Large Language Models & Transformers** — tokenization, transformer basics, sampling, instruction tuning
- **Chapter 10: NLP Basics** — text preprocessing, vectorization, similarity
- Python fundamentals, JSON, regular expressions
- Comfort reading and writing small classes and functions

---

## What You'll Build

- **Prompt template library** — reusable Jinja-style templates for zero-shot, few-shot, CoT, and ReAct patterns
- **Evaluation harness** — golden datasets, exact/regex/embedding graders, A/B tester with bootstrap CIs
- **Prompt-injection defense kit** — allowlist filters, sandwich/hierarchy guards, output validators
- **Structured-output parser** — Pydantic-validated JSON extraction with safe fallback
- **Versioned prompt registry** — file-based registry with named, dated prompt revisions

---

## Time Commitment

| Section | Time |
|---------|------|
| Notebook 01: Prompt Basics (anatomy, zero/few-shot, structured outputs) | 1.5–2 hours |
| Notebook 02: Advanced Prompting (CoT, self-consistency, ReAct, tool use) | 1.5–2 hours |
| Notebook 03: Prompt Systems (eval, A/B, injection defense, production) | 1.5–2 hours |
| Exercises (Problem Sets 1 & 2) | 0.5–1 hour |
| **Total** | **6 hours** |

---

## Technology Stack

- **Templating & schemas**: `jinja2`, `pydantic>=2`
- **Data & ML**: `numpy`, `pandas`, `scikit-learn` (TF-IDF for embedding-style match)
- **Notebooks**: `jupyter`, `ipywidgets`
- **Utilities**: `pyyaml`, `tqdm`
- **Optional (commented out)**: `openai`, `anthropic`, `transformers` — chapter runs fully offline with the bundled mock LLM client

---

## Quick Start

1. **Clone and enter the chapter**
   ```bash
   cd chapters/chapter-12-prompt-engineering-and-in-context-learning
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Run the notebooks**
   ```bash
   jupyter notebook notebooks/
   ```
   Start with `01_prompt_basics.ipynb`, then `02_advanced_prompting.ipynb`, then `03_prompt_systems.ipynb`.

---

## Notebook Guide

| Notebook | Focus |
|----------|--------|
| **01_prompt_basics.ipynb** | Prompt anatomy, zero/few-shot, in-context learning, system vs user, structured outputs with Pydantic, sensitivity to wording |
| **02_advanced_prompting.ipynb** | Chain-of-thought, self-consistency, ReAct, tool/function calling, JSON-mode parsing, retrieval cues, prompt patterns and limits |
| **03_prompt_systems.ipynb** | Evaluation (golden sets, graders, LLM-as-judge), A/B testing with CIs, versioning + registry, injection defenses, production observability, capstone |

---

## Exercise Guide

- **Problem Set 1** (`exercises/problem_set_1.ipynb`) — rewrite a vague prompt, build few-shot examples, design a structured-output schema, classify a tricky example, count tokens, parse JSON safely
- **Problem Set 2** (`exercises/problem_set_2.ipynb`) — implement self-consistency, build an eval harness, detect prompt injection, A/B test two prompts, design a ReAct loop for math word problems, build a versioned prompt registry
- **Solutions** — in `exercises/solutions/` with runnable code, explanations, and alternatives

---

## How to Run Locally

- Use Python 3.9+ and the versions in `requirements.txt` for reproducibility.
- Notebooks import from `scripts/` via `sys.path` and assume the chapter root as the working directory.
- All LLM calls in this chapter use the bundled `MockLLMClient` (deterministic, rule-based). To wire up a real provider, install the optional SDK (`openai` or `anthropic`) and swap the client; the abstract `BaseLLMClient` interface keeps the rest of the code unchanged.
- Datasets live in `datasets/`; prompt registry artifacts are written under `registry/` (created on demand).

---

## Common Troubleshooting

- **Pydantic v1 vs v2** — This chapter requires `pydantic>=2`. Upgrade with `pip install -U "pydantic>=2"`.
- **`jinja2` not found** — `pip install jinja2>=3`.
- **Optional SDKs missing** — `openai`, `anthropic`, and `transformers` are intentionally optional. Notebooks fall back to `MockLLMClient` automatically.
- **JSON parse errors in structured output** — The parser includes a fallback that extracts the first JSON-looking block; check the exercise on safe parsing.
- **Notebook can't find `scripts/`** — Run Jupyter from the chapter root, or adjust `sys.path.insert(...)` in cell 1.

---

## Next Steps

- **Chapter 13: Retrieval-Augmented Generation (RAG)** — Builds directly on the prompting patterns here: structured prompts, evaluation harnesses, and registry-managed templates plug into a retrieval pipeline so LLMs can ground their answers in your own documents.

---

**Generated by Berta AI**

Part of [Berta Chapters](https://github.com/your-org/berta-chapters) — open-source AI curriculum.  
*March 2026 — Berta Chapters*

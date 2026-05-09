# Chapter 14: Fine-tuning & Adaptation Techniques

**Track**: Practitioner | **Time**: 8 hours | **Prerequisites**: [Chapters 1–13](../) (especially [Chapter 11: LLMs](../chapter-11-large-language-models-and-transformers/) and [Chapter 13: RAG](../chapter-13-retrieval-augmented-generation/))

---

Fine-tuning teaches a pre-trained model new behaviors using your data. Where prompting and retrieval-augmented generation (RAG) shape outputs at inference time, fine-tuning updates the model's weights so it permanently absorbs your domain, style, and task structure. This chapter shows when to reach for fine-tuning, how to do it efficiently with parameter-efficient methods (LoRA, QLoRA, adapters, prefix tuning, IA³), and how to measure whether the result is actually better.

You will format instruction datasets, build a tiny supervised fine-tuning (SFT) loop, implement a LoRA adapter from scratch in NumPy, run a Direct Preference Optimization (DPO) loss demo, and design an evaluation harness that catches catastrophic forgetting and safety regressions. Heavy frameworks (`transformers`, `peft`, `trl`, `bitsandbytes`) are sketched with try/except so the chapter runs on a CPU laptop, while still teaching the production workflow you'll deploy in Chapter 15 (MLOps).

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Decide when to fine-tune** — vs. prompt engineering and RAG, with cost/latency/quality trade-offs
2. **Prepare instruction datasets** — formatting, splits, tokenization budgets, response masking
3. **Run a supervised fine-tuning (SFT) loop** — loss masking, learning-rate schedules, early stopping
4. **Implement LoRA from scratch** — low-rank adapters, scaling, merging, parameter-efficiency math
5. **Apply PEFT methods at scale** — QLoRA, adapters, prefix tuning, IA³, multi-adapter serving
6. **Use preference data** — RLHF and DPO concepts, with a NumPy DPO loss implementation
7. **Evaluate adapted models rigorously** — held-out tasks, win rates, LLM-as-judge caveats, regression checks
8. **Plan deployment** — model registry, versioning, hand-off to MLOps in Chapter 15

---

## Prerequisites

- **Chapter 11: Large Language Models & Transformers** — tokenization, transformer blocks, pre-training vs. fine-tuning
- **Chapter 13: Retrieval-Augmented Generation** — when retrieval suffices vs. when you need new weights
- Comfort with NumPy, scikit-learn, and basic gradient descent (Chapters 6–9)
- Familiarity with notebooks and command-line Python

---

## What You'll Build

- **SFT pipeline** — instruction formatting, train/val splits, masked-loss training loop on a small linear model
- **LoRA implementation** — NumPy adapter with rank, alpha, scaling; merge and serve helpers
- **Evaluation harness** — exact match, F1, held-out win-rate stub, drift / forgetting checks
- **Model registry stub** — versioned entries with hyperparams, eval scores, and adapter pointers ready for Chapter 15

---

## Time Commitment

| Section | Time |
|---------|------|
| Notebook 01: Fine-tuning Basics (when to FT, datasets, SFT loop, eval) | 2 hours |
| Notebook 02: PEFT & LoRA (LoRA math, NumPy adapter, QLoRA, adapter merging) | 2.5 hours |
| Notebook 03: Advanced Adaptation (instruction tuning, DPO, eval, deployment) | 2 hours |
| Exercises (Problem Sets 1 & 2) | 1.5 hours |
| **Total** | **8 hours** |

---

## Technology Stack

- **Core**: `numpy`, `pandas`, `scikit-learn` — hands-on math and SFT analog
- **Visualization**: `matplotlib` — loss curves, parameter-count comparisons
- **Notebooks**: `jupyter`, `ipywidgets`
- **Config / data**: `pyyaml`, `tqdm`
- **Optional (heavy, GPU helpful)**: `torch`, `transformers`, `peft`, `accelerate`, `datasets`, `trl`, `bitsandbytes`

---

## Quick Start

1. **Enter the chapter**
   ```bash
   cd chapters/chapter-14-fine-tuning-and-adaptation
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

   Optional heavy dependencies (only if you have a GPU and want the framework demos to actually run):
   ```bash
   pip install torch transformers peft accelerate datasets trl bitsandbytes
   ```

3. **Run the notebooks**
   ```bash
   jupyter notebook notebooks/
   ```
   Start with `01_fine_tuning_basics.ipynb`, then `02_peft_lora.ipynb`, then `03_advanced_adaptation.ipynb`.

---

## Notebook Guide

| Notebook | Focus |
|----------|--------|
| **01_fine_tuning_basics.ipynb** | Decision tree (prompt / RAG / FT), instruction dataset prep, SFT concepts, sklearn-analog SFT loop, evaluation basics |
| **02_peft_lora.ipynb** | Full FT vs PEFT trade-offs, LoRA math and NumPy implementation, QLoRA conceptual, adapters / prefix / IA³, merging and multi-adapter serving |
| **03_advanced_adaptation.ipynb** | Instruction tuning datasets (Alpaca format), RLHF and DPO (NumPy DPO loss), evaluation, catastrophic forgetting, registry / versioning, capstone design |

---

## Exercise Guide

- **Problem Set 1** (`exercises/problem_set_1.ipynb`) — format an instruction dataset, compute token budgets, write loss masking, choose hyperparameters, decide FT vs RAG, run a tiny SFT loop
- **Problem Set 2** (`exercises/problem_set_2.ipynb`) — implement LoRA forward, compute parameter-efficiency ratios, merge adapters, DPO loss in NumPy, held-out win-rate evaluation, design a registry entry
- **Solutions** — in `exercises/solutions/` with runnable notebooks and a CI-friendly `solutions.py`

---

## How to Run Locally

- Use Python 3.9+ and the versions in `requirements.txt` for reproducibility.
- All hands-on code is CPU-friendly: NumPy, pandas, and scikit-learn carry the load.
- Heavy framework cells (`transformers`, `peft`, `trl`) are wrapped in `try/except` and print a hint instead of failing if the package is missing.
- Scripts in `scripts/` can be imported from notebooks; notebooks add `../scripts` to `sys.path` like in Chapter 10.
- Datasets live in `datasets/` (small JSONL files) and are loaded relative to the chapter root.

---

## Common Troubleshooting

- **`transformers` / `peft` not installed** — Optional. Install with `pip install transformers peft accelerate trl`. Without it, the framework sketch cells print the workflow instead of running it.
- **Out-of-memory during SFT** — Reduce batch size, sequence length, or use gradient accumulation; for full models try QLoRA (4-bit base).
- **Loss not decreasing** — Check your loss mask (you should mask the prompt tokens, not the response), verify learning rate, and confirm targets are shifted by one.
- **Eval scores collapse on general benchmarks after fine-tuning** — Catastrophic forgetting; mix in some general data, lower the learning rate, or use a smaller LoRA rank.
- **Adapter merge changes outputs** — Verify alpha / scaling and that you merge `B @ A * (alpha / r)` into the base weights.

---

## Next Steps

- **Chapter 15: MLOps for AI Systems** — Picks up the model registry stub from this chapter and turns it into a real deployment pipeline: CI for models, versioning, monitoring, rollback, and serving infrastructure for fine-tuned models and PEFT adapters.

---

**Generated by Berta AI**

Part of [Berta Chapters](https://github.com/your-org/berta-chapters) — open-source AI curriculum.  
*May 2026 — Berta Chapters*

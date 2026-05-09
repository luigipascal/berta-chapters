# Chapter 11: Large Language Models & Transformers

**Track**: Practitioner | **Time**: 10 hours | **Prerequisites**: [Chapter 10: Natural Language Processing Basics](../chapter-10-natural-language-processing-basics/)

---

Large language models (LLMs) and the **Transformer** architecture power most of modern AI: ChatGPT, Claude, Gemini, Llama, and the embedding/RAG systems built on top of them. This chapter takes the attention and transfer-learning ideas from Chapter 10 and builds them up into a full understanding of how transformers work, how pretrained LLMs are used, and how to build real applications around them.

You will implement **scaled dot-product attention**, **multi-head attention**, **positional encodings**, and a **transformer block** in pure NumPy; work with **pretrained models** (BERT, DistilBERT, GPT-style) through a graceful Hugging Face fallback; generate embeddings; explore **decoding strategies** (greedy, top-k, top-p, temperature); and study **scaling laws**, **evaluation**, and how to ship LLM-powered features.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Explain the Transformer architecture** — self-attention, multi-head attention, positional encoding, residuals, layer norm
2. **Implement attention from scratch** — scaled dot-product and multi-head attention in NumPy
3. **Distinguish encoder, decoder, and encoder–decoder models** — and pick the right family for a task
4. **Use pretrained LLMs** — tokenize, extract embeddings, run inference with Hugging Face `transformers`
5. **Apply LLM embeddings to downstream tasks** — similarity search and frozen-embedding classifiers
6. **Generate text with controlled decoding** — greedy, sampling, temperature, top-k, top-p, repetition penalty
7. **Evaluate LLMs** — perplexity, BLEU/ROUGE, win-rate, and the limits of LLM-as-judge
8. **Design LLM-powered systems** — chunking, streaming, function calling, and the road to RAG and fine-tuning

---

## Prerequisites

- **Chapter 10: Natural Language Processing Basics** — tokenization, embeddings, attention intuition, transfer learning
- **Chapter 9: Deep Learning Fundamentals** — backprop, layers, optimizers, training loops
- Comfort with NumPy, linear algebra (matmul, softmax), and basic probability
- Optional: PyTorch for the deeper sections (the chapter runs without it)

---

## What You'll Build

- **Mini-Transformer in NumPy** — scaled dot-product attention, multi-head attention, positional encoding, and a single encoder block you can run end-to-end
- **Embedding service** — wrap a pretrained model (or fallback) to turn text into vectors and search by similarity
- **Frozen-embedding classifier** — sentence embeddings + scikit-learn for a fast, strong text classifier
- **Decoding playground** — greedy, temperature, top-k and top-p samplers operating on real logit distributions
- **LLM application sketch** — chunking, prompt assembly, and streaming patterns that lead into Chapter 12 (Prompt Engineering) and Chapter 13 (RAG)

---

## Time Commitment

| Section | Time |
|---------|------|
| Notebook 01: Transformer Architecture (attention, multi-head, positional encoding, blocks) | 3 hours |
| Notebook 02: Pretrained LLMs (tokenizers, embeddings, classification, model selection) | 3 hours |
| Notebook 03: Advanced LLMs (decoding, KV cache, scaling, evaluation, apps) | 2.5 hours |
| Exercises (Problem Sets 1 & 2) | 1.5 hours |
| **Total** | **10 hours** |

---

## Technology Stack

- **Numerics**: `numpy`, `pandas`, `scikit-learn`
- **Visualization**: `matplotlib`
- **Notebooks**: `jupyter`, `ipywidgets`
- **Optional (LLMs)**: `transformers`, `tokenizers`, `accelerate`, `datasets`, `sentencepiece`, `huggingface-hub`
- **Optional (DL)**: `torch` for the deeper transformer/embedding sections

---

## Quick Start

1. **Clone and enter the chapter**
   ```bash
   cd chapters/chapter-11-large-language-models-and-transformers
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   # Optional, for the pretrained-LLM sections:
   # pip install torch transformers tokenizers accelerate datasets sentencepiece huggingface-hub
   ```

3. **Run the notebooks**
   ```bash
   jupyter notebook notebooks/
   ```
   Start with `01_transformer_architecture.ipynb`, then `02_pretrained_llms.ipynb`, then `03_advanced_llms.ipynb`.

---

## Notebook Guide

| Notebook | Focus |
|----------|--------|
| **01_transformer_architecture.ipynb** | From RNN limits to attention; scaled dot-product and multi-head attention in NumPy; sinusoidal positional encoding; encoder block; encoder/decoder/decoder-only families; tokenization (BPE/WordPiece) intuition |
| **02_pretrained_llms.ipynb** | Loading pretrained models with `transformers` (with fallback); `AutoTokenizer`; extracting and visualizing embeddings; mean pooling for sentence vectors; frozen-embedding classification; choosing BERT vs RoBERTa vs DistilBERT vs GPT |
| **03_advanced_llms.ipynb** | Decoding strategies (greedy, sampling, temperature, top-k, top-p); KV cache shapes; scaling laws; evaluation (perplexity, BLEU/ROUGE, LLM-as-judge); building LLM apps (chunking, streaming, function calling); capstone design |

---

## Exercise Guide

- **Problem Set 1** (`exercises/problem_set_1.ipynb`) — implement scaled dot-product attention; build sinusoidal positional encoding; plot an attention heatmap; tokenize text and reason about BPE; multi-head attention shape check; compare encoder/decoder/encoder–decoder
- **Problem Set 2** (`exercises/problem_set_2.ipynb`) — implement top-k sampling; build a tiny transformer block from scratch; compute perplexity; train an embedding-based classifier; reason about prompt vs context-window trade-offs; evaluate generations
- **Solutions** — in `exercises/solutions/` with runnable code, explanations, and alternatives

---

## How to Run Locally

- Use Python 3.9+ and the versions in `requirements.txt` for reproducibility.
- The numpy-only sections (Notebook 01, large parts of 03, all Problem Set 1) require **no** transformer installs.
- For Notebook 02 and the embedding sections, install the optional `transformers` / `torch` extras shown above.
- Scripts in `scripts/` can be run from the chapter root; notebooks assume that root as working directory.

---

## Common Troubleshooting

- **`transformers` not installed** — Notebooks fall back to NumPy/sklearn stubs and print a `pip install transformers` hint; install when you want the real models
- **Hugging Face download blocked / offline** — Set `HF_HUB_OFFLINE=1` and use a locally cached model, or rely on the fallback paths in the notebooks
- **Out-of-memory loading a large model** — Switch `MODEL_NAME` in `scripts/config.py` to `distilbert-base-uncased` or `sentence-transformers/all-MiniLM-L6-v2`
- **CUDA/GPU** — Optional; everything runs on CPU. Set `CUDA_VISIBLE_DEVICES=""` to force CPU if a GPU is misbehaving
- **Slow first run** — Pretrained model download can take a few minutes; subsequent runs hit the local cache

---

## Next Steps

- **Chapter 12: Prompt Engineering** — Now that you understand how LLMs tokenize, attend, and decode, Chapter 12 turns to *steering* them: prompt patterns, few-shot, chain-of-thought, structured output, and evaluation of prompts.

---

**Generated by Berta AI**

Part of [Berta Chapters](https://github.com/your-org/berta-chapters) — open-source AI curriculum.  
*March 2026 — Berta Chapters*

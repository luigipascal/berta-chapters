# Chapter 13: Retrieval-Augmented Generation (RAG)

**Track**: Practitioner | **Time**: 8 hours | **Prerequisites**: [Chapter 11: Large Language Models](../chapter-11-large-language-models/) and [Chapter 12: Prompt Engineering](../chapter-12-prompt-engineering/)

---

Retrieval-Augmented Generation (RAG) makes large language models practical for the real world: it grounds them in your private data, keeps them up to date, and reduces hallucination by injecting *retrieved* evidence into the prompt at query time. This chapter is the bridge between raw LLM knowledge (Chapter 11) and the production systems that ship to users.

You will build a RAG system end-to-end — chunking documents, computing embeddings, indexing them in a vector store, retrieving relevant context, assembling prompts, and evaluating answer quality. Everything runs offline with mocked LLMs and pure-Python/numpy backends so you can experiment without API keys, then plug in real models (OpenAI, Anthropic, Sentence-Transformers, FAISS, Chroma) when you're ready.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Explain the motivation for RAG** — hallucination, recency, private data, and context-window limits
2. **Implement vector similarity from scratch** — cosine similarity, top-k retrieval, in-memory indices
3. **Choose a chunking strategy** — fixed, sliding-window, sentence, and semantic chunking trade-offs
4. **Use embeddings effectively** — embedding models, dimensionality, normalization, and TF-IDF fallbacks
5. **Build hybrid search** — combine dense (vector) and sparse (BM25) retrieval with reciprocal rank fusion
6. **Apply reranking and query rewriting** — cross-encoders, HyDE, multi-query expansion
7. **Evaluate a RAG system** — hit@k, MRR, faithfulness, answer relevance, context precision/recall
8. **Design for production** — latency, caching, freshness, sharding, cost, and monitoring

---

## Prerequisites

- **Chapter 11: Large Language Models & Transformers** — token embeddings, prompts, in-context learning
- **Chapter 12: Prompt Engineering** — system/user messages, few-shot patterns, structured outputs
- Python fundamentals, comfort with NumPy, pandas, and scikit-learn (Chapters 1–6)

---

## What You'll Build

- **In-memory vector store from scratch** — `add`, `search`, `save`, `load`, all in NumPy
- **End-to-end RAG pipeline** — load → chunk → embed → index → retrieve → prompt → generate → cite
- **Hybrid retriever** — dense + BM25 with reciprocal rank fusion and an optional reranker
- **RAG evaluation harness** — hit@k, MRR, faithfulness, answer relevance, latency

---

## Time Commitment

| Section | Time |
|---------|------|
| Notebook 01: RAG Fundamentals (motivation, embeddings, naive retrieval, first end-to-end) | 2.5 hours |
| Notebook 02: RAG Pipeline (chunking, embeddings, vector stores, reranking, citations) | 2.5 hours |
| Notebook 03: Advanced RAG (hybrid search, query rewriting, evaluation, production, capstone) | 2 hours |
| Exercises (Problem Sets 1 & 2) | 1 hour |
| **Total** | **8 hours** |

---

## Technology Stack

- **Core**: `numpy`, `pandas`, `scikit-learn` for embeddings (TF-IDF), math, and metrics
- **Sparse retrieval**: `rank-bm25` for BM25, `nltk` for tokenization
- **Notebooks**: `jupyter`, `ipywidgets`, `matplotlib`
- **Optional dense embeddings**: `sentence-transformers` (auto-fallback to TF-IDF if missing)
- **Optional vector stores**: `faiss-cpu`, `chromadb` (in-memory NumPy index used by default)
- **Optional LLMs**: `openai`, `anthropic`, `tiktoken` (a `MockLLM` is used by default — no API keys required)

---

## Quick Start

1. **Clone and enter the chapter**
   ```bash
   cd chapters/chapter-13-retrieval-augmented-generation
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('punkt')"
   ```

3. **Run the notebooks**
   ```bash
   jupyter notebook notebooks/
   ```
   Start with `01_rag_fundamentals.ipynb`, then `02_rag_pipeline.ipynb`, then `03_advanced_rag.ipynb`.

---

## Notebook Guide

| Notebook | Focus |
|----------|--------|
| **01_rag_fundamentals.ipynb** | Why RAG, embeddings recap, cosine similarity, in-memory vector store from scratch, naive retrieval, first end-to-end RAG with a mock LLM, hit@k / MRR / precision@k |
| **02_rag_pipeline.ipynb** | Chunking strategies (fixed / sliding / sentence / semantic), embedding model choices with TF-IDF fallback, vector store options (FAISS / Chroma sketches), full pipeline, reranking, prompt assembly with citations |
| **03_advanced_rag.ipynb** | Hybrid search (dense + BM25 with RRF), query rewriting / HyDE / multi-query, faithfulness and answer-relevance metrics, agentic / multi-hop intuition, production concerns (latency, caching, freshness, sharding, cost), capstone design |

---

## Exercise Guide

- **Problem Set 1** (`exercises/problem_set_1.ipynb`) — cosine similarity from scratch, build a chunker, encode + retrieve, top-k accuracy, compare chunk sizes, source-citing prompt template
- **Problem Set 2** (`exercises/problem_set_2.ipynb`) — BM25 + dense hybrid, query rewriting, faithfulness scorer, multi-hop retrieval simulation, RAG evaluation harness, latency profiling
- **Solutions** — in `exercises/solutions/` with runnable code, explanations, and alternatives

---

## How to Run Locally

- Use Python 3.9+ and the versions in `requirements.txt` for reproducibility.
- The notebooks default to **offline mode** with TF-IDF embeddings and a `MockLLM` so they run without API keys, FAISS, or sentence-transformers.
- Optional installs (`faiss-cpu`, `sentence-transformers`, `chromadb`, `openai`, `anthropic`) are wrapped in `try/except` and fall back gracefully.
- Scripts in `scripts/` can be run from the chapter root; notebooks add `scripts/` to `sys.path` so imports work from `notebooks/`.

---

## Common Troubleshooting

- **`sentence-transformers` not installed** — Notebooks fall back to TF-IDF embeddings automatically. Install with `pip install sentence-transformers` for higher-quality vectors.
- **`faiss` import error** — The default `InMemoryVectorStore` uses NumPy and works everywhere. Install `faiss-cpu` only if you need scale.
- **`rank-bm25` missing** — Install with `pip install rank-bm25`. The hybrid retriever requires it.
- **NLTK punkt missing** — Run `python -c "import nltk; nltk.download('punkt')"`.
- **No API keys** — All notebooks use `MockLLM` by default. To use a real LLM, set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and swap the client in `RAGPipeline`.

---

## Next Steps

- **Chapter 14: Fine-tuning & Adaptation** — When retrieval isn't enough, fine-tune. Chapter 14 builds on the data preparation, evaluation, and prompt patterns you learned here to adapt models to your domain.

---

**Generated by Berta AI**

Part of [Berta Chapters](https://github.com/your-org/berta-chapters) — open-source AI curriculum.  
*May 2026 — Berta Chapters*

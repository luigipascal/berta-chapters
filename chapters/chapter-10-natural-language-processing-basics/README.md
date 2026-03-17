# Chapter 10: Natural Language Processing Basics

**Track**: Practitioner | **Time**: 8–10 hours | **Prerequisites**: [Chapter 9: Deep Learning Fundamentals](../chapter-09-deep-learning-fundamentals/)

---

Natural language processing (NLP) is how machines read, understand, and generate human language. This chapter bridges the deep learning fundamentals you learned in Chapter 9 with the text-based models and applications that power modern AI—from sentiment analysis and named entity recognition to the foundations that lead into large language models in Chapter 11.

You will work with real text: tokenizing, cleaning, and representing it with classic methods (TF-IDF, word2vec, GloVe) and with neural models (embeddings, LSTMs). By the end you will have built text classification pipelines, NER systems, and sentiment analyzers you can deploy.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Understand text representation** — tokenization, vectorization, and word embeddings
2. **Master classic NLP techniques** — TF-IDF, word2vec, GloVe, and when to use each
3. **Build text classification models** — from logistic regression to LSTMs
4. **Implement named entity recognition** — with spaCy and custom taggers
5. **Create sentiment analysis pipelines** — preprocessing → features → model → evaluation
6. **Understand sequence models for NLP** — RNNs, LSTMs, and their role in text
7. **Know when to use which technique** — trade-offs between traditional and neural approaches
8. **Deploy simple NLP models** — serialization, inference, and production considerations

---

## Prerequisites

- **Chapter 9: Deep Learning Fundamentals** — neural networks, backpropagation, layers, PyTorch or TensorFlow basics
- Python fundamentals, data structures, basic ML (Chapters 1–6 or equivalent)
- Comfort with NumPy, pandas, and scikit-learn

---

## What You'll Build

- **Sentiment analyzer** — preprocess text, extract TF-IDF features, train a classifier, evaluate and test on new examples
- **Multi-class text classifier** — news/document categorization with embeddings and LSTMs
- **Named entity recognition pipeline** — extract persons, organizations, locations with spaCy and analyze results
- **End-to-end NLP system** — combine classification, NER, and sentiment in one pipeline

---

## Time Commitment

| Section | Time |
|---------|------|
| Notebook 01: NLP Fundamentals (tokenization, TF-IDF, embeddings, sentiment intro) | 2.5–3 hours |
| Notebook 02: NLP Classification (deep learning, NER, clustering, pipelines) | 2.5–3 hours |
| Notebook 03: NLP Advanced (attention, seq2seq, transfer learning, production) | 2.5–3 hours |
| Exercises (Problem Sets 1 & 2) | 1–2 hours |
| **Total** | **8–10 hours** |

---

## Technology Stack

- **NLP & text**: `nltk`, `spacy`, `gensim`
- **ML & vectors**: `scikit-learn`, `numpy`, `pandas`
- **Deep learning**: `tensorflow` or `torch`
- **Notebooks**: `jupyter`, `ipywidgets`
- **Optional**: `transformers`, `datasets`, `huggingface-hub` for transfer learning

---

## Quick Start

1. **Clone and enter the chapter**
   ```bash
   cd chapters/chapter-10-natural-language-processing-basics
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
   ```

3. **Run the notebooks**
   ```bash
   jupyter notebook notebooks/
   ```
   Start with `01_nlp_fundamentals.ipynb`, then `02_nlp_classification.ipynb`, then `03_nlp_advanced.ipynb`.

---

## Notebook Guide

| Notebook | Focus |
|----------|--------|
| **01_nlp_fundamentals.ipynb** | Tokenization, stemming/lemmatization, Bag of Words, TF-IDF, word embeddings (GloVe), first sentiment analysis with logistic regression |
| **02_nlp_classification.ipynb** | Deep learning for text (CNN/LSTM), multi-class classification, NER with spaCy, text similarity and clustering, full pipeline and common pitfalls |
| **03_nlp_advanced.ipynb** | Attention, seq2seq, transfer learning (e.g. BERT), multi-task NLP system, language generation basics, production considerations, capstone design |

---

## Exercise Guide

- **Problem Set 1** (`exercises/problem_set_1.ipynb`) — tokenization, text cleaning, TF-IDF from scratch, word similarity/analogies, simple sentiment, vocabulary building, representation comparison
- **Problem Set 2** (`exercises/problem_set_2.ipynb`) — LSTM classification, NER, document clustering, sentiment with deep learning, multi-task learning, similarity search, BERT fine-tuning preview
- **Solutions** — in `exercises/solutions/` with runnable code, explanations, and alternatives

---

## How to Run Locally

- Use Python 3.9+ and the versions in `requirements.txt` for reproducibility.
- After installing dependencies, download the spaCy model: `python -m spacy download en_core_web_sm`
- Download NLTK data as in Quick Start (punkt, stopwords, wordnet, averaged_perceptron_tagger).
- Scripts in `scripts/` can be run from the chapter root; notebooks assume that root as working directory or adjust paths to `data/` and `models/` as in `config.py`.

---

## Common Troubleshooting

- **spacy model not found** — Run `python -m spacy download en_core_web_sm`
- **NLTK data missing** — Run the `nltk.download(...)` commands from Quick Start in a Python shell or notebook
- **Out-of-memory with large embeddings** — Use a smaller GloVe file (e.g. 50d or 100d) or load a subset of vocabulary
- **CUDA/GPU** — Optional; notebooks run on CPU; set `CUDA_VISIBLE_DEVICES` or framework-specific device settings if needed

---

## Next Steps

- **Chapter 11: Large Language Models & Transformers** — Builds directly on the NLP concepts and representations you learned here: attention, sequence modeling, and transfer learning lead into transformers and LLMs.

---

**Generated by Berta AI**

Part of [Berta Chapters](https://github.com/your-org/berta-chapters) — open-source AI curriculum.  
*March 2026 — Berta Chapters*

# Chapter 15: MLOps & Model Deployment

**Track**: Practitioner | **Time**: 8 hours | **Prerequisites**: [Chapters 1–14](../)

---

MLOps is the discipline that takes a trained model from a notebook on your laptop to a reliable, observable, and continuously improving service in production. This chapter ties together everything from the Practitioner track—classical ML, deep learning, NLP, LLMs, RAG, and fine-tuning—into the production lifecycle: **package**, **serve**, **deploy**, **monitor**, and **improve**.

You will package a real scikit-learn pipeline with `joblib`, wrap it in a **FastAPI** service with typed Pydantic schemas, write a Dockerfile, build a tiny **model registry** with stage transitions, design a CI/CD workflow that gates on evaluation thresholds, and stand up monitoring with **drift detection** (PSI / KS), latency tracking, and structured logs. Everything runs offline with no real Docker, no real cloud—just the patterns you'd use in production.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Package ML models for production** — serialize sklearn pipelines, freeze dependencies, define typed I/O schemas
2. **Serve models behind an HTTP API** — FastAPI with `/predict`, `/health`, `/version`, batching, and async
3. **Containerize and reason about deployments** — Dockerfile layers, image size, health checks, readiness
4. **Build reproducible ML pipelines** — sklearn `Pipeline`, seeds, lockfiles, the data/code/model versioning triplet
5. **Track experiments and manage a model registry** — stages (None / Staging / Production / Archived) and promotion gates
6. **Design CI/CD for ML** — lint → test → train → eval → register → deploy with automated quality gates
7. **Monitor models in production** — data drift (PSI, KS), prediction drift, latency budgets, error rates, structured logs
8. **Operate models safely at scale** — A/B tests, canary releases, rollback policy, autoscaling and cost trade-offs

---

## Prerequisites

- **Chapters 1–14** — Python, ML fundamentals, deep learning, NLP, LLMs, RAG, fine-tuning
- Familiarity with the command line and HTTP basics
- Comfort with NumPy, pandas, and scikit-learn pipelines

---

## What You'll Build

- **FastAPI prediction service** — typed request/response, `/predict`, `/health`, `/version`, batch endpoint
- **Dockerfile** — minimal multi-stage container spec for the service (no real `docker run` required)
- **File-backed model registry** — register artifacts, transition stages, fetch the current Production model
- **Monitoring dashboard data** — drift report (PSI / KS), latency percentiles, structured JSON logs
- **CI workflow** — sample GitHub Actions YAML that gates deploys on eval metrics

---

## Time Commitment

| Section | Time |
|---------|------|
| Notebook 01: Packaging & Serving (joblib, Pydantic, FastAPI, Docker, health checks) | 2 hours |
| Notebook 02: Pipelines & CI/CD (sklearn Pipeline, tracking, registry, GitHub Actions, reproducibility) | 2.5 hours |
| Notebook 03: Advanced MLOps (drift, A/B & canary, observability, scaling, capstone) | 2.5 hours |
| Exercises (Problem Sets 1 & 2) | 1 hour |
| **Total** | **8 hours** |

---

## Technology Stack

- **Serving**: `fastapi`, `uvicorn`, `httpx` (test client)
- **Schemas**: `pydantic>=2`
- **Modeling**: `scikit-learn`, `numpy`, `pandas`, `joblib`
- **Monitoring**: NumPy-based drift (PSI / KS); optional `evidently`, `prometheus-client`
- **Notebooks**: `jupyter`, `ipywidgets`
- **Optional**: `mlflow`, `bentoml`, `docker` (none required to complete the chapter)

---

## Quick Start

1. **Clone and enter the chapter**
   ```bash
   cd chapters/chapter-15-mlops-and-model-deployment
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
   Start with `01_packaging_serving.ipynb`, then `02_pipelines_cicd.ipynb`, then `03_advanced_mlops.ipynb`.

---

## Notebook Guide

| Notebook | Focus |
|----------|--------|
| **01_packaging_serving.ipynb** | Lifecycle overview, joblib serialization, Pydantic schemas, FastAPI app with TestClient, batching & latency, Dockerfile authoring, health/readiness probes |
| **02_pipelines_cicd.ipynb** | sklearn `Pipeline`, reproducibility (seeds, lockfiles), experiment tracking (mlflow + JSON fallback), file-backed model registry, GitHub Actions CI, the data/code/model triplet |
| **03_advanced_mlops.ipynb** | Data & prediction drift (PSI, KS), Evidently sketch with NumPy fallback, A/B and canary traffic splitting, structured logs and Prometheus metrics, scaling & cost, capstone design |

---

## Exercise Guide

- **Problem Set 1** (`exercises/problem_set_1.ipynb`) — package a model with joblib, write a Pydantic schema, build `/predict`, write a Dockerfile string, batch predictions, add a `/version` endpoint
- **Problem Set 2** (`exercises/problem_set_2.ipynb`) — detect drift via PSI, implement a canary splitter, write a CI YAML with eval gates, build a tiny registry, structured logging middleware, design a rollback policy
- **Solutions** — in `exercises/solutions/` with runnable code, explanations, and alternatives

---

## How to Run Locally

- Use Python 3.9+ and the versions in `requirements.txt` for reproducibility.
- Notebooks are fully self-contained and run **offline**: no real Docker daemon, no cloud account, no MLflow server required. FastAPI is exercised via `fastapi.testclient.TestClient`.
- Scripts in `scripts/` can be imported from notebooks (they prepend `../scripts` to `sys.path`).
- Optional integrations (`mlflow`, `evidently`, `prometheus_client`, `bentoml`) are wrapped in `try/except` and fall back to local implementations if not installed.

---

## Common Troubleshooting

- **`ModuleNotFoundError: fastapi`** — Run `pip install -r requirements.txt`; FastAPI and Uvicorn are required for Notebook 01.
- **`pydantic.v1` import errors** — This chapter targets Pydantic v2. If you have v1 installed, run `pip install -U "pydantic>=2"`.
- **Port already in use** — The notebooks use `TestClient` (in-process) and never bind a port. If you launch `uvicorn` separately, change `--port`.
- **MLflow / Evidently not installed** — Expected; the notebooks fall back to a JSON tracker and a NumPy drift implementation.
- **Joblib version mismatch** — Models pickled by one joblib version may not load on another; pin `joblib` in `requirements.txt` for production.

---

## Next Steps

- **Chapter 16: Advanced Topics & Research Frontiers** — This chapter completes the **Practitioner Track**. Next, the **Advanced Track** explores agentic systems, evaluation at scale, multimodal models, alignment, and the open research questions shaping the next generation of AI.
- Apply the patterns here to your own projects: take a model you trained in Chapters 6–14, package it, register it, deploy it behind FastAPI, and add a drift monitor.

---

**Generated by Berta AI**

Part of [Berta Chapters](https://github.com/your-org/berta-chapters) — open-source AI curriculum.  
*May 2026 — Berta Chapters*

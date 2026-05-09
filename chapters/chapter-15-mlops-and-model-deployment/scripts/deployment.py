"""
Production-style serving module for Chapter 15: MLOps & Model Deployment.

Exposes a `ModelService` that wraps a trained sklearn pipeline and a
`build_app()` factory that returns a FastAPI application with `/predict`,
`/predict/batch`, `/health`, and `/version` endpoints.

The module is self-contained: it can be exercised in-process via
`fastapi.testclient.TestClient` without binding a real port.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

def _import_pydantic():
    try:
        from pydantic import BaseModel, Field, ConfigDict  # type: ignore
        return BaseModel, Field, ConfigDict
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pydantic>=2 is required. Install with: pip install 'pydantic>=2'"
        ) from exc


BaseModel, Field, ConfigDict = _import_pydantic()


class PredictRequest(BaseModel):
    """Single-record prediction request."""
    model_config = ConfigDict(extra="forbid")
    feature_a: float = Field(..., description="Numeric feature A.")
    feature_b: float = Field(..., description="Numeric feature B.")


class BatchPredictRequest(BaseModel):
    """Batch prediction request with up to N records."""
    model_config = ConfigDict(extra="forbid")
    records: List[PredictRequest] = Field(..., min_length=1, max_length=256)


class PredictResponse(BaseModel):
    """Single-record prediction response."""
    prediction: float
    probability: Optional[List[float]] = None
    model_version: str
    latency_ms: float


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[float]
    model_version: str
    latency_ms: float
    n: int


class HealthResponse(BaseModel):
    status: str
    ready: bool


class VersionResponse(BaseModel):
    name: str
    version: str
    stage: str
    framework: str


# ---------------------------------------------------------------------------
# Model service
# ---------------------------------------------------------------------------

@dataclass
class ModelService:
    """
    Wraps a trained sklearn-style estimator and serves predictions.

    Attributes:
        model: A fitted estimator with `.predict` (and optional `.predict_proba`).
        name: Logical model name (e.g. 'churn-classifier').
        version: Semantic version of the loaded artifact (e.g. '1.2.3').
        stage: Lifecycle stage label (e.g. 'Production', 'Staging').
        framework: Origin framework string (e.g. 'sklearn', 'pytorch').
    """
    model: Any
    name: str = "default-model"
    version: str = "0.1.0"
    stage: str = "None"
    framework: str = "sklearn"
    _ready: bool = field(default=True, repr=False)

    @classmethod
    def from_joblib(
        cls,
        path: str | Path,
        name: str = "default-model",
        version: str = "0.1.0",
        stage: str = "None",
    ) -> "ModelService":
        """Load a joblib-pickled estimator from disk."""
        import joblib  # local import keeps module importable without joblib
        model = joblib.load(Path(path))
        return cls(model=model, name=name, version=version, stage=stage)

    def predict_one(self, record: Dict[str, float]) -> Dict[str, Any]:
        """Score a single record. Returns prediction + optional probabilities."""
        x = np.asarray([[record["feature_a"], record["feature_b"]]], dtype=float)
        t0 = time.perf_counter()
        y = self.model.predict(x)
        proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(x)[0].tolist()
            except Exception:  # estimator without classes
                proba = None
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "prediction": float(y[0]),
            "probability": proba,
            "model_version": self.version,
            "latency_ms": latency_ms,
        }

    def predict_batch(self, records: Sequence[Dict[str, float]]) -> Dict[str, Any]:
        """Score a batch of records in a single estimator call (vectorized)."""
        if not records:
            return {"predictions": [], "model_version": self.version,
                    "latency_ms": 0.0, "n": 0}
        x = np.asarray(
            [[r["feature_a"], r["feature_b"]] for r in records],
            dtype=float,
        )
        t0 = time.perf_counter()
        y = self.model.predict(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "predictions": [float(v) for v in y],
            "model_version": self.version,
            "latency_ms": latency_ms,
            "n": int(len(records)),
        }

    def is_ready(self) -> bool:
        return bool(self._ready and self.model is not None)


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def build_app(service: ModelService):
    """
    Build a FastAPI application that exposes the given ModelService.

    Endpoints:
        GET  /health   -> liveness/readiness probe
        GET  /version  -> name, version, stage, framework
        POST /predict  -> single-record prediction
        POST /predict/batch -> batch prediction
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fastapi is required. Install with: pip install fastapi uvicorn"
        ) from exc

    app = FastAPI(title=f"{service.name} :: model service", version=service.version)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if service.is_ready() else "loading",
            ready=service.is_ready(),
        )

    @app.get("/version", response_model=VersionResponse)
    def version() -> VersionResponse:
        return VersionResponse(
            name=service.name,
            version=service.version,
            stage=service.stage,
            framework=service.framework,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if not service.is_ready():
            raise HTTPException(status_code=503, detail="model not ready")
        result = service.predict_one(req.model_dump())
        return PredictResponse(**result)

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
        if not service.is_ready():
            raise HTTPException(status_code=503, detail="model not ready")
        records = [r.model_dump() for r in req.records]
        result = service.predict_batch(records)
        return BatchPredictResponse(**result)

    return app


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def predict_batch(model: Any, records: Sequence[Dict[str, float]]) -> List[float]:
    """
    Stateless batch helper used by tests and notebooks. Vectorizes the call
    so latency scales sublinearly with batch size.
    """
    if not records:
        return []
    x = np.asarray(
        [[r["feature_a"], r["feature_b"]] for r in records],
        dtype=float,
    )
    y = model.predict(x)
    return [float(v) for v in y]


__all__ = [
    "ModelService",
    "build_app",
    "predict_batch",
    "PredictRequest",
    "BatchPredictRequest",
    "PredictResponse",
    "BatchPredictResponse",
    "HealthResponse",
    "VersionResponse",
]

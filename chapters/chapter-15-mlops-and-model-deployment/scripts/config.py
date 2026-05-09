"""
Configuration and constants for Chapter 15: MLOps & Model Deployment.
Centralizes paths, thresholds, and service settings for scripts and notebooks.
"""

from pathlib import Path

# --- File paths (relative to chapter root) ---
DATA_DIR = "datasets/"
MODEL_DIR = "models/"
REGISTRY_DIR = "registry/"
LOGS_DIR = "logs/"
RESULTS_DIR = "results/"

# Default artifact filenames
DEFAULT_MODEL_FILE = "model.joblib"
REGISTRY_INDEX_FILE = "index.json"
PREDICTION_LOG_FILE = "predictions.jsonl"

# --- Service settings ---
SERVICE_HOST = "0.0.0.0"
SERVICE_PORT = 8000
REQUEST_TIMEOUT_S = 30
MAX_BATCH_SIZE = 256

# --- Latency budgets (milliseconds) ---
P50_LATENCY_MS = 50
P95_LATENCY_MS = 150
P99_LATENCY_MS = 300

# --- Drift thresholds ---
DRIFT_PSI_WARN = 0.1     # Population Stability Index: 0.1 = small shift
DRIFT_PSI_ALERT = 0.25   # 0.25+ is a meaningful shift
DRIFT_KS_PVALUE = 0.05   # KS test significance threshold

# --- Quality gates for promotion to Production ---
MIN_ACCURACY = 0.80
MIN_F1 = 0.75
MAX_LATENCY_P95_MS = 200

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Registry stages ---
STAGE_NONE = "None"
STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"
VALID_STAGES = (STAGE_NONE, STAGE_STAGING, STAGE_PRODUCTION, STAGE_ARCHIVED)


def chapter_root() -> Path:
    """Return the chapter root directory (parent of this scripts/ folder)."""
    return Path(__file__).resolve().parent.parent

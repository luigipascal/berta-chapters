"""
Configuration and constants for Chapter 12: Prompt Engineering & In-Context Learning.
Centralizes paths, model names, default sampling, eval thresholds, and the
prompt registry location.
"""

# --- Default model identifiers (used as labels; no real API calls) ---
DEFAULT_MODEL = "mock-llm-v1"
ALT_MODEL = "mock-llm-v2"

# --- Sampling defaults ---
DEFAULT_TEMPERATURE = 0.0          # Deterministic by default
HIGH_TEMPERATURE = 0.7             # For self-consistency / creative tasks
DEFAULT_MAX_TOKENS = 256
DEFAULT_TOP_P = 1.0
RANDOM_SEED = 42

# --- Few-shot defaults ---
DEFAULT_NUM_EXAMPLES = 3
MAX_FEW_SHOT_EXAMPLES = 8

# --- Self-consistency defaults ---
SELF_CONSISTENCY_SAMPLES = 5

# --- Evaluation thresholds ---
EXACT_MATCH_THRESHOLD = 1.0        # Pass = perfect match
COSINE_MATCH_THRESHOLD = 0.75      # TF-IDF cosine similarity pass bar
RUBRIC_PASS_SCORE = 0.7            # Average rubric score >= this to pass
AB_BOOTSTRAP_ITERATIONS = 1000     # Bootstrap resamples for A/B CIs

# --- Latency / production budgets (illustrative) ---
LATENCY_BUDGET_MS = 2000
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.5

# --- File paths (relative to chapter root) ---
DATA_DIR = "datasets/"
REGISTRY_DIR = "registry/"
RESULTS_DIR = "results/"
EVAL_TASKS_FILE = "datasets/eval_tasks.csv"
EXAMPLE_PROMPTS_FILE = "datasets/example_prompts.json"
INJECTION_EXAMPLES_FILE = "datasets/injection_examples.txt"

# --- Injection defense: simple allow/deny heuristics ---
INJECTION_DENY_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (the )?(system|previous)",
    r"reveal (your )?(system )?prompt",
    r"forget (everything|all)",
    r"act as .*(?:dan|jailbreak|developer mode)",
]

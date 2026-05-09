"""
Configuration and constants for Chapter 11: Large Language Models & Transformers.
Centralizes paths, hyperparameters, and model names for scripts and notebooks.
"""

# --- Default model names (Hugging Face hub IDs) ---
MODEL_NAME = "distilbert-base-uncased"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "gpt2"

# --- Tokenization / sequence ---
MAX_LENGTH = 128
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1

# --- Transformer architecture (pure-NumPy demos) ---
EMBED_DIM = 64          # d_model in the demos
NUM_HEADS = 4           # multi-head attention heads
NUM_LAYERS = 2          # encoder block stack depth
FFN_HIDDEN = 256        # feed-forward inner dim
DROPOUT_RATE = 0.1      # used conceptually; numpy demos run dropout-free

# --- Decoding / generation ---
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_NEW_TOKENS = 32
REPETITION_PENALTY = 1.1

# --- Training (frozen-embedding head, etc.) ---
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 5e-5
RANDOM_SEED = 42

# --- File paths (relative to chapter root) ---
DATA_DIR = "datasets/"
MODEL_DIR = "models/"
RESULTS_DIR = "results/"

# --- Curated model registry ---
MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "gpt2": "gpt2",
}

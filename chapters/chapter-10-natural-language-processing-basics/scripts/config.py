"""
Configuration and constants for Chapter 10: Natural Language Processing Basics.
Centralizes paths, hyperparameters, and model names for scripts and notebooks.
"""

# --- Text preprocessing ---
STOPWORDS_LANGUAGE = "english"
MIN_TOKEN_LENGTH = 2
MAX_VOCAB_SIZE = 10000

# --- Model hyperparameters ---
EMBEDDING_DIM = 100
LSTM_UNITS = 128
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001

# --- Training ---
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# --- File paths (relative to chapter root) ---
DATA_DIR = "data/"
MODEL_DIR = "models/"
RESULTS_DIR = "results/"

# --- Available models (paths under MODEL_DIR) ---
MODELS = {
    "sentiment": "models/sentiment_model.pkl",
    "classifier": "models/text_classifier.h5",
    "embeddings": "models/embeddings.pkl",
}

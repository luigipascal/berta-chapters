"""
Configuration and constants for Chapter 13: Retrieval-Augmented Generation.
Centralizes paths, hyperparameters, and model names for scripts and notebooks.
"""

# --- Chunking ---
CHUNK_SIZE = 256              # Target tokens per chunk
CHUNK_OVERLAP = 32            # Sliding-window overlap in tokens
MIN_CHUNK_TOKENS = 16         # Drop chunks shorter than this
MAX_CHUNK_TOKENS = 512        # Hard cap

# --- Embeddings ---
EMBEDDING_DIM = 384           # Default for sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Used if sentence-transformers is installed
TFIDF_MAX_FEATURES = 4096     # Fallback embedding when ST is unavailable
NORMALIZE_EMBEDDINGS = True   # L2-normalize so dot product == cosine

# --- Retrieval ---
TOP_K = 5                     # Default number of chunks returned
RERANK_TOP_K = 20             # Candidates pulled before reranking
HYBRID_DENSE_WEIGHT = 0.5     # Weight for dense scores in linear-fusion
HYBRID_SPARSE_WEIGHT = 0.5    # Weight for BM25 scores in linear-fusion
RRF_K = 60                    # Reciprocal-rank-fusion constant

# --- LLM / generation ---
LLM_MODEL = "mock-llm"        # Default offline model used by RAGPipeline
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.0         # Low temperature for grounded answers

# --- Evaluation ---
EVAL_TOP_KS = (1, 3, 5, 10)
RANDOM_SEED = 42

# --- File paths (relative to chapter root) ---
DATA_DIR = "datasets/"
INDEX_DIR = "indexes/"
RESULTS_DIR = "results/"

CORPUS_PATH = "datasets/sample_corpus.txt"
QUERIES_PATH = "datasets/queries.csv"
QA_PAIRS_PATH = "datasets/qa_pairs.json"

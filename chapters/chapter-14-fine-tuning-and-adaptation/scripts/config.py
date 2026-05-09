"""
Configuration and constants for Chapter 14: Fine-tuning & Adaptation Techniques.
Centralizes paths, hyperparameters, and model names for scripts and notebooks.
"""

# --- Dataset & tokenization ---
INSTRUCTION_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
RESPONSE_TEMPLATE = "{output}"
MAX_SEQ_LEN = 512
TRAIN_FRACTION = 0.9
RANDOM_SEED = 42

# --- SFT hyperparameters ---
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
GRAD_CLIP = 1.0

# --- LoRA / PEFT hyperparameters ---
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("q_proj", "v_proj")  # typical attention projections

# --- DPO ---
DPO_BETA = 0.1

# --- File paths (relative to chapter root) ---
DATA_DIR = "datasets/"
ADAPTER_DIR = "adapters/"
REGISTRY_DIR = "registry/"
RESULTS_DIR = "results/"

# --- Model / adapter registry stub (paths and metadata) ---
MODELS = {
    "base_lm": "gpt2",                          # placeholder backbone for sketches
    "sft_adapter": "adapters/sft_v1.bin",
    "lora_adapter": "adapters/lora_v1.safetensors",
    "dpo_adapter": "adapters/dpo_v1.safetensors",
    "registry_index": "registry/index.yaml",
}

"""
Configuration for the compression solution.
"""
from pathlib import Path

# Paths
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "compression_challenge_submission"
MODEL_DIR = HERE / "models" / "checkpoints"

# Dataset
DATA_FILES = {"train": ["data-0000.tar.gz", "data-0001.tar.gz"]}
DATASET_NAME = "commaai/commavq"

# Token properties
NUM_TOKENS = 1024          # VQ codebook size (10 bits)
TOKENS_PER_FRAME = 128     # 8 × 16 tokens per frame
FRAMES_PER_VIDEO = 1200    # 1 minute at 20 fps
BITS_PER_TOKEN = 10

# Compression settings
COMPRESSION_METHOD = "arithmetic"  # "lzma", "arithmetic", "ans"
USE_NEURAL_MODEL = False           # Set True to use GPT-style predictor
CONTEXT_SIZE = 256                 # Context window for prediction

# Model settings (if using neural predictor)
MODEL_CONFIG = {
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 4,
    "vocab_size": NUM_TOKENS,
    "block_size": CONTEXT_SIZE,
    "dropout": 0.0,
}


#!/usr/bin/env python3
"""
Main compression script for the commaVQ compression challenge.

This script:
1. Loads the first two splits of the commaVQ dataset
2. Compresses each example using your chosen method
3. Creates a submission zip file with compressed data + decompress.py

Usage:
    python compress.py

The submission will be saved to: compression_challenge_submission.zip
"""
import os
import lzma
import shutil
import multiprocessing
import numpy as np
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset
from config import (
    HERE, OUTPUT_DIR, DATA_FILES, DATASET_NAME,
    NUM_TOKENS, BITS_PER_TOKEN, COMPRESSION_METHOD
)

# ============================================================================
# Compression Methods
# ============================================================================

def compress_lzma(tokens: np.ndarray) -> bytes:
    """
    Baseline compression using LZMA.
    Expected compression rate: ~1.6
    """
    # Transpose to group similar values (improves compression!)
    tokens = tokens.astype(np.int16).reshape(-1, 128).T.ravel().tobytes()
    return lzma.compress(tokens, preset=9)


def compress_with_preprocessing(tokens: np.ndarray) -> bytes:
    """
    Enhanced LZMA with better preprocessing.
    
    Ideas to try:
    - Different data layouts (transpose, interleave, etc.)
    - Delta encoding between frames
    - Quantization-aware rearrangement
    """
    # Reshape to (frames, height, width) = (1200, 8, 16)
    tokens = tokens.reshape(1200, 8, 16)
    
    # Option 1: Delta encoding between consecutive frames
    # deltas = np.zeros_like(tokens)
    # deltas[0] = tokens[0]
    # deltas[1:] = tokens[1:] - tokens[:-1]
    # data = deltas.astype(np.int16).tobytes()
    
    # Option 2: Transpose each frame (group columns together)
    # transposed = tokens.transpose(0, 2, 1)  # (1200, 16, 8)
    # data = transposed.astype(np.int16).tobytes()
    
    # Option 3: Interleave spatial positions across frames
    # This groups tokens at the same position across all frames
    interleaved = tokens.transpose(1, 2, 0)  # (8, 16, 1200)
    data = interleaved.astype(np.int16).tobytes()
    
    return lzma.compress(data, preset=9)


def compress_arithmetic(tokens: np.ndarray) -> bytes:
    """
    Arithmetic coding with statistical model.
    
    For better compression:
    1. Train a prediction model on the dataset
    2. Use model predictions for probability distributions
    3. Apply arithmetic coding with those probabilities
    
    Expected compression rate: 2.0-2.5 with good statistics
    """
    from encoders.arithmetic import ArithmeticEncoder, estimate_marginal_distribution
    from models.statistical import MarkovPredictor, SpatialPredictor
    
    tokens_flat = tokens.ravel()
    
    # Initialize predictor
    # Option 1: Simple marginal distribution
    # probs = estimate_marginal_distribution(tokens_flat, NUM_TOKENS)
    # encoder = ArithmeticEncoder(NUM_TOKENS)
    # for token in tokens_flat:
    #     encoder.encode_symbol(token, probs)
    
    # Option 2: Adaptive Markov model
    predictor = MarkovPredictor(num_symbols=NUM_TOKENS, order=2, smoothing=0.1)
    encoder = ArithmeticEncoder(NUM_TOKENS)
    
    context = np.array([], dtype=np.int64)
    for token in tqdm(tokens_flat, desc="Encoding", leave=False):
        probs = predictor.predict(context)
        encoder.encode_symbol(int(token), probs)
        predictor.update(int(token), context)
        context = np.append(context, token)[-256:]  # Keep last 256 tokens
    
    return encoder.finish()


def compress_example(example, method: str = "lzma") -> dict:
    """
    Compress a single example from the dataset.
    
    Args:
        example: Dataset example with 'token.npy' and 'json' fields
        method: Compression method to use
        
    Returns:
        Example with added 'compressed' and 'compression_rate' fields
    """
    tokens = np.array(example['token.npy'])
    name = example['json']['file_name']
    
    # Choose compression method
    if method == "lzma":
        compressed = compress_lzma(tokens)
    elif method == "lzma_enhanced":
        compressed = compress_with_preprocessing(tokens)
    elif method == "arithmetic":
        compressed = compress_arithmetic(tokens)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate compression rate
    original_bits = tokens.size * BITS_PER_TOKEN
    original_bytes = original_bits / 8
    compression_rate = original_bytes / len(compressed)
    
    # Save compressed file
    with open(OUTPUT_DIR / name, 'wb') as f:
        f.write(compressed)
    
    return {
        **example,
        'compression_rate': compression_rate,
        'compressed_size': len(compressed),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("CommaVQ Compression Challenge")
    print("=" * 60)
    print(f"Method: {COMPRESSION_METHOD}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    num_proc = multiprocessing.cpu_count()
    ds = load_dataset(DATASET_NAME, num_proc=num_proc, data_files=DATA_FILES)
    
    total_examples = sum(ds.num_rows.values())
    print(f"Loaded {total_examples} examples")
    print()
    
    # Compress all examples
    print("Compressing...")
    
    # For arithmetic coding, we need sequential processing
    if COMPRESSION_METHOD == "arithmetic":
        results = []
        for example in tqdm(ds['train'], desc="Compressing"):
            result = compress_example(example, method=COMPRESSION_METHOD)
            results.append(result)
        
        avg_rate = np.mean([r['compression_rate'] for r in results])
    else:
        # For LZMA methods, we can use parallel processing
        def compress_wrapper(example):
            return compress_example(example, method=COMPRESSION_METHOD)
        
        results = ds.map(
            compress_wrapper,
            desc="Compressing",
            num_proc=num_proc,
            load_from_cache_file=False
        )
        avg_rate = np.mean(results['train']['compression_rate'])
    
    print(f"Average compression rate per file: {avg_rate:.2f}")
    print()
    
    # Copy decompress.py to output directory
    print("Creating submission archive...")
    shutil.copy(HERE / 'decompress.py', OUTPUT_DIR)
    
    # Also copy any model files if using neural compression
    model_file = HERE / 'models' / 'checkpoints' / 'model.pt'
    if model_file.exists():
        shutil.copy(model_file, OUTPUT_DIR)
    
    # Create zip archive
    archive_path = HERE / 'compression_challenge_submission'
    shutil.make_archive(str(archive_path), 'zip', OUTPUT_DIR)
    
    # Calculate final compression rate
    zip_size = os.path.getsize(str(archive_path) + '.zip')
    original_bits = total_examples * 1200 * 128 * BITS_PER_TOKEN
    original_bytes = original_bits / 8
    final_rate = original_bytes / zip_size
    
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Original size:   {original_bytes / 1e6:.1f} MB")
    print(f"Compressed size: {zip_size / 1e6:.1f} MB")
    print(f"Compression rate: {final_rate:.2f}")
    print()
    print(f"Submission: {archive_path}.zip")
    print("=" * 60)


if __name__ == '__main__':
    main()


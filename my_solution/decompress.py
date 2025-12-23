#!/usr/bin/env python3
"""
Decompression script for the commaVQ compression challenge.

This script is included in the submission and is used by the evaluator
to decompress your submission and verify correctness.

Usage (by evaluator):
    OUTPUT_DIR=/path/to/output python decompress.py

The script will:
1. Load the original dataset for file names and ground truth
2. Decompress each file from the current directory
3. Save decompressed .npy files to OUTPUT_DIR
4. Verify that decompressed data matches original
"""
import os
import lzma
import numpy as np
import multiprocessing
from pathlib import Path
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', HERE / 'decompressed'))

# ============================================================================
# Decompression Methods
# ============================================================================

def decompress_lzma(data: bytes) -> np.ndarray:
    """
    Decompress LZMA compressed tokens.
    Inverse of compress_lzma in compress.py
    """
    tokens = np.frombuffer(lzma.decompress(data), dtype=np.int16)
    # Reverse the transpose: was (128, frames) -> now (frames, 128)
    return tokens.reshape(128, -1).T.reshape(-1, 8, 16)


def decompress_lzma_enhanced(data: bytes) -> np.ndarray:
    """
    Decompress enhanced LZMA with preprocessing.
    Inverse of compress_with_preprocessing in compress.py
    """
    tokens = np.frombuffer(lzma.decompress(data), dtype=np.int16)
    # Reverse the interleave: was (8, 16, 1200) -> now (1200, 8, 16)
    tokens = tokens.reshape(8, 16, 1200).transpose(2, 0, 1)
    return tokens


def decompress_arithmetic(data: bytes, num_tokens: int) -> np.ndarray:
    """
    Decompress arithmetic coded tokens.
    
    Note: This requires the same prediction model used during compression!
    """
    from encoders.arithmetic import ArithmeticDecoder
    from models.statistical import MarkovPredictor
    
    decoder = ArithmeticDecoder(data, num_symbols=1024)
    predictor = MarkovPredictor(num_symbols=1024, order=2, smoothing=0.1)
    
    tokens = []
    context = np.array([], dtype=np.int64)
    
    for _ in range(num_tokens):
        probs = predictor.predict(context)
        token = decoder.decode_symbol(probs)
        tokens.append(token)
        predictor.update(token, context)
        context = np.append(context, token)[-256:]
    
    return np.array(tokens, dtype=np.int16).reshape(-1, 8, 16)


# ============================================================================
# Main Decompression
# ============================================================================

def decompress_file(compressed_path: Path, method: str = "lzma") -> np.ndarray:
    """
    Decompress a single file.
    
    Args:
        compressed_path: Path to compressed file
        method: Decompression method (must match compression method)
        
    Returns:
        Decompressed tokens of shape (1200, 8, 16)
    """
    with open(compressed_path, 'rb') as f:
        data = f.read()
    
    if method == "lzma":
        return decompress_lzma(data)
    elif method == "lzma_enhanced":
        return decompress_lzma_enhanced(data)
    elif method == "arithmetic":
        return decompress_arithmetic(data, num_tokens=1200 * 8 * 16)
    else:
        raise ValueError(f"Unknown method: {method}")


def decompress_example(example):
    """
    Decompress and verify a single example.
    
    Args:
        example: Dataset example with 'json' field containing 'file_name'
    """
    name = example['json']['file_name']
    compressed_path = HERE / name
    
    # Check if file exists
    if not compressed_path.exists():
        raise FileNotFoundError(f"Compressed file not found: {compressed_path}")
    
    # Decompress
    # TODO: Change method to match your compression method!
    tokens = decompress_file(compressed_path, method="lzma")
    
    # Save decompressed file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_DIR / name, tokens)
    
    # Verify against ground truth
    gt_tokens = np.array(example['token.npy'])
    if not np.all(tokens == gt_tokens):
        raise AssertionError(f"Decompressed data does not match original for {name}")
    
    return example


def main():
    print("Decompressing submission...")
    print(f"Input:  {HERE}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset for file names and ground truth
    num_proc = multiprocessing.cpu_count()
    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)
    
    total = sum(ds.num_rows.values())
    print(f"Decompressing {total} files...")
    
    # Decompress all files
    # Note: For arithmetic coding, we need sequential processing
    # For LZMA, we can use parallel processing
    
    ds.map(
        decompress_example,
        desc="Decompressing",
        num_proc=num_proc,
        load_from_cache_file=False
    )
    
    print()
    print("Decompression complete! All files verified.")


if __name__ == '__main__':
    main()


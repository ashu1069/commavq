#!/usr/bin/env python3
"""
Analyze the token data to understand patterns for better compression.

This script helps you understand:
- Token distribution (entropy)
- Spatial correlations (within frames)
- Temporal correlations (across frames)
- Potential compression limits

Usage:
    python scripts/analyze_data.py
"""
import sys
import numpy as np
import multiprocessing
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from config import DATA_FILES, DATASET_NAME, NUM_TOKENS


def entropy(probs):
    """Calculate entropy in bits."""
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def analyze_distribution(tokens_list):
    """Analyze global token distribution."""
    print("\n" + "=" * 60)
    print("Token Distribution Analysis")
    print("=" * 60)
    
    # Count all tokens
    all_tokens = np.concatenate([t.ravel() for t in tokens_list])
    counts = np.bincount(all_tokens, minlength=NUM_TOKENS)
    probs = counts / counts.sum()
    
    print(f"Total tokens: {len(all_tokens):,}")
    print(f"Unique tokens used: {np.sum(counts > 0)} / {NUM_TOKENS}")
    print()
    
    # Entropy
    h = entropy(probs)
    print(f"Entropy (0-order): {h:.3f} bits")
    print(f"Original encoding: {np.log2(NUM_TOKENS):.3f} bits (10 bits)")
    print(f"Theoretical compression limit: {10/h:.2f}x")
    print()
    
    # Most common tokens
    print("Most common tokens:")
    top_k = 10
    sorted_idx = np.argsort(-counts)[:top_k]
    for i, idx in enumerate(sorted_idx):
        pct = counts[idx] / counts.sum() * 100
        print(f"  {i+1}. Token {idx}: {counts[idx]:,} ({pct:.2f}%)")


def analyze_spatial_correlation(tokens_list):
    """Analyze correlation between adjacent tokens in a frame."""
    print("\n" + "=" * 60)
    print("Spatial Correlation Analysis")
    print("=" * 60)
    
    # Collect statistics
    horizontal_matches = 0
    horizontal_total = 0
    vertical_matches = 0
    vertical_total = 0
    
    for tokens in tqdm(tokens_list[:100], desc="Analyzing spatial"):  # Sample
        # tokens shape: (1200, 8, 16)
        for frame in tokens:
            # Horizontal (left-right)
            horizontal_matches += np.sum(frame[:, :-1] == frame[:, 1:])
            horizontal_total += frame[:, :-1].size
            
            # Vertical (up-down)
            vertical_matches += np.sum(frame[:-1, :] == frame[1:, :])
            vertical_total += frame[:-1, :].size
    
    h_match_rate = horizontal_matches / horizontal_total
    v_match_rate = vertical_matches / vertical_total
    
    print(f"Horizontal match rate: {h_match_rate:.4f} ({h_match_rate*100:.2f}%)")
    print(f"Vertical match rate: {v_match_rate:.4f} ({v_match_rate*100:.2f}%)")
    print()
    print("Higher match rates indicate more spatial redundancy to exploit!")


def analyze_temporal_correlation(tokens_list):
    """Analyze correlation between consecutive frames."""
    print("\n" + "=" * 60)
    print("Temporal Correlation Analysis")
    print("=" * 60)
    
    frame_matches = 0
    frame_total = 0
    
    for tokens in tqdm(tokens_list[:100], desc="Analyzing temporal"):  # Sample
        # Compare consecutive frames
        matches = np.sum(tokens[:-1] == tokens[1:])
        total = tokens[:-1].size
        
        frame_matches += matches
        frame_total += total
    
    match_rate = frame_matches / frame_total
    
    print(f"Temporal match rate: {match_rate:.4f} ({match_rate*100:.2f}%)")
    print()
    print("Higher temporal match rate = more redundancy between frames!")


def analyze_conditional_entropy(tokens_list):
    """Estimate conditional entropy given previous token."""
    print("\n" + "=" * 60)
    print("Conditional Entropy Analysis")
    print("=" * 60)
    
    # Count bigrams
    bigram_counts = Counter()
    unigram_counts = Counter()
    
    for tokens in tqdm(tokens_list[:50], desc="Counting bigrams"):  # Sample
        flat = tokens.ravel()
        for i in range(len(flat) - 1):
            bigram_counts[(flat[i], flat[i+1])] += 1
            unigram_counts[flat[i]] += 1
    
    # Calculate conditional entropy H(X|Y)
    # H(X|Y) = sum_y P(y) * H(X|Y=y)
    total_bigrams = sum(bigram_counts.values())
    
    conditional_entropy = 0
    for prev_token, prev_count in unigram_counts.items():
        # Get distribution of next token given prev_token
        next_counts = np.zeros(NUM_TOKENS)
        for (t1, t2), count in bigram_counts.items():
            if t1 == prev_token:
                next_counts[t2] = count
        
        if next_counts.sum() > 0:
            probs = next_counts / next_counts.sum()
            h = entropy(probs)
            p_prev = prev_count / sum(unigram_counts.values())
            conditional_entropy += p_prev * h
    
    print(f"Conditional entropy H(X|X_prev): {conditional_entropy:.3f} bits")
    print(f"1st-order compression limit: {10/conditional_entropy:.2f}x")


def main():
    print("=" * 60)
    print("CommaVQ Data Analysis")
    print("=" * 60)
    
    # Load a sample of the dataset
    print("\nLoading dataset...")
    num_proc = min(4, multiprocessing.cpu_count())
    ds = load_dataset(DATASET_NAME, num_proc=num_proc, data_files=DATA_FILES)
    
    # Extract tokens
    print("Extracting tokens...")
    tokens_list = [np.array(ex['token.npy']) for ex in tqdm(ds['train'])]
    
    print(f"\nLoaded {len(tokens_list)} videos")
    print(f"Each video: {tokens_list[0].shape} = (frames, height, width)")
    
    # Run analyses
    analyze_distribution(tokens_list)
    analyze_spatial_correlation(tokens_list)
    analyze_temporal_correlation(tokens_list)
    analyze_conditional_entropy(tokens_list)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print()
    print("Key insights:")
    print("- 0-order entropy tells you the theoretical limit with no context")
    print("- Conditional entropy shows improvement possible with 1 token of context")
    print("- Spatial/temporal correlations suggest where to focus modeling effort")
    print("- Higher-order models (GPT) can achieve even better predictions!")
    print()


if __name__ == '__main__':
    main()


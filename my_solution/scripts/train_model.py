#!/usr/bin/env python3
"""
Train a prediction model for compression.

For competitive compression rates, you want a model that:
1. Accurately predicts P(next_token | context)
2. Is small enough that model size doesn't hurt compression ratio
3. Is fast enough for practical compression/decompression

Usage:
    python scripts/train_model.py

The trained model will be saved to: models/checkpoints/model.pt
"""
import os
import sys
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    HERE, MODEL_DIR, DATA_FILES, DATASET_NAME, 
    NUM_TOKENS, MODEL_CONFIG
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")
    sys.exit(1)

from datasets import load_dataset
from models.neural import GPTModel


class TokenDataset(Dataset):
    """Dataset of token sequences for training."""
    
    def __init__(self, tokens_list: list, block_size: int = 256):
        self.block_size = block_size
        
        # Concatenate all token sequences
        self.data = []
        for tokens in tokens_list:
            flat = tokens.ravel()
            self.data.extend(flat.tolist())
        self.data = np.array(self.data, dtype=np.int64)
        
        print(f"Total tokens: {len(self.data):,}")
        print(f"Sequences of length {block_size}: {len(self):,}")
    
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        
        # Cross-entropy loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate model and compute bits per token."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    # Convert cross-entropy to bits: CE is in nats, multiply by log2(e)
    bits_per_token = avg_loss / np.log(2)
    
    return avg_loss, bits_per_token


def main():
    print("=" * 60)
    print("Training Prediction Model")
    print("=" * 60)
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    batch_size = 64
    num_epochs = 10
    learning_rate = 3e-4
    block_size = MODEL_CONFIG.get('block_size', 256)
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    num_proc = multiprocessing.cpu_count()
    ds = load_dataset(DATASET_NAME, num_proc=num_proc, data_files=DATA_FILES)
    
    # Extract tokens
    print("Extracting tokens...")
    tokens_list = [np.array(ex['token.npy']) for ex in tqdm(ds['train'])]
    
    # Split into train/val
    split_idx = int(len(tokens_list) * 0.95)
    train_tokens = tokens_list[:split_idx]
    val_tokens = tokens_list[split_idx:]
    
    print(f"Train examples: {len(train_tokens)}")
    print(f"Val examples: {len(val_tokens)}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TokenDataset(train_tokens, block_size=block_size)
    val_dataset = TokenDataset(val_tokens, block_size=block_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model = GPTModel(
        vocab_size=MODEL_CONFIG.get('vocab_size', NUM_TOKENS),
        n_embd=MODEL_CONFIG.get('n_embd', 128),
        n_head=MODEL_CONFIG.get('n_head', 4),
        n_layer=MODEL_CONFIG.get('n_layer', 4),
        block_size=block_size,
        dropout=MODEL_CONFIG.get('dropout', 0.0),
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = num_params * 4 / 1e6  # float32 = 4 bytes
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, bits_per_token = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  Bits per token: {bits_per_token:.2f}")
        print(f"  Theoretical compression rate: {10 / bits_per_token:.2f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = MODEL_DIR / 'model.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
        
        print()
    
    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_DIR / 'model.pt'}")
    print("=" * 60)


if __name__ == '__main__':
    main()


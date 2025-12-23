"""
Neural network prediction models (GPT-style).

For competitive compression rates (2.5+), you'll want to:
1. Train a GPT-style model on the token sequences
2. Use the model's probability predictions for arithmetic coding
3. Include the trained model weights in your submission

Note: The model size counts against your compression ratio!
      Smaller models with good predictions are ideal.
"""
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base import BasePredictor


if TORCH_AVAILABLE:
    class CausalSelfAttention(nn.Module):
        """Multi-head causal self-attention."""
        
        def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
            super().__init__()
            assert n_embd % n_head == 0
            
            self.n_head = n_head
            self.n_embd = n_embd
            self.head_dim = n_embd // n_head
            
            self.c_attn = nn.Linear(n_embd, 3 * n_embd)
            self.c_proj = nn.Linear(n_embd, n_embd)
            self.dropout = nn.Dropout(dropout)
            
            # Causal mask
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, C = x.size()
            
            # Project to Q, K, V
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            
            # Reshape for multi-head attention
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            
            # Attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.c_proj(y)
            
            return y
    
    
    class TransformerBlock(nn.Module):
        """Transformer block with attention and MLP."""
        
        def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
            super().__init__()
            self.ln1 = nn.LayerNorm(n_embd)
            self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
            self.ln2 = nn.LayerNorm(n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x
    
    
    class GPTModel(nn.Module):
        """
        Small GPT model for token prediction.
        
        This model predicts P(next_token | previous_tokens) which is used
        for arithmetic coding compression.
        """
        
        def __init__(
            self,
            vocab_size: int = 1024,
            n_embd: int = 128,
            n_head: int = 4,
            n_layer: int = 4,
            block_size: int = 256,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.block_size = block_size
            self.vocab_size = vocab_size
            
            self.tok_emb = nn.Embedding(vocab_size, n_embd)
            self.pos_emb = nn.Embedding(block_size, n_embd)
            self.drop = nn.Dropout(dropout)
            
            self.blocks = nn.Sequential(*[
                TransformerBlock(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ])
            
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, vocab_size, bias=False)
            
            # Initialize weights
            self.apply(self._init_weights)
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                idx: Token indices of shape (batch, seq_len)
                
            Returns:
                Logits of shape (batch, seq_len, vocab_size)
            """
            B, T = idx.size()
            assert T <= self.block_size, f"Sequence length {T} > block size {self.block_size}"
            
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
            
            tok_emb = self.tok_emb(idx)
            pos_emb = self.pos_emb(pos)
            x = self.drop(tok_emb + pos_emb)
            
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.head(x)
            
            return logits
        
        def get_probs(self, idx: torch.Tensor) -> torch.Tensor:
            """Get probability distributions for each position."""
            logits = self.forward(idx)
            return F.softmax(logits, dim=-1)
        
        def predict_next(self, context: torch.Tensor) -> torch.Tensor:
            """
            Predict probability distribution for the next token.
            
            Args:
                context: Token indices of shape (seq_len,) or (1, seq_len)
                
            Returns:
                Probability distribution of shape (vocab_size,)
            """
            if context.dim() == 1:
                context = context.unsqueeze(0)
            
            # Truncate to block size
            if context.size(1) > self.block_size:
                context = context[:, -self.block_size:]
            
            logits = self.forward(context)
            probs = F.softmax(logits[0, -1], dim=-1)
            return probs


class NeuralPredictor(BasePredictor):
    """
    Neural network based predictor using GPT-style model.
    
    Usage:
        predictor = NeuralPredictor()
        predictor.load("model.pt")  # Load trained model
        
        context = np.array([...])  # Previous tokens
        probs = predictor.predict(context)  # Get next token probabilities
    """
    
    def __init__(
        self,
        num_symbols: int = 1024,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 256,
        device: str = "cpu",
    ):
        super().__init__(num_symbols)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralPredictor")
        
        self.device = device
        self.block_size = block_size
        
        self.model = GPTModel(
            vocab_size=num_symbols,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size,
        ).to(device)
        
        self.model.eval()
    
    def predict(self, context: np.ndarray) -> np.ndarray:
        """
        Predict next token probability distribution.
        
        Args:
            context: Array of previous tokens
            
        Returns:
            Probability distribution of shape (num_symbols,)
        """
        if len(context) == 0:
            # No context: return uniform distribution
            return np.ones(self.num_symbols) / self.num_symbols
        
        with torch.no_grad():
            ctx_tensor = torch.tensor(context[-self.block_size:], dtype=torch.long, device=self.device)
            probs = self.model.predict_next(ctx_tensor)
            return probs.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
    
    def get_model_size(self) -> int:
        """Get model size in bytes (important for compression ratio!)."""
        return sum(p.numel() * p.element_size() for p in self.model.parameters())


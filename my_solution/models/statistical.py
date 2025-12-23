"""
Statistical prediction models (non-neural).
"""
import numpy as np
from typing import Optional, Dict, Tuple
from collections import defaultdict
import pickle

from .base import BasePredictor


class StatisticalPredictor(BasePredictor):
    """
    Simple statistical predictor using global token frequencies.
    
    This is a baseline that doesn't use context - just marginal frequencies.
    """
    
    def __init__(self, num_symbols: int = 1024, smoothing: float = 1.0):
        super().__init__(num_symbols)
        self.smoothing = smoothing
        self.counts = np.ones(num_symbols, dtype=np.float64) * smoothing
        self.total = num_symbols * smoothing
    
    def predict(self, context: np.ndarray) -> np.ndarray:
        """Return the marginal distribution (ignores context)."""
        return self.counts / self.total
    
    def update(self, token: int) -> None:
        """Update counts with observed token."""
        self.counts[token] += 1
        self.total += 1
    
    def fit(self, tokens: np.ndarray) -> None:
        """Fit the model to a batch of tokens."""
        for token in tokens.ravel():
            self.update(token)
    
    def save(self, path: str) -> None:
        """Save counts to file."""
        np.save(path, self.counts)
    
    def load(self, path: str) -> None:
        """Load counts from file."""
        self.counts = np.load(path)
        self.total = self.counts.sum()


class MarkovPredictor(BasePredictor):
    """
    Context-based predictor using n-gram statistics.
    
    Uses the previous `order` tokens to predict the next token.
    Falls back to lower-order models when context is unseen.
    """
    
    def __init__(self, num_symbols: int = 1024, order: int = 1, smoothing: float = 0.1):
        super().__init__(num_symbols)
        self.order = order
        self.smoothing = smoothing
        
        # Transition counts: context -> [count for each symbol]
        # We store counts for all orders 0..order for fallback
        self.transitions: Dict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.ones(num_symbols, dtype=np.float64) * smoothing
        )
        
        # Global counts (order 0)
        self.global_counts = np.ones(num_symbols, dtype=np.float64) * smoothing
        self.global_total = num_symbols * smoothing
    
    def predict(self, context: np.ndarray) -> np.ndarray:
        """
        Predict next token probability using n-gram context.
        Falls back to lower orders if context unseen.
        """
        if len(context) == 0:
            return self.global_counts / self.global_total
        
        # Try progressively shorter contexts
        for n in range(min(self.order, len(context)), 0, -1):
            ctx_tuple = tuple(context[-n:].tolist())
            if ctx_tuple in self.transitions:
                counts = self.transitions[ctx_tuple]
                return counts / counts.sum()
        
        # Fallback to global distribution
        return self.global_counts / self.global_total
    
    def update(self, token: int, context: Optional[np.ndarray] = None) -> None:
        """Update counts with observed token and context."""
        self.global_counts[token] += 1
        self.global_total += 1
        
        if context is not None and len(context) > 0:
            for n in range(1, min(self.order, len(context)) + 1):
                ctx_tuple = tuple(context[-n:].tolist())
                self.transitions[ctx_tuple][token] += 1
    
    def fit(self, tokens: np.ndarray) -> None:
        """
        Fit the model to a sequence of tokens.
        
        Args:
            tokens: 1D array of tokens
        """
        tokens = tokens.ravel()
        for i, token in enumerate(tokens):
            context = tokens[max(0, i - self.order):i]
            self.update(token, context)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            "order": self.order,
            "smoothing": self.smoothing,
            "transitions": dict(self.transitions),
            "global_counts": self.global_counts,
            "global_total": self.global_total,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> None:
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.order = data["order"]
        self.smoothing = data["smoothing"]
        self.transitions = defaultdict(
            lambda: np.ones(self.num_symbols, dtype=np.float64) * self.smoothing,
            data["transitions"]
        )
        self.global_counts = data["global_counts"]
        self.global_total = data["global_total"]


class SpatialPredictor(BasePredictor):
    """
    Predictor that exploits spatial structure in the 8×16 token grid.
    
    Uses neighboring tokens (above, left) as context for prediction.
    """
    
    def __init__(self, num_symbols: int = 1024, smoothing: float = 0.1):
        super().__init__(num_symbols)
        self.smoothing = smoothing
        
        # Conditional counts: (above_token, left_token) -> counts
        self.spatial_counts: Dict[Tuple[int, int], np.ndarray] = defaultdict(
            lambda: np.ones(num_symbols, dtype=np.float64) * smoothing
        )
        
        # Fallback: just left token
        self.left_counts: Dict[int, np.ndarray] = defaultdict(
            lambda: np.ones(num_symbols, dtype=np.float64) * smoothing
        )
        
        # Global fallback
        self.global_counts = np.ones(num_symbols, dtype=np.float64) * smoothing
        self.global_total = num_symbols * smoothing
    
    def predict_with_neighbors(
        self, 
        above: Optional[int] = None, 
        left: Optional[int] = None
    ) -> np.ndarray:
        """
        Predict using spatial neighbors.
        
        Args:
            above: Token directly above (or None if first row)
            left: Token to the left (or None if first column)
        """
        if above is not None and left is not None:
            key = (above, left)
            if key in self.spatial_counts:
                counts = self.spatial_counts[key]
                return counts / counts.sum()
        
        if left is not None and left in self.left_counts:
            counts = self.left_counts[left]
            return counts / counts.sum()
        
        return self.global_counts / self.global_total
    
    def predict(self, context: np.ndarray) -> np.ndarray:
        """Fallback predict using last tokens as pseudo-context."""
        if len(context) >= 2:
            return self.predict_with_neighbors(
                above=int(context[-16]) if len(context) >= 16 else None,
                left=int(context[-1])
            )
        elif len(context) >= 1:
            return self.predict_with_neighbors(left=int(context[-1]))
        return self.global_counts / self.global_total
    
    def update_with_neighbors(
        self, 
        token: int,
        above: Optional[int] = None,
        left: Optional[int] = None
    ) -> None:
        """Update counts with spatial context."""
        self.global_counts[token] += 1
        self.global_total += 1
        
        if left is not None:
            self.left_counts[left][token] += 1
        
        if above is not None and left is not None:
            self.spatial_counts[(above, left)][token] += 1
    
    def update(self, token: int) -> None:
        """Simple update without context (use update_with_neighbors for full)."""
        self.global_counts[token] += 1
        self.global_total += 1
    
    def fit_frame(self, frame: np.ndarray) -> None:
        """
        Fit the model to a single frame of shape (8, 16).
        """
        assert frame.shape == (8, 16), f"Expected (8, 16), got {frame.shape}"
        
        for i in range(8):
            for j in range(16):
                token = int(frame[i, j])
                above = int(frame[i - 1, j]) if i > 0 else None
                left = int(frame[i, j - 1]) if j > 0 else None
                self.update_with_neighbors(token, above, left)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            "spatial_counts": dict(self.spatial_counts),
            "left_counts": dict(self.left_counts),
            "global_counts": self.global_counts,
            "global_total": self.global_total,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> None:
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.spatial_counts = defaultdict(
            lambda: np.ones(self.num_symbols, dtype=np.float64) * self.smoothing,
            data["spatial_counts"]
        )
        self.left_counts = defaultdict(
            lambda: np.ones(self.num_symbols, dtype=np.float64) * self.smoothing,
            data["left_counts"]
        )
        self.global_counts = data["global_counts"]
        self.global_total = data["global_total"]


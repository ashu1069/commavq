"""
Base class for prediction models.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class BasePredictor(ABC):
    """
    Abstract base class for token predictors.
    
    A predictor estimates P(next_token | context) to enable
    efficient entropy coding.
    """
    
    def __init__(self, num_symbols: int = 1024):
        self.num_symbols = num_symbols
    
    @abstractmethod
    def predict(self, context: np.ndarray) -> np.ndarray:
        """
        Predict probability distribution for the next token.
        
        Args:
            context: Array of previous tokens
            
        Returns:
            Probability distribution of shape (num_symbols,)
        """
        pass
    
    def reset(self) -> None:
        """Reset any internal state (optional)."""
        pass
    
    def update(self, token: int) -> None:
        """Update internal state with observed token (optional)."""
        pass
    
    def save(self, path: str) -> None:
        """Save model parameters to file."""
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """Load model parameters from file."""
        raise NotImplementedError


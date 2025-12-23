"""
Arithmetic Coding implementation for lossless compression.

Arithmetic coding achieves near-optimal compression when given accurate
probability distributions. The compression rate approaches the entropy:
    rate = 1 / entropy(P)

For the compression challenge, we want to:
1. Predict P(next_token | context) as accurately as possible
2. Use arithmetic coding to encode tokens with those probabilities
"""
import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ArithmeticCoderState:
    """State for arithmetic coding."""
    low: int = 0
    high: int = 0xFFFFFFFF
    pending_bits: int = 0


class ArithmeticEncoder:
    """
    Arithmetic encoder for sequences with adaptive probabilities.
    
    Usage:
        encoder = ArithmeticEncoder(num_symbols=1024)
        for token, probs in zip(tokens, probability_distributions):
            encoder.encode_symbol(token, probs)
        compressed_bytes = encoder.finish()
    """
    
    PRECISION = 32
    MAX_VALUE = (1 << PRECISION) - 1
    HALF = 1 << (PRECISION - 1)
    QUARTER = 1 << (PRECISION - 2)
    
    def __init__(self, num_symbols: int = 1024):
        self.num_symbols = num_symbols
        self.low = 0
        self.high = self.MAX_VALUE
        self.pending_bits = 0
        self.output_bits: List[int] = []
    
    def encode_symbol(self, symbol: int, probs: np.ndarray) -> None:
        """
        Encode a symbol given its probability distribution.
        
        Args:
            symbol: The symbol to encode (0 to num_symbols-1)
            probs: Probability distribution over all symbols (must sum to 1)
        """
        assert 0 <= symbol < self.num_symbols
        assert len(probs) == self.num_symbols
        
        # Convert probabilities to cumulative distribution
        cumprobs = np.zeros(self.num_symbols + 1, dtype=np.float64)
        cumprobs[1:] = np.cumsum(probs)
        cumprobs = np.clip(cumprobs, 0, 1)
        
        # Update range
        range_size = self.high - self.low + 1
        self.high = self.low + int(range_size * cumprobs[symbol + 1]) - 1
        self.low = self.low + int(range_size * cumprobs[symbol])
        
        # Normalize and output bits
        self._normalize()
    
    def _normalize(self) -> None:
        """Normalize the range and output bits."""
        while True:
            if self.high < self.HALF:
                # Output 0 and pending 1s
                self._output_bit(0)
                self._output_pending_bits(1)
            elif self.low >= self.HALF:
                # Output 1 and pending 0s
                self._output_bit(1)
                self._output_pending_bits(0)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                # Middle case: defer decision
                self.pending_bits += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            
            # Scale up
            self.low = 2 * self.low
            self.high = 2 * self.high + 1
    
    def _output_bit(self, bit: int) -> None:
        """Output a single bit."""
        self.output_bits.append(bit)
    
    def _output_pending_bits(self, bit: int) -> None:
        """Output pending bits."""
        for _ in range(self.pending_bits):
            self.output_bits.append(bit)
        self.pending_bits = 0
    
    def finish(self) -> bytes:
        """
        Finish encoding and return compressed bytes.
        """
        # Output remaining bits to disambiguate
        self.pending_bits += 1
        if self.low < self.QUARTER:
            self._output_bit(0)
            self._output_pending_bits(1)
        else:
            self._output_bit(1)
            self._output_pending_bits(0)
        
        # Pad to byte boundary
        while len(self.output_bits) % 8 != 0:
            self.output_bits.append(0)
        
        # Convert bits to bytes
        result = bytearray()
        for i in range(0, len(self.output_bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self.output_bits[i + j]
            result.append(byte)
        
        return bytes(result)


class ArithmeticDecoder:
    """
    Arithmetic decoder for sequences with adaptive probabilities.
    
    Usage:
        decoder = ArithmeticDecoder(compressed_bytes, num_symbols=1024)
        for _ in range(num_tokens):
            probs = get_probability_distribution(context)
            token = decoder.decode_symbol(probs)
    """
    
    PRECISION = 32
    MAX_VALUE = (1 << PRECISION) - 1
    HALF = 1 << (PRECISION - 1)
    QUARTER = 1 << (PRECISION - 2)
    
    def __init__(self, data: bytes, num_symbols: int = 1024):
        self.num_symbols = num_symbols
        self.low = 0
        self.high = self.MAX_VALUE
        
        # Convert bytes to bits
        self.bits = []
        for byte in data:
            for i in range(7, -1, -1):
                self.bits.append((byte >> i) & 1)
        self.bit_index = 0
        
        # Initialize value from first PRECISION bits
        self.value = 0
        for _ in range(self.PRECISION):
            self.value = (self.value << 1) | self._read_bit()
    
    def _read_bit(self) -> int:
        """Read the next bit from input."""
        if self.bit_index < len(self.bits):
            bit = self.bits[self.bit_index]
            self.bit_index += 1
            return bit
        return 0
    
    def decode_symbol(self, probs: np.ndarray) -> int:
        """
        Decode a symbol given its probability distribution.
        
        Args:
            probs: Probability distribution over all symbols
            
        Returns:
            The decoded symbol
        """
        assert len(probs) == self.num_symbols
        
        # Convert probabilities to cumulative distribution
        cumprobs = np.zeros(self.num_symbols + 1, dtype=np.float64)
        cumprobs[1:] = np.cumsum(probs)
        cumprobs = np.clip(cumprobs, 0, 1)
        
        # Find the symbol
        range_size = self.high - self.low + 1
        scaled_value = (self.value - self.low) / range_size
        
        # Binary search for symbol
        symbol = np.searchsorted(cumprobs[1:], scaled_value, side='right')
        symbol = min(symbol, self.num_symbols - 1)
        
        # Update range
        self.high = self.low + int(range_size * cumprobs[symbol + 1]) - 1
        self.low = self.low + int(range_size * cumprobs[symbol])
        
        # Normalize
        self._normalize()
        
        return symbol
    
    def _normalize(self) -> None:
        """Normalize the range and read more bits."""
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.value -= self.HALF
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                self.value -= self.QUARTER
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            
            self.low = 2 * self.low
            self.high = 2 * self.high + 1
            self.value = 2 * self.value + self._read_bit()


# ============================================================================
# Simple probability estimation helpers
# ============================================================================

def uniform_distribution(num_symbols: int = 1024) -> np.ndarray:
    """Return uniform probability distribution."""
    return np.ones(num_symbols, dtype=np.float64) / num_symbols


def estimate_marginal_distribution(tokens: np.ndarray, num_symbols: int = 1024) -> np.ndarray:
    """
    Estimate marginal probability distribution from token counts.
    Uses Laplace smoothing to avoid zero probabilities.
    """
    counts = np.bincount(tokens.ravel(), minlength=num_symbols)
    probs = (counts + 1) / (tokens.size + num_symbols)  # Laplace smoothing
    return probs.astype(np.float64)


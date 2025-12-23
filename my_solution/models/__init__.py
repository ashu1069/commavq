"""
Prediction models for compression.
"""
from .base import BasePredictor
from .statistical import StatisticalPredictor, MarkovPredictor

__all__ = ["BasePredictor", "StatisticalPredictor", "MarkovPredictor"]


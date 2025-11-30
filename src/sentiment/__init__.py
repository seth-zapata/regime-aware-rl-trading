"""
Sentiment analysis module for SEC filings.

This module provides tools for:
1. FinBERT-based sentiment scoring of financial text
2. Aggregating sentiment across filing sections
3. Creating time-series sentiment features
"""

from .finbert_analyzer import FinBERTAnalyzer
from .filing_sentiment import FilingSentimentPipeline

__all__ = ['FinBERTAnalyzer', 'FilingSentimentPipeline']

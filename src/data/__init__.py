"""
Data Module

Provides data loading and processing for multiple sources:
- Price data (Yahoo Finance)
- Macroeconomic indicators (FRED)
- SEC filings (EDGAR)
- Unified data pipeline
"""

from .price_loader import PriceLoader
from .fred_loader import FREDLoader, FRED_SERIES, DEFAULT_REGIME_SERIES
from .edgar_loader import EDGARLoader, Filing
from .data_pipeline import DataPipeline

__all__ = [
    'PriceLoader',
    'FREDLoader',
    'FRED_SERIES',
    'DEFAULT_REGIME_SERIES',
    'EDGARLoader',
    'Filing',
    'DataPipeline',
]

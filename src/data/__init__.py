"""
Data acquisition and processing module.

This module handles:
- Fetching data from various sources (Yahoo Finance, Alpha Vantage, IBKR)
- Data cleaning and validation
- Feature engineering
"""

from .data_loader import (
    get_data,
    DataLoader,
    check_data_quality,
    detect_anomalies
)
from .feature_engineering import FeatureEngine
from .preprocessing import DataPreprocessor
from .sequence_generator import (
    SequenceDataset,
    create_sequences,
    prepare_data_for_lstm,
    verify_no_lookahead
)

__all__ = [
    'get_data',
    'DataLoader',
    'check_data_quality',
    'detect_anomalies',
    'FeatureEngine',
    'DataPreprocessor',
    'SequenceDataset',
    'create_sequences',
    'prepare_data_for_lstm',
    'verify_no_lookahead',
]

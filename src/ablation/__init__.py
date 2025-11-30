"""
Ablation Study Framework for RL Trading Strategies.

This module provides systematic comparison of different feature configurations
to answer the research question: Does regime/sentiment information improve
trading performance over price-only models?
"""

from .study import AblationStudy, AblationConfig, AblationResult
from .statistical_tests import (
    compare_returns,
    paired_t_test,
    bootstrap_confidence_interval,
    calculate_effect_size
)

__all__ = [
    'AblationStudy',
    'AblationConfig',
    'AblationResult',
    'compare_returns',
    'paired_t_test',
    'bootstrap_confidence_interval',
    'calculate_effect_size'
]

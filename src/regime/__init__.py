"""
Regime detection module for macro-aware trading.

This module provides tools for identifying market regimes using:
1. Hidden Markov Models (HMM) on macroeconomic indicators
2. Rule-based classification using economic thresholds
3. Regime-conditional analysis utilities
"""

from .hmm_detector import HMMRegimeDetector
from .rule_based_detector import RuleBasedRegimeDetector
from .regime_analysis import (
    compute_regime_correlations,
    compute_regime_statistics,
    analyze_regime_transitions,
    plot_regime_overlay,
    plot_regime_comparison,
    plot_regime_correlation_heatmap
)

__all__ = [
    'HMMRegimeDetector',
    'RuleBasedRegimeDetector',
    'compute_regime_correlations',
    'compute_regime_statistics',
    'analyze_regime_transitions',
    'plot_regime_overlay',
    'plot_regime_comparison',
    'plot_regime_correlation_heatmap'
]

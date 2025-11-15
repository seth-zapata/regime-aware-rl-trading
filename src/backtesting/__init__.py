"""
Backtesting framework module.

Contains:
- Core backtest engine
- Performance metrics calculation
- Result visualization
"""

from .backtest import Backtest
from .metrics import PerformanceMetrics
from .visualization import plot_equity_curve, plot_returns, plot_drawdown

__all__ = [
    'Backtest',
    'PerformanceMetrics',
    'plot_equity_curve',
    'plot_returns',
    'plot_drawdown',
]

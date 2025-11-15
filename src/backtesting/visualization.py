"""
Visualization utilities for backtesting results.

Create plots for:
- Equity curves
- Drawdowns
- Returns distribution
- Trade analysis
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


def plot_equity_curve(
    results: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    title: str = "Portfolio Equity Curve",
    figsize: tuple = (12, 6)
):
    """
    Plot portfolio value over time.

    Args:
        results: Backtest results DataFrame with 'portfolio_value'
        benchmark: Optional benchmark series for comparison
        title: Plot title
        figsize: Figure size
    """
    # TODO: Implement equity curve plotting
    # 1. Create figure
    # 2. Plot portfolio_value over time
    # 3. If benchmark provided, normalize and plot for comparison
    # 4. Add labels, legend, grid
    # 5. Show or return figure

    fig, ax = plt.subplots(figsize=figsize)

    # Plot strategy equity
    ax.plot(results.index, results['portfolio_value'], label='Strategy', linewidth=2)

    # Plot benchmark if provided
    if benchmark is not None:
        # Normalize benchmark to same starting value
        norm_benchmark = benchmark / benchmark.iloc[0] * results['portfolio_value'].iloc[0]
        ax.plot(results.index, norm_benchmark, label='Benchmark', linewidth=2, alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.show()  # Uncomment to display
    return fig


def plot_returns(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Returns Distribution",
    figsize: tuple = (12, 8)
):
    """
    Plot returns distribution and histogram.

    Args:
        returns: Strategy returns series
        benchmark_returns: Optional benchmark returns for comparison
        title: Plot title
        figsize: Figure size
    """
    # TODO: Implement returns plotting
    # Create 2 subplots:
    # 1. Cumulative returns over time
    # 2. Histogram of returns distribution

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    axes[0].plot(returns.index, cum_returns, label='Strategy')

    if benchmark_returns is not None:
        cum_bench = (1 + benchmark_returns).cumprod()
        axes[0].plot(benchmark_returns.index, cum_bench, label='Benchmark', alpha=0.7)

    axes[0].set_ylabel('Cumulative Return')
    axes[0].set_title('Cumulative Returns Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Returns distribution
    axes[1].hist(returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
    axes[1].axvline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Return')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Returns Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_drawdown(
    results: pd.DataFrame,
    title: str = "Drawdown Over Time",
    figsize: tuple = (12, 5)
):
    """
    Plot drawdown series.

    Args:
        results: Backtest results with 'portfolio_value'
        title: Plot title
        figsize: Figure size
    """
    # TODO: Implement drawdown plotting
    # 1. Calculate drawdown series
    # 2. Plot as area chart (filled)
    # 3. Highlight max drawdown point

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate drawdown
    portfolio_value = results['portfolio_value']
    running_max = portfolio_value.expanding().max()
    drawdown = (portfolio_value - running_max) / running_max

    # Plot drawdown
    ax.fill_between(results.index, 0, drawdown, alpha=0.3, color='red')
    ax.plot(results.index, drawdown, color='red', linewidth=1)

    # Highlight max drawdown
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax.axhline(max_dd, color='darkred', linestyle='--',
               label=f'Max Drawdown: {max_dd*100:.2f}%')
    ax.plot(max_dd_date, max_dd, 'ro', markersize=8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))

    plt.tight_layout()
    return fig


def plot_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns Heatmap",
    figsize: tuple = (12, 6)
):
    """
    Plot monthly returns as a heatmap.

    Args:
        returns: Daily returns series
        title: Plot title
        figsize: Figure size
    """
    # TODO: Implement monthly returns heatmap
    # 1. Resample returns to monthly
    # 2. Reshape into year x month grid
    # 3. Plot as heatmap with color scale
    pass


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    title: str = "Rolling Sharpe Ratio",
    figsize: tuple = (12, 5)
):
    """
    Plot rolling Sharpe ratio over time.

    Args:
        returns: Returns series
        window: Rolling window size (in periods)
        title: Plot title
        figsize: Figure size
    """
    # TODO: Implement rolling Sharpe plotting
    # 1. Calculate rolling mean and std
    # 2. Compute rolling Sharpe = rolling_mean / rolling_std * sqrt(252)
    # 3. Plot over time
    pass


def plot_strategy_comparison(
    results_dict: Dict[str, pd.DataFrame],
    title: str = "Strategy Comparison",
    figsize: tuple = (14, 7)
):
    """
    Plot multiple strategies for comparison.

    Args:
        results_dict: Dict mapping strategy names to results DataFrames
        title: Plot title
        figsize: Figure size
    """
    # TODO: Implement multi-strategy comparison
    # Plot normalized equity curves for all strategies on same axes
    # Normalize all to start at same value for fair comparison

    fig, ax = plt.subplots(figsize=figsize)

    for name, results in results_dict.items():
        # Normalize to start at 100
        normalized = results['portfolio_value'] / results['portfolio_value'].iloc[0] * 100
        ax.plot(results.index, normalized, label=name, linewidth=2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (Normalized to 100)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_performance_dashboard(
    results: pd.DataFrame,
    metrics: Dict[str, float],
    benchmark: Optional[pd.Series] = None,
    figsize: tuple = (16, 10)
):
    """
    Create comprehensive performance dashboard with multiple subplots.

    Args:
        results: Backtest results
        metrics: Performance metrics dictionary
        benchmark: Optional benchmark series
        figsize: Figure size
    """
    # TODO: Implement dashboard with multiple subplots:
    # - Equity curve
    # - Drawdown
    # - Returns distribution
    # - Monthly returns heatmap
    # - Metrics table (as text)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # TODO: Add each subplot
    # ax1 = fig.add_subplot(gs[0, :])  # Equity curve (full width)
    # ax2 = fig.add_subplot(gs[1, :])  # Drawdown (full width)
    # ax3 = fig.add_subplot(gs[2, 0])  # Returns histogram
    # ax4 = fig.add_subplot(gs[2, 1])  # Metrics table

    plt.suptitle('Performance Dashboard', fontsize=16, fontweight='bold')
    return fig


# TODO: Additional visualization functions
# - plot_trade_distribution(): Analyze trade durations, P&L
# - plot_correlation_with_benchmark()
# - plot_rolling_beta()
# - plot_underwater_plot(): Alternative drawdown visualization

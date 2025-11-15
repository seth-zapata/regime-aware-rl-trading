"""
Performance metrics calculation for backtesting.

Key metrics:
- Total return, annualized return
- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Win rate, profit factor
- Volatility
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class PerformanceMetrics:
    """
    Calculate trading strategy performance metrics.
    """

    def __init__(
        self,
        returns: pd.Series,
        portfolio_values: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ):
        """
        Initialize metrics calculator.

        Args:
            returns: Series of period returns
            portfolio_values: Series of portfolio values over time
            risk_free_rate: Annual risk-free rate (default 0)
            periods_per_year: Number of periods per year (252 for daily)
        """
        self.returns = returns.dropna()
        self.portfolio_values = portfolio_values
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def total_return(self) -> float:
        """
        Calculate total cumulative return.

        Returns:
            Total return as decimal (e.g., 0.5 = 50% return)
        """
        initial_value = self.portfolio_values.iloc[0]
        final_value = self.portfolio_values.iloc[-1]
        return (final_value - initial_value) / initial_value

    def annualized_return(self) -> float:
        """
        Calculate annualized return (CAGR).

        Returns:
            Annualized return as decimal
        """
        total_ret = self.total_return()
        years = len(self.returns) / self.periods_per_year

        if years == 0:
            return 0.0

        # CAGR = (1 + total_return)^(1/years) - 1
        return (1 + total_ret) ** (1 / years) - 1

    def volatility(self, annualized: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns).

        Args:
            annualized: If True, annualize the volatility

        Returns:
            Volatility as decimal
        """
        vol = self.returns.std()

        if annualized:
            vol *= np.sqrt(self.periods_per_year)

        return vol

    def sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio.

        Sharpe = (Return - Risk_Free_Rate) / Volatility

        Returns:
            Sharpe ratio
        """
        ann_return = self.annualized_return()
        ann_volatility = self.volatility(annualized=True)

        if ann_volatility == 0:
            return 0.0

        return (ann_return - self.risk_free_rate) / ann_volatility

    def sortino_ratio(self, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (like Sharpe but only downside risk).

        Args:
            target_return: Target or minimum acceptable return

        Returns:
            Sortino ratio
        """
        excess_return = self.annualized_return() - target_return

        # Calculate downside deviation (only negative returns)
        downside_returns = self.returns[self.returns < target_return]

        if len(downside_returns) == 0:
            return 0.0

        downside_deviation = downside_returns.std() * np.sqrt(self.periods_per_year)

        if downside_deviation == 0:
            return 0.0

        return excess_return / downside_deviation

    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.

        Max drawdown = max peak-to-trough decline in portfolio value.

        Returns:
            Maximum drawdown as decimal (positive number)
        """
        cumulative_max = self.portfolio_values.cummax()
        drawdown = (cumulative_max - self.portfolio_values) / cumulative_max
        return drawdown.max()

    def drawdown_series(self) -> pd.Series:
        """
        Calculate drawdown series over time.

        Returns:
            Series with drawdown at each point
        """
        cumulative_max = self.portfolio_values.cummax()
        drawdown = (cumulative_max - self.portfolio_values) / cumulative_max
        return drawdown

    def calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio.

        Calmar = Annualized Return / Max Drawdown

        Returns:
            Calmar ratio
        """
        ann_return = self.annualized_return()
        max_dd = self.max_drawdown()

        if max_dd == 0:
            return 0.0

        return ann_return / max_dd

    def win_rate(self, trades: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate win rate (fraction of profitable trades).

        Args:
            trades: DataFrame with individual trades
                   Must have 'pnl' or 'return_pct' column

        Returns:
            Win rate as decimal (e.g., 0.6 = 60% winners)
        """
        if trades is None or len(trades) == 0:
            return 0.0

        # Use pnl column if available, otherwise return_pct
        if 'pnl' in trades.columns:
            profitable = (trades['pnl'] > 0).sum()
        elif 'return_pct' in trades.columns:
            profitable = (trades['return_pct'] > 0).sum()
        else:
            return 0.0

        return profitable / len(trades)

    def profit_factor(self, trades: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate profit factor.

        Profit Factor = Total Profits / Total Losses

        Args:
            trades: DataFrame with individual trades

        Returns:
            Profit factor (> 1 means profitable overall)
        """
        if trades is None or len(trades) == 0:
            return 0.0

        # Use pnl column
        if 'pnl' not in trades.columns:
            return 0.0

        profits = trades[trades['pnl'] > 0]['pnl'].sum()
        losses = abs(trades[trades['pnl'] < 0]['pnl'].sum())

        if losses == 0:
            return np.inf if profits > 0 else 0.0

        return profits / losses

    def calculate_all_metrics(self, trades: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate all available metrics.

        Args:
            trades: Optional DataFrame with individual trades

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'total_return': self.total_return(),
            'annualized_return': self.annualized_return(),
            'volatility': self.volatility(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'max_drawdown': self.max_drawdown(),
            'calmar_ratio': self.calmar_ratio(),
        }

        # Add trade-based metrics if trades provided
        if trades is not None and len(trades) > 0:
            metrics['win_rate'] = self.win_rate(trades)
            metrics['profit_factor'] = self.profit_factor(trades)
            metrics['num_trades'] = len(trades)

        return metrics


def compare_strategies(
    results_dict: Dict[str, pd.DataFrame],
    risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    Compare multiple strategies side-by-side.

    Args:
        results_dict: Dictionary mapping strategy names to backtest results DataFrames
        risk_free_rate: Risk-free rate for Sharpe calculation

    Returns:
        DataFrame with metrics for each strategy (strategies as columns)
    """
    comparison = {}

    for strategy_name, results in results_dict.items():
        # Calculate metrics for this strategy
        metrics_calc = PerformanceMetrics(
            returns=results['returns'],
            portfolio_values=results['portfolio_value'],
            risk_free_rate=risk_free_rate
        )

        metrics = metrics_calc.calculate_all_metrics()
        comparison[strategy_name] = metrics

    # Convert to DataFrame with strategies as columns
    comparison_df = pd.DataFrame(comparison)

    return comparison_df


def print_metrics_report(metrics: Dict[str, float]):
    """
    Print formatted metrics report.

    Args:
        metrics: Dictionary of metrics
    """
    # TODO: Implement formatted printing
    # Print in nice format with appropriate decimal places
    print("=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    for name, value in metrics.items():
        if 'ratio' in name or 'rate' in name:
            print(f"{name:.<30} {value:.3f}")
        elif 'return' in name or 'drawdown' in name or 'volatility' in name:
            print(f"{name:.<30} {value*100:.2f}%")
        else:
            print(f"{name:.<30} {value:.2f}")
    print("=" * 50)


# TODO: Additional metrics to consider
# - Alpha, Beta (requires benchmark)
# - Information Ratio
# - Ulcer Index
# - Value at Risk (VaR)
# - Conditional VaR (CVaR)
# - R-squared vs benchmark

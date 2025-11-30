"""
Walk-Forward Validation Framework for RL Trading Agents.

Walk-forward validation is critical for time-series models because:
1. Standard cross-validation causes look-ahead bias
2. Markets are non-stationary - past performance doesn't guarantee future results
3. We need to simulate realistic model retraining schedules

This module implements:
- Rolling window walk-forward validation
- Anchored (expanding window) walk-forward validation
- Statistical aggregation of results across windows
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .ppo_trader import PPOTrader


@dataclass
class WalkForwardWindow:
    """Container for a single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass
class WalkForwardResult:
    """Container for results from a single window."""
    window_id: int
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    model: Optional[PPOTrader] = None


class WalkForwardValidator:
    """
    Walk-forward validation for RL trading strategies.

    This class handles:
    - Splitting data into train/test windows
    - Training a fresh model on each window
    - Aggregating metrics with proper statistics
    - Comparing against baselines (buy & hold)

    Attributes:
        df: Full dataset
        feature_columns: Feature column names
        n_windows: Number of validation windows
        train_pct: Fraction of each window for training

    Example:
        >>> validator = WalkForwardValidator(df, feature_columns, n_windows=5)
        >>> results = validator.run(total_timesteps=50000)
        >>> print(results['summary'])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        regime_column: Optional[str] = None,
        n_regimes: int = 3,
        price_column: str = 'Close',
        n_windows: int = 5,
        train_pct: float = 0.7,
        min_train_days: int = 252,
        min_test_days: int = 63,
        gap_days: int = 0,
        anchored: bool = False,
        verbose: int = 1
    ):
        """
        Initialize walk-forward validator.

        Args:
            df: Full dataset with DatetimeIndex
            feature_columns: List of feature column names
            regime_column: Optional regime column
            n_regimes: Number of regimes for encoding
            price_column: Price column name
            n_windows: Number of walk-forward windows
            train_pct: Fraction of each window for training
            min_train_days: Minimum training days per window
            min_test_days: Minimum test days per window
            gap_days: Gap between train and test (avoid leakage)
            anchored: If True, use expanding window (anchor at start)
            verbose: Verbosity level
        """
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.regime_column = regime_column
        self.n_regimes = n_regimes
        self.price_column = price_column
        self.n_windows = n_windows
        self.train_pct = train_pct
        self.min_train_days = min_train_days
        self.min_test_days = min_test_days
        self.gap_days = gap_days
        self.anchored = anchored
        self.verbose = verbose

        # Validate index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Generate windows
        self.windows = self._generate_windows()

    def _generate_windows(self) -> List[WalkForwardWindow]:
        """
        Generate walk-forward windows.

        Returns:
            List of WalkForwardWindow objects
        """
        windows = []
        total_days = len(self.df)

        if self.anchored:
            # Anchored: training always starts at beginning, test windows move forward
            test_size = (total_days - self.min_train_days - self.gap_days) // self.n_windows

            for i in range(self.n_windows):
                train_start_idx = 0
                train_end_idx = self.min_train_days + i * test_size
                test_start_idx = train_end_idx + self.gap_days
                test_end_idx = min(test_start_idx + test_size, total_days)

                if test_end_idx - test_start_idx < self.min_test_days:
                    continue

                windows.append(self._create_window(
                    i, train_start_idx, train_end_idx, test_start_idx, test_end_idx
                ))
        else:
            # Rolling: fixed window size moves through time
            window_size = total_days // self.n_windows

            for i in range(self.n_windows):
                window_start = i * window_size
                window_end = min((i + 1) * window_size + window_size // 2, total_days)

                # Split into train/test within this window
                train_size = int((window_end - window_start) * self.train_pct)
                train_start_idx = window_start
                train_end_idx = window_start + train_size
                test_start_idx = train_end_idx + self.gap_days
                test_end_idx = window_end

                if train_end_idx - train_start_idx < self.min_train_days:
                    continue
                if test_end_idx - test_start_idx < self.min_test_days:
                    continue

                windows.append(self._create_window(
                    i, train_start_idx, train_end_idx, test_start_idx, test_end_idx
                ))

        if len(windows) == 0:
            raise ValueError(
                f"Could not generate any valid windows. "
                f"Data has {total_days} days, need at least {self.min_train_days + self.min_test_days}"
            )

        return windows

    def _create_window(
        self,
        window_id: int,
        train_start_idx: int,
        train_end_idx: int,
        test_start_idx: int,
        test_end_idx: int
    ) -> WalkForwardWindow:
        """Create a WalkForwardWindow object."""
        return WalkForwardWindow(
            window_id=window_id,
            train_start=self.df.index[train_start_idx],
            train_end=self.df.index[train_end_idx - 1],
            test_start=self.df.index[test_start_idx],
            test_end=self.df.index[test_end_idx - 1],
            train_df=self.df.iloc[train_start_idx:train_end_idx].copy(),
            test_df=self.df.iloc[test_start_idx:test_end_idx].copy()
        )

    def run(
        self,
        total_timesteps: int = 50000,
        ppo_params: Optional[Dict] = None,
        seeds: List[int] = [42],
        save_models: bool = False,
        model_dir: str = 'models/walk_forward'
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            total_timesteps: Training steps per window
            ppo_params: PPO hyperparameters
            seeds: Random seeds to use (multiple for robustness)
            save_models: Whether to save trained models
            model_dir: Directory for saving models

        Returns:
            Dictionary with results and summary statistics
        """
        all_results = []

        for window in self.windows:
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Window {window.window_id + 1}/{len(self.windows)}")
                print(f"Train: {window.train_start.date()} to {window.train_end.date()} "
                      f"({len(window.train_df)} days)")
                print(f"Test:  {window.test_start.date()} to {window.test_end.date()} "
                      f"({len(window.test_df)} days)")
                print('='*60)

            # Train and evaluate for each seed
            seed_results = []
            for seed in seeds:
                result = self._run_single_window(
                    window=window,
                    total_timesteps=total_timesteps,
                    ppo_params=ppo_params,
                    seed=seed,
                    save_model=save_models,
                    model_dir=model_dir
                )
                seed_results.append(result)

            # Average across seeds
            avg_result = self._average_seed_results(seed_results, window.window_id)
            all_results.append(avg_result)

        # Aggregate across windows
        summary = self._aggregate_results(all_results)

        # Calculate buy & hold baseline
        baseline = self._calculate_baseline()

        return {
            'window_results': all_results,
            'summary': summary,
            'baseline': baseline,
            'windows': self.windows,
            'n_windows': len(self.windows),
            'n_seeds': len(seeds),
        }

    def _run_single_window(
        self,
        window: WalkForwardWindow,
        total_timesteps: int,
        ppo_params: Optional[Dict],
        seed: int,
        save_model: bool,
        model_dir: str
    ) -> WalkForwardResult:
        """Run training and evaluation for a single window and seed."""
        # Create and train trader
        trader = PPOTrader(
            df=window.train_df,
            feature_columns=self.feature_columns,
            regime_column=self.regime_column,
            n_regimes=self.n_regimes,
            price_column=self.price_column,
            ppo_params=ppo_params,
            seed=seed,
            verbose=0
        )

        train_stats = trader.train(
            total_timesteps=total_timesteps,
            progress_bar=self.verbose > 0
        )

        # Evaluate on train set
        train_metrics = trader.evaluate(window.train_df)

        # Evaluate on test set
        test_metrics = trader.evaluate(window.test_df)

        if self.verbose > 0:
            print(f"  Seed {seed}: Train Return={train_metrics['total_return']*100:.1f}%, "
                  f"Test Return={test_metrics['total_return']*100:.1f}%")

        # Save model if requested
        if save_model:
            import os
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/window_{window.window_id}_seed_{seed}"
            trader.save(model_path)

        return WalkForwardResult(
            window_id=window.window_id,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            model=trader
        )

    def _average_seed_results(
        self,
        seed_results: List[WalkForwardResult],
        window_id: int
    ) -> WalkForwardResult:
        """Average results across seeds for a single window."""
        # Keys to average
        metric_keys = ['total_return', 'sharpe_ratio', 'max_drawdown', 'num_trades']

        avg_train = {}
        avg_test = {}

        for key in metric_keys:
            train_values = [r.train_metrics.get(key, 0) for r in seed_results]
            test_values = [r.test_metrics.get(key, 0) for r in seed_results]

            avg_train[key] = np.mean(train_values)
            avg_train[f'{key}_std'] = np.std(train_values)
            avg_test[key] = np.mean(test_values)
            avg_test[f'{key}_std'] = np.std(test_values)

        return WalkForwardResult(
            window_id=window_id,
            train_metrics=avg_train,
            test_metrics=avg_test,
            model=seed_results[0].model  # Keep first model
        )

    def _aggregate_results(
        self,
        results: List[WalkForwardResult]
    ) -> Dict[str, Any]:
        """Aggregate results across all windows."""
        test_returns = [r.test_metrics['total_return'] for r in results]
        test_sharpes = [r.test_metrics['sharpe_ratio'] for r in results]
        test_drawdowns = [r.test_metrics['max_drawdown'] for r in results]

        train_returns = [r.train_metrics['total_return'] for r in results]

        summary = {
            # Test metrics (out-of-sample)
            'test_mean_return': np.mean(test_returns),
            'test_std_return': np.std(test_returns),
            'test_mean_sharpe': np.mean(test_sharpes),
            'test_std_sharpe': np.std(test_sharpes),
            'test_mean_drawdown': np.mean(test_drawdowns),
            'test_win_rate': np.mean([r > 0 for r in test_returns]),

            # Train metrics (in-sample, for comparison)
            'train_mean_return': np.mean(train_returns),
            'train_std_return': np.std(train_returns),

            # Overfitting indicator
            'overfit_ratio': np.mean(train_returns) / (np.mean(test_returns) + 1e-8),

            # Per-window returns
            'per_window_returns': test_returns,
            'per_window_sharpes': test_sharpes,
        }

        # Statistical significance (t-test vs 0)
        if len(test_returns) >= 3:
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(test_returns, 0)
            summary['t_statistic'] = t_stat
            summary['p_value'] = p_value
            summary['significant_at_05'] = p_value < 0.05
        else:
            summary['t_statistic'] = None
            summary['p_value'] = None
            summary['significant_at_05'] = None

        return summary

    def _calculate_baseline(self) -> Dict[str, float]:
        """Calculate buy & hold baseline across all test windows."""
        baseline_returns = []

        for window in self.windows:
            start_price = window.test_df[self.price_column].iloc[0]
            end_price = window.test_df[self.price_column].iloc[-1]
            ret = (end_price - start_price) / start_price
            baseline_returns.append(ret)

        return {
            'mean_return': np.mean(baseline_returns),
            'std_return': np.std(baseline_returns),
            'per_window_returns': baseline_returns,
        }

    def print_summary(self, results: Dict[str, Any]):
        """Print formatted summary of walk-forward results."""
        summary = results['summary']
        baseline = results['baseline']

        print("\n" + "="*70)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("="*70)

        print(f"\nWindows: {results['n_windows']}, Seeds per window: {results['n_seeds']}")

        print(f"\n{'Metric':<30} {'Strategy':<15} {'Buy & Hold':<15}")
        print("-"*60)

        print(f"{'Mean Return':<30} {summary['test_mean_return']*100:>+.2f}%{'':<8} "
              f"{baseline['mean_return']*100:>+.2f}%")
        print(f"{'Std Return':<30} {summary['test_std_return']*100:>.2f}%{'':<9} "
              f"{baseline['std_return']*100:>.2f}%")
        print(f"{'Mean Sharpe':<30} {summary['test_mean_sharpe']:>.3f}")
        print(f"{'Mean Max Drawdown':<30} {summary['test_mean_drawdown']*100:>.2f}%")
        print(f"{'Win Rate (windows > 0)':<30} {summary['test_win_rate']*100:.1f}%")

        if summary['p_value'] is not None:
            print(f"\n{'Statistical Significance':<30}")
            print(f"{'  t-statistic':<30} {summary['t_statistic']:.3f}")
            print(f"{'  p-value':<30} {summary['p_value']:.4f}")
            print(f"{'  Significant at 5%?':<30} {summary['significant_at_05']}")

        print(f"\n{'Overfitting Check':<30}")
        print(f"{'  Train Mean Return':<30} {summary['train_mean_return']*100:>+.2f}%")
        print(f"{'  Test Mean Return':<30} {summary['test_mean_return']*100:>+.2f}%")
        print(f"{'  Overfit Ratio':<30} {summary['overfit_ratio']:.2f}x")

        print("\n" + "="*70)


def compare_strategies(
    df: pd.DataFrame,
    feature_columns: List[str],
    configs: Dict[str, Dict],
    n_windows: int = 5,
    total_timesteps: int = 50000,
    **kwargs
) -> pd.DataFrame:
    """
    Compare multiple strategy configurations using walk-forward validation.

    Args:
        df: Full dataset
        feature_columns: Feature columns
        configs: Dictionary mapping strategy names to their config dicts
        n_windows: Number of walk-forward windows
        total_timesteps: Training steps
        **kwargs: Additional arguments for WalkForwardValidator

    Returns:
        DataFrame comparing strategies
    """
    results = {}

    for name, config in configs.items():
        print(f"\n{'#'*60}")
        print(f"# Strategy: {name}")
        print('#'*60)

        validator = WalkForwardValidator(
            df=df,
            feature_columns=feature_columns,
            n_windows=n_windows,
            **{**kwargs, **config}
        )

        result = validator.run(total_timesteps=total_timesteps)
        results[name] = result['summary']

    # Convert to DataFrame
    comparison = pd.DataFrame(results).T
    comparison = comparison[['test_mean_return', 'test_std_return',
                            'test_mean_sharpe', 'test_mean_drawdown',
                            'test_win_rate']]

    return comparison

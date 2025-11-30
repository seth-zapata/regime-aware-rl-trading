"""
Ablation Study Framework for systematic feature comparison.

This module runs controlled experiments comparing different feature
configurations to isolate the contribution of each component.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from rl.ppo_trader import PPOTrader
from rl.walk_forward import WalkForwardValidator
from ablation.statistical_tests import compare_returns, bootstrap_confidence_interval


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    feature_columns: List[str]
    use_regime: bool = False
    regime_column: Optional[str] = None
    n_regimes: int = 3
    description: str = ""

    def __post_init__(self):
        if self.use_regime and self.regime_column is None:
            raise ValueError("regime_column required when use_regime=True")


@dataclass
class AblationResult:
    """Results from a single ablation configuration."""
    config: AblationConfig
    train_returns: List[float]
    test_returns: List[float]
    train_sharpes: List[float]
    test_sharpes: List[float]
    test_drawdowns: List[float]
    num_trades: List[float]
    training_time_sec: float

    @property
    def mean_test_return(self) -> float:
        return np.mean(self.test_returns)

    @property
    def std_test_return(self) -> float:
        return np.std(self.test_returns)

    @property
    def mean_test_sharpe(self) -> float:
        return np.mean(self.test_sharpes)

    @property
    def mean_test_drawdown(self) -> float:
        return np.mean(self.test_drawdowns)

    @property
    def win_rate(self) -> float:
        """Fraction of windows with positive returns."""
        return np.mean([r > 0 for r in self.test_returns])

    def summary_dict(self) -> Dict[str, Any]:
        """Return summary statistics as dictionary."""
        return {
            'name': self.config.name,
            'n_features': len(self.config.feature_columns),
            'use_regime': self.config.use_regime,
            'mean_test_return': self.mean_test_return,
            'std_test_return': self.std_test_return,
            'mean_test_sharpe': self.mean_test_sharpe,
            'mean_test_drawdown': self.mean_test_drawdown,
            'win_rate': self.win_rate,
            'training_time_sec': self.training_time_sec,
        }


class AblationStudy:
    """
    Run systematic ablation studies comparing feature configurations.

    This class manages:
    - Multiple configuration comparisons
    - Walk-forward validation for each config
    - Statistical significance testing
    - Result aggregation and reporting

    Example:
        >>> study = AblationStudy(df, price_column='price_Close')
        >>> study.add_config(AblationConfig(
        ...     name='price_only',
        ...     feature_columns=price_features,
        ...     use_regime=False
        ... ))
        >>> study.add_config(AblationConfig(
        ...     name='price_regime',
        ...     feature_columns=price_features,
        ...     use_regime=True,
        ...     regime_column='regime_numeric'
        ... ))
        >>> results = study.run(total_timesteps=50000, n_windows=3, seeds=[42, 123])
        >>> study.print_comparison()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        price_column: str = 'price_Close',
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        verbose: int = 1
    ):
        """
        Initialize ablation study.

        Args:
            df: Full dataset with DatetimeIndex
            price_column: Column name for prices
            initial_balance: Starting capital for each run
            transaction_cost: Trading cost as fraction
            verbose: Verbosity level
        """
        self.df = df.copy()
        self.price_column = price_column
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.verbose = verbose

        self.configs: List[AblationConfig] = []
        self.results: List[AblationResult] = []
        self.baseline_returns: Optional[List[float]] = None

    def add_config(self, config: AblationConfig):
        """Add a configuration to test."""
        # Validate features exist
        missing = [f for f in config.feature_columns if f not in self.df.columns]
        if missing:
            raise ValueError(f"Missing features in config '{config.name}': {missing}")

        if config.use_regime and config.regime_column not in self.df.columns:
            raise ValueError(f"Missing regime column: {config.regime_column}")

        self.configs.append(config)

    def run(
        self,
        total_timesteps: int = 50000,
        n_windows: int = 3,
        seeds: List[int] = [42],
        train_pct: float = 0.7,
        min_train_days: int = 200,
        min_test_days: int = 50,
    ) -> Dict[str, Any]:
        """
        Run ablation study across all configurations.

        Args:
            total_timesteps: Training steps per window
            n_windows: Number of walk-forward windows
            seeds: Random seeds for each run
            train_pct: Fraction of each window for training
            min_train_days: Minimum training days
            min_test_days: Minimum test days

        Returns:
            Dictionary with results and comparisons
        """
        import time

        if len(self.configs) == 0:
            raise ValueError("No configurations added. Use add_config() first.")

        self.results = []

        for config in self.configs:
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Running: {config.name}")
                print(f"Features: {len(config.feature_columns)}, Regime: {config.use_regime}")
                print('='*60)

            start_time = time.time()

            # Prepare data - drop NaN for this config's features
            required_cols = config.feature_columns + [self.price_column]
            if config.use_regime:
                required_cols.append(config.regime_column)

            df_clean = self.df.dropna(subset=required_cols)

            # Run walk-forward validation
            validator = WalkForwardValidator(
                df=df_clean,
                feature_columns=config.feature_columns,
                regime_column=config.regime_column if config.use_regime else None,
                n_regimes=config.n_regimes,
                price_column=self.price_column,
                n_windows=n_windows,
                train_pct=train_pct,
                min_train_days=min_train_days,
                min_test_days=min_test_days,
                verbose=self.verbose
            )

            wf_results = validator.run(
                total_timesteps=total_timesteps,
                seeds=seeds,
                save_models=False
            )

            training_time = time.time() - start_time

            # Extract results
            train_returns = [r.train_metrics['total_return'] for r in wf_results['window_results']]
            test_returns = [r.test_metrics['total_return'] for r in wf_results['window_results']]
            train_sharpes = [r.train_metrics['sharpe_ratio'] for r in wf_results['window_results']]
            test_sharpes = [r.test_metrics['sharpe_ratio'] for r in wf_results['window_results']]
            test_drawdowns = [r.test_metrics['max_drawdown'] for r in wf_results['window_results']]
            num_trades = [r.test_metrics['num_trades'] for r in wf_results['window_results']]

            result = AblationResult(
                config=config,
                train_returns=train_returns,
                test_returns=test_returns,
                train_sharpes=train_sharpes,
                test_sharpes=test_sharpes,
                test_drawdowns=test_drawdowns,
                num_trades=num_trades,
                training_time_sec=training_time
            )

            self.results.append(result)

            # Store baseline (buy & hold) returns
            if self.baseline_returns is None:
                self.baseline_returns = wf_results['baseline']['per_window_returns']

            if self.verbose > 0:
                print(f"\nResults for {config.name}:")
                print(f"  Mean Test Return: {result.mean_test_return*100:+.2f}%")
                print(f"  Mean Test Sharpe: {result.mean_test_sharpe:.3f}")
                print(f"  Win Rate: {result.win_rate*100:.1f}%")
                print(f"  Training Time: {training_time:.1f}s")

        # Generate comparisons
        comparisons = self._generate_comparisons()

        return {
            'results': self.results,
            'comparisons': comparisons,
            'baseline_returns': self.baseline_returns,
            'n_windows': n_windows,
            'n_seeds': len(seeds),
            'total_timesteps': total_timesteps,
        }

    def _generate_comparisons(self) -> Dict[str, Any]:
        """Generate pairwise statistical comparisons."""
        comparisons = {}

        # Compare each config to baseline
        if self.baseline_returns is not None:
            for result in self.results:
                key = f"{result.config.name}_vs_baseline"
                comparisons[key] = compare_returns(
                    result.test_returns,
                    self.baseline_returns
                )

        # Compare configs to each other (first config as reference)
        if len(self.results) >= 2:
            reference = self.results[0]
            for result in self.results[1:]:
                key = f"{result.config.name}_vs_{reference.config.name}"
                comparisons[key] = compare_returns(
                    result.test_returns,
                    reference.test_returns
                )

        return comparisons

    def get_summary_df(self) -> pd.DataFrame:
        """Get summary DataFrame of all results."""
        summaries = [r.summary_dict() for r in self.results]

        # Add baseline
        if self.baseline_returns is not None:
            summaries.append({
                'name': 'Buy & Hold',
                'n_features': 0,
                'use_regime': False,
                'mean_test_return': np.mean(self.baseline_returns),
                'std_test_return': np.std(self.baseline_returns),
                'mean_test_sharpe': np.nan,
                'mean_test_drawdown': np.nan,
                'win_rate': np.mean([r > 0 for r in self.baseline_returns]),
                'training_time_sec': 0,
            })

        return pd.DataFrame(summaries)

    def print_comparison(self):
        """Print formatted comparison of results."""
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80)

        # Summary table
        df = self.get_summary_df()
        print("\n--- Performance Summary ---\n")
        print(f"{'Configuration':<20} {'Test Return':>12} {'Sharpe':>10} {'Drawdown':>10} {'Win Rate':>10}")
        print("-"*65)

        for _, row in df.iterrows():
            sharpe_str = f"{row['mean_test_sharpe']:.3f}" if not np.isnan(row['mean_test_sharpe']) else "N/A"
            dd_str = f"{row['mean_test_drawdown']*100:.1f}%" if not np.isnan(row['mean_test_drawdown']) else "N/A"

            print(f"{row['name']:<20} {row['mean_test_return']*100:>+11.2f}% {sharpe_str:>10} {dd_str:>10} {row['win_rate']*100:>9.1f}%")

        # Statistical comparisons
        if len(self.results) >= 2:
            print("\n--- Statistical Comparisons ---\n")

            reference = self.results[0]
            print(f"Reference: {reference.config.name}")
            print()

            for result in self.results[1:]:
                comparison = compare_returns(result.test_returns, reference.test_returns)

                print(f"{result.config.name} vs {reference.config.name}:")
                print(f"  Mean difference: {comparison['mean_diff']*100:+.2f}%")
                print(f"  Effect size: {comparison['effect_size']} (Cohen's d = {comparison['cohens_d']:.3f})")

                if 'paired_p_value' in comparison:
                    sig = "Yes" if comparison['paired_significant'] else "No"
                    print(f"  Paired t-test: p={comparison['paired_p_value']:.4f} (Significant: {sig})")
                print()

        # Key findings
        print("--- Key Findings ---\n")

        # Best performer
        best_idx = np.argmax([r.mean_test_return for r in self.results])
        best = self.results[best_idx]
        print(f"Best test return: {best.config.name} ({best.mean_test_return*100:+.2f}%)")

        # Regime effect (if applicable)
        no_regime = [r for r in self.results if not r.config.use_regime]
        with_regime = [r for r in self.results if r.config.use_regime]

        if no_regime and with_regime:
            avg_no_regime = np.mean([r.mean_test_return for r in no_regime])
            avg_with_regime = np.mean([r.mean_test_return for r in with_regime])
            regime_effect = avg_with_regime - avg_no_regime

            print(f"Average regime effect: {regime_effect*100:+.2f}%")
            if regime_effect > 0:
                print("  -> Regime information appears HELPFUL")
            else:
                print("  -> Regime information appears NOT HELPFUL (or harmful)")

        print("\n" + "="*80)


def create_standard_configs(
    df: pd.DataFrame,
    price_features: List[str],
    macro_features: List[str],
    sentiment_features: Optional[List[str]] = None,
    regime_column: str = 'regime_numeric'
) -> List[AblationConfig]:
    """
    Create standard ablation configurations for comparison.

    Args:
        df: Dataset to validate features against
        price_features: Price-based feature columns
        macro_features: Macro indicator feature columns
        sentiment_features: Sentiment feature columns (optional)
        regime_column: Regime label column

    Returns:
        List of AblationConfig objects
    """
    configs = []

    # Filter to available features
    available_price = [f for f in price_features if f in df.columns]
    available_macro = [f for f in macro_features if f in df.columns]
    available_sentiment = [f for f in (sentiment_features or []) if f in df.columns]

    # Config 1: Price only
    configs.append(AblationConfig(
        name='price_only',
        feature_columns=available_price,
        use_regime=False,
        description='Baseline: price features only'
    ))

    # Config 2: Price + Macro
    if available_macro:
        configs.append(AblationConfig(
            name='price_macro',
            feature_columns=available_price + available_macro,
            use_regime=False,
            description='Price features + macro indicators'
        ))

    # Config 3: Price + Regime
    if regime_column in df.columns:
        configs.append(AblationConfig(
            name='price_regime',
            feature_columns=available_price,
            use_regime=True,
            regime_column=regime_column,
            description='Price features + regime conditioning'
        ))

    # Config 4: Price + Macro + Regime
    if available_macro and regime_column in df.columns:
        configs.append(AblationConfig(
            name='price_macro_regime',
            feature_columns=available_price + available_macro,
            use_regime=True,
            regime_column=regime_column,
            description='Price + macro + regime conditioning'
        ))

    # Config 5: Full model (with sentiment if available)
    if available_sentiment and regime_column in df.columns:
        configs.append(AblationConfig(
            name='full_model',
            feature_columns=available_price + available_macro + available_sentiment,
            use_regime=True,
            regime_column=regime_column,
            description='All features + regime conditioning'
        ))

    return configs

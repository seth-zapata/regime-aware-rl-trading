"""
Regime-conditional analysis utilities.

This module provides tools for analyzing how feature-target relationships
vary across market regimes. The core hypothesis is that correlations
that are weak overall may be stronger within specific regimes.

Key analyses:
1. Regime-conditional correlations
2. Feature importance by regime
3. Regime transition analysis
4. Regime-aware train/test splitting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def compute_regime_correlations(
    features: pd.DataFrame,
    target: pd.Series,
    regimes: pd.Series,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Compute feature-target correlations within each regime.

    This is the core test of our hypothesis: do correlations become
    stronger when we condition on regime?

    Args:
        features: DataFrame of feature columns.
        target: Series of target values.
        regimes: Series of regime labels.
        method: Correlation method ('pearson', 'spearman', 'kendall').

    Returns:
        DataFrame with columns:
        - feature: Feature name
        - overall_corr: Correlation across all data
        - {regime}_corr: Correlation within each regime
        - max_regime_corr: Highest absolute regime correlation
        - regime_lift: max_regime_corr / abs(overall_corr)
    """
    results = []

    for col in features.columns:
        row = {'feature': col}

        # Overall correlation
        valid_mask = features[col].notna() & target.notna()
        if valid_mask.sum() > 10:
            overall = features[col][valid_mask].corr(target[valid_mask], method=method)
        else:
            overall = np.nan
        row['overall_corr'] = overall

        # Per-regime correlations
        max_abs_corr = 0
        for regime in regimes.unique():
            regime_mask = (regimes == regime) & valid_mask
            if regime_mask.sum() > 10:
                regime_corr = features[col][regime_mask].corr(
                    target[regime_mask], method=method
                )
            else:
                regime_corr = np.nan

            row[f'{regime}_corr'] = regime_corr
            if not np.isnan(regime_corr):
                max_abs_corr = max(max_abs_corr, abs(regime_corr))

        row['max_regime_corr'] = max_abs_corr

        # Regime lift: how much better is the best regime correlation?
        if not np.isnan(overall) and abs(overall) > 0.001:
            row['regime_lift'] = max_abs_corr / abs(overall)
        else:
            row['regime_lift'] = np.nan

        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values('max_regime_corr', ascending=False, key=abs)

    return df


def test_correlation_significance(
    features: pd.DataFrame,
    target: pd.Series,
    regimes: pd.Series,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test statistical significance of regime-conditional correlations.

    Uses Fisher's z-transformation to compare correlations across regimes.

    Args:
        features: DataFrame of feature columns.
        target: Series of target values.
        regimes: Series of regime labels.
        alpha: Significance level.

    Returns:
        DataFrame with correlation values and p-values.
    """
    results = []

    for col in features.columns:
        valid_mask = features[col].notna() & target.notna()
        row = {'feature': col}

        for regime in regimes.unique():
            regime_mask = (regimes == regime) & valid_mask
            n = regime_mask.sum()

            if n > 10:
                r, p_value = stats.pearsonr(
                    features[col][regime_mask],
                    target[regime_mask]
                )
                row[f'{regime}_corr'] = r
                row[f'{regime}_pvalue'] = p_value
                row[f'{regime}_n'] = n
                row[f'{regime}_significant'] = p_value < alpha
            else:
                row[f'{regime}_corr'] = np.nan
                row[f'{regime}_pvalue'] = np.nan
                row[f'{regime}_n'] = n
                row[f'{regime}_significant'] = False

        results.append(row)

    return pd.DataFrame(results)


def compute_regime_statistics(
    data: pd.DataFrame,
    regimes: pd.Series,
    target_col: str = 'target'
) -> pd.DataFrame:
    """
    Compute summary statistics for each regime.

    Args:
        data: DataFrame with features and target.
        regimes: Series of regime labels.
        target_col: Name of target column.

    Returns:
        DataFrame with regime statistics.
    """
    data = data.copy()
    data['regime'] = regimes

    stats_list = []
    for regime in regimes.unique():
        regime_data = data[data['regime'] == regime]

        stat = {
            'regime': regime,
            'count': len(regime_data),
            'proportion': len(regime_data) / len(data),
        }

        if target_col in regime_data.columns:
            target_vals = regime_data[target_col]
            stat['target_mean'] = target_vals.mean()
            stat['target_std'] = target_vals.std()
            stat['up_rate'] = (target_vals == 1).mean() if target_vals.dtype in ['int64', 'int32'] else np.nan

        stats_list.append(stat)

    return pd.DataFrame(stats_list).set_index('regime')


def analyze_regime_transitions(regimes: pd.Series) -> Dict:
    """
    Analyze regime transition patterns.

    Args:
        regimes: Series of regime labels with datetime index.

    Returns:
        Dictionary with transition analysis:
        - transition_matrix: Empirical transition probabilities
        - avg_duration: Average days spent in each regime
        - transition_counts: Raw count of each transition
    """
    # Compute transitions
    prev_regime = regimes.shift(1)
    transitions = pd.DataFrame({
        'from': prev_regime,
        'to': regimes
    }).dropna()

    # Transition matrix
    transition_counts = pd.crosstab(
        transitions['from'],
        transitions['to'],
        normalize='index'
    )

    # Average duration in each regime
    regime_runs = []
    current_regime = regimes.iloc[0]
    current_start = regimes.index[0]

    for date, regime in regimes.items():
        if regime != current_regime:
            duration = (date - current_start).days
            regime_runs.append({
                'regime': current_regime,
                'start': current_start,
                'end': date,
                'duration_days': duration
            })
            current_regime = regime
            current_start = date

    # Add final run
    regime_runs.append({
        'regime': current_regime,
        'start': current_start,
        'end': regimes.index[-1],
        'duration_days': (regimes.index[-1] - current_start).days
    })

    runs_df = pd.DataFrame(regime_runs)
    avg_duration = runs_df.groupby('regime')['duration_days'].mean()

    return {
        'transition_matrix': transition_counts,
        'avg_duration_days': avg_duration.to_dict(),
        'runs': runs_df,
        'raw_counts': pd.crosstab(transitions['from'], transitions['to'])
    }


def plot_regime_correlation_heatmap(
    regime_corr_df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot heatmap of feature correlations by regime.

    Args:
        regime_corr_df: Output from compute_regime_correlations().
        top_n: Number of top features to show.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    # Get correlation columns
    corr_cols = [c for c in regime_corr_df.columns if c.endswith('_corr') and c != 'overall_corr' and c != 'max_regime_corr']

    # Select top features
    df = regime_corr_df.head(top_n).copy()

    # Create heatmap data
    heatmap_data = df.set_index('feature')[['overall_corr'] + corr_cols]

    # Rename columns for display
    heatmap_data.columns = [c.replace('_corr', '').title() for c in heatmap_data.columns]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-0.3,
        vmax=0.3,
        ax=ax
    )

    ax.set_title('Feature-Target Correlations by Regime\n(Top features by max regime correlation)')
    ax.set_xlabel('Regime')
    ax.set_ylabel('Feature')

    plt.tight_layout()
    return fig


def plot_regime_overlay(
    price_data: pd.DataFrame,
    regimes: pd.Series,
    price_col: str = 'Close',
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot price chart with regime overlay.

    Args:
        price_data: DataFrame with price column.
        regimes: Series of regime labels.
        price_col: Name of price column.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for regimes
    regime_colors = {
        'Expansion': '#90EE90',   # Light green
        'Contraction': '#FFD700', # Gold/yellow
        'Crisis': '#FF6B6B',      # Light red
    }

    # Default colors for unknown regimes
    unique_regimes = regimes.unique()
    for regime in unique_regimes:
        if regime not in regime_colors:
            regime_colors[regime] = plt.cm.Set3(hash(regime) % 12)

    # Align data
    common_idx = price_data.index.intersection(regimes.index)
    prices = price_data.loc[common_idx, price_col]
    aligned_regimes = regimes.loc[common_idx]

    # Plot price
    ax.plot(prices.index, prices.values, 'k-', linewidth=1, alpha=0.8)

    # Shade regimes
    prev_regime = aligned_regimes.iloc[0]
    start_idx = 0

    for i, (date, regime) in enumerate(aligned_regimes.items()):
        if regime != prev_regime or i == len(aligned_regimes) - 1:
            # End of a regime block
            end_idx = i if regime != prev_regime else i + 1
            if end_idx > start_idx:
                ax.axvspan(
                    aligned_regimes.index[start_idx],
                    aligned_regimes.index[min(end_idx, len(aligned_regimes)-1)],
                    alpha=0.3,
                    color=regime_colors.get(prev_regime, 'gray'),
                    label=prev_regime if start_idx == 0 or prev_regime not in [aligned_regimes.iloc[j] for j in range(start_idx)] else ''
                )
            start_idx = i
            prev_regime = regime

    # Create legend (avoid duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    ax.set_title('Price with Regime Overlay')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')

    # Format x-axis
    fig.autofmt_xdate()
    plt.tight_layout()

    return fig


def plot_regime_comparison(
    rule_regimes: pd.Series,
    hmm_regimes: pd.Series,
    price_data: pd.DataFrame,
    price_col: str = 'Close',
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Compare rule-based and HMM regimes side by side.

    Args:
        rule_regimes: Series from rule-based detector.
        hmm_regimes: Series from HMM detector.
        price_data: DataFrame with price column.
        price_col: Name of price column.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Common index
    common_idx = price_data.index.intersection(rule_regimes.index).intersection(hmm_regimes.index)

    # Plot rule-based
    ax = axes[0]
    _plot_single_regime_overlay(ax, price_data.loc[common_idx], rule_regimes.loc[common_idx], price_col)
    ax.set_title('Rule-Based Regime Detection')
    ax.set_ylabel('Price ($)')

    # Plot HMM
    ax = axes[1]
    _plot_single_regime_overlay(ax, price_data.loc[common_idx], hmm_regimes.loc[common_idx], price_col)
    ax.set_title('HMM Regime Detection')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')

    fig.autofmt_xdate()
    plt.tight_layout()

    return fig


def _plot_single_regime_overlay(
    ax: plt.Axes,
    price_data: pd.DataFrame,
    regimes: pd.Series,
    price_col: str
) -> None:
    """Helper to plot regime overlay on given axes."""
    regime_colors = {
        'Expansion': '#90EE90',
        'Contraction': '#FFD700',
        'Crisis': '#FF6B6B',
    }

    # Add default colors for unknown regimes
    for regime in regimes.unique():
        if regime not in regime_colors:
            regime_colors[regime] = plt.cm.Set3(hash(regime) % 12)

    prices = price_data[price_col]
    ax.plot(prices.index, prices.values, 'k-', linewidth=1, alpha=0.8)

    # Shade regimes
    prev_regime = regimes.iloc[0]
    start_idx = 0

    for i, (date, regime) in enumerate(regimes.items()):
        if regime != prev_regime or i == len(regimes) - 1:
            end_idx = i if regime != prev_regime else i + 1
            if end_idx > start_idx:
                ax.axvspan(
                    regimes.index[start_idx],
                    regimes.index[min(end_idx, len(regimes)-1)],
                    alpha=0.3,
                    color=regime_colors.get(prev_regime, 'gray')
                )
            start_idx = i
            prev_regime = regime

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=regime_colors.get(r, 'gray'), alpha=0.3, label=r)
        for r in sorted(regimes.unique())
    ]
    ax.legend(handles=legend_handles, loc='upper left')

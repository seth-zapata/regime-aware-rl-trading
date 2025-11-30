"""
Statistical tests for comparing trading strategy performance.

These tests help determine whether observed differences are statistically
significant or could be due to random chance.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats


def compare_returns(
    returns_a: List[float],
    returns_b: List[float],
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Compare two sets of returns using multiple statistical tests.

    Args:
        returns_a: Returns from strategy A
        returns_b: Returns from strategy B
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    returns_a = np.array(returns_a)
    returns_b = np.array(returns_b)

    results = {
        'mean_a': np.mean(returns_a),
        'mean_b': np.mean(returns_b),
        'std_a': np.std(returns_a),
        'std_b': np.std(returns_b),
        'mean_diff': np.mean(returns_a) - np.mean(returns_b),
    }

    # Paired t-test (if same length - paired observations)
    if len(returns_a) == len(returns_b):
        t_stat, p_value = stats.ttest_rel(returns_a, returns_b)
        results['paired_t_stat'] = t_stat
        results['paired_p_value'] = p_value
        results['paired_significant'] = p_value < alpha

    # Independent t-test
    t_stat, p_value = stats.ttest_ind(returns_a, returns_b)
    results['ind_t_stat'] = t_stat
    results['ind_p_value'] = p_value
    results['ind_significant'] = p_value < alpha

    # Mann-Whitney U test (non-parametric alternative)
    u_stat, p_value = stats.mannwhitneyu(returns_a, returns_b, alternative='two-sided')
    results['mannwhitney_u'] = u_stat
    results['mannwhitney_p'] = p_value
    results['mannwhitney_significant'] = p_value < alpha

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(returns_a) + np.var(returns_b)) / 2)
    if pooled_std > 0:
        results['cohens_d'] = (np.mean(returns_a) - np.mean(returns_b)) / pooled_std
        # Interpret effect size
        d = abs(results['cohens_d'])
        if d < 0.2:
            results['effect_size'] = 'negligible'
        elif d < 0.5:
            results['effect_size'] = 'small'
        elif d < 0.8:
            results['effect_size'] = 'medium'
        else:
            results['effect_size'] = 'large'
    else:
        results['cohens_d'] = 0.0
        results['effect_size'] = 'negligible'

    return results


def paired_t_test(
    returns_a: List[float],
    returns_b: List[float],
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Perform paired t-test on returns.

    Args:
        returns_a: Returns from strategy A
        returns_b: Returns from strategy B
        alpha: Significance level

    Returns:
        t_statistic, p_value, is_significant
    """
    t_stat, p_value = stats.ttest_rel(returns_a, returns_b)
    return t_stat, p_value, p_value < alpha


def bootstrap_confidence_interval(
    returns: List[float],
    statistic: str = 'mean',
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.

    Args:
        returns: Return observations
        statistic: Which statistic ('mean', 'sharpe', 'median')
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    returns = np.array(returns)
    n = len(returns)

    # Calculate statistic function
    if statistic == 'mean':
        stat_func = np.mean
    elif statistic == 'median':
        stat_func = np.median
    elif statistic == 'sharpe':
        def stat_func(x):
            if np.std(x) == 0:
                return 0
            return np.mean(x) / np.std(x) * np.sqrt(252)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Point estimate
    point_estimate = stat_func(returns)

    # Bootstrap samples
    bootstrap_stats = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(returns, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))

    # Confidence interval (percentile method)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return point_estimate, lower, upper


def calculate_effect_size(
    returns_a: List[float],
    returns_b: List[float]
) -> Dict[str, float]:
    """
    Calculate multiple effect size measures.

    Args:
        returns_a: Returns from strategy A
        returns_b: Returns from strategy B

    Returns:
        Dictionary with effect size measures
    """
    returns_a = np.array(returns_a)
    returns_b = np.array(returns_b)

    # Cohen's d
    pooled_std = np.sqrt((np.var(returns_a) + np.var(returns_b)) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(returns_a) - np.mean(returns_b)) / pooled_std
    else:
        cohens_d = 0.0

    # Hedges' g (bias-corrected Cohen's d)
    n_a, n_b = len(returns_a), len(returns_b)
    correction = 1 - (3 / (4 * (n_a + n_b) - 9))
    hedges_g = cohens_d * correction

    # Common Language Effect Size (probability that A > B)
    # Using U statistic from Mann-Whitney
    if len(returns_a) > 0 and len(returns_b) > 0:
        u_stat, _ = stats.mannwhitneyu(returns_a, returns_b, alternative='two-sided')
        cles = u_stat / (len(returns_a) * len(returns_b))
    else:
        cles = 0.5

    return {
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'cles': cles,  # Probability that random draw from A > random draw from B
    }


def multiple_comparison_correction(
    p_values: List[float],
    method: str = 'bonferroni'
) -> List[float]:
    """
    Correct p-values for multiple comparisons.

    Args:
        p_values: List of p-values
        method: Correction method ('bonferroni', 'holm', 'fdr')

    Returns:
        Corrected p-values
    """
    p_values = np.array(p_values)
    n = len(p_values)

    if method == 'bonferroni':
        # Simple but conservative
        return np.minimum(p_values * n, 1.0).tolist()

    elif method == 'holm':
        # Step-down procedure (less conservative)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        corrected = np.zeros(n)

        for i, (idx, p) in enumerate(zip(sorted_idx, sorted_p)):
            corrected[idx] = min(p * (n - i), 1.0)

        # Enforce monotonicity
        for i in range(1, n):
            if corrected[sorted_idx[i]] < corrected[sorted_idx[i-1]]:
                corrected[sorted_idx[i]] = corrected[sorted_idx[i-1]]

        return corrected.tolist()

    elif method == 'fdr':
        # Benjamini-Hochberg False Discovery Rate
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        corrected = np.zeros(n)

        for i, (idx, p) in enumerate(zip(sorted_idx, sorted_p)):
            corrected[idx] = p * n / (i + 1)

        # Enforce monotonicity (from end)
        corrected_sorted = corrected[sorted_idx]
        for i in range(n - 2, -1, -1):
            if corrected_sorted[i] > corrected_sorted[i + 1]:
                corrected[sorted_idx[i]] = corrected[sorted_idx[i + 1]]

        return np.minimum(corrected, 1.0).tolist()

    else:
        raise ValueError(f"Unknown method: {method}")

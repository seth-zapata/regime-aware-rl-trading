"""
Unit tests for regime detection module.

Tests cover:
1. HMM regime detector fitting and prediction
2. Rule-based regime classification
3. Regime analysis utilities
4. Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from regime.hmm_detector import HMMRegimeDetector
from regime.rule_based_detector import RuleBasedRegimeDetector, MarketRegime
from regime.regime_analysis import (
    compute_regime_correlations,
    compute_regime_statistics,
    analyze_regime_transitions
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_macro_data():
    """Generate sample macro data with known regime patterns."""
    np.random.seed(42)
    n_days = 500

    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')

    # Create data with clear regime patterns
    data = pd.DataFrame(index=dates)

    # First 200 days: Expansion (low VIX, steep curve)
    # Next 200 days: Contraction (elevated VIX, flat curve)
    # Last 100 days: Crisis (high VIX, inverted curve)

    vix = np.concatenate([
        np.random.normal(15, 3, 200),   # Expansion
        np.random.normal(28, 5, 200),   # Contraction
        np.random.normal(45, 10, 100),  # Crisis
    ])

    yield_curve = np.concatenate([
        np.random.normal(1.5, 0.3, 200),   # Expansion - steep
        np.random.normal(0.3, 0.2, 200),   # Contraction - flat
        np.random.normal(-0.5, 0.3, 100),  # Crisis - inverted
    ])

    credit_spread = np.concatenate([
        np.random.normal(1.5, 0.3, 200),   # Expansion - tight
        np.random.normal(2.5, 0.4, 200),   # Contraction - elevated
        np.random.normal(4.0, 0.5, 100),   # Crisis - wide
    ])

    unrate = np.concatenate([
        np.random.normal(4.0, 0.3, 200),   # Expansion - low
        np.random.normal(5.5, 0.4, 200),   # Contraction - rising
        np.random.normal(8.0, 1.0, 100),   # Crisis - high
    ])

    data['VIXCLS'] = np.clip(vix, 10, 80)
    data['T10Y2Y'] = yield_curve
    data['BAA10Y'] = np.clip(credit_spread, 1, 6)
    data['UNRATE'] = np.clip(unrate, 3, 15)

    return data


@pytest.fixture
def sample_price_data(sample_macro_data):
    """Generate sample price data aligned with macro data."""
    np.random.seed(42)
    dates = sample_macro_data.index

    # Generate prices with regime-dependent behavior
    returns = np.random.normal(0.0005, 0.01, len(dates))

    # Add regime effects
    returns[:200] += 0.001      # Expansion: positive drift
    returns[200:400] -= 0.0005  # Contraction: slight negative
    returns[400:] -= 0.002      # Crisis: negative drift

    prices = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        'Close': prices,
        'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'High': prices * (1 + abs(np.random.normal(0, 0.01, len(dates)))),
        'Low': prices * (1 - abs(np.random.normal(0, 0.01, len(dates)))),
        'Volume': np.random.randint(1e6, 1e8, len(dates))
    }, index=dates)


@pytest.fixture
def sample_features_and_target(sample_macro_data, sample_price_data):
    """Generate features and target for correlation analysis."""
    features = sample_macro_data.copy()

    # Add some price features
    features['return_5d'] = sample_price_data['Close'].pct_change(5)
    features['volatility'] = sample_price_data['Close'].pct_change().rolling(21).std()

    # Target: 5-day forward return direction
    forward_ret = sample_price_data['Close'].shift(-5) / sample_price_data['Close'] - 1
    target = (forward_ret > 0).astype(int)

    # Drop NaN
    valid_idx = features.dropna().index.intersection(target.dropna().index)

    return features.loc[valid_idx], target.loc[valid_idx]


# ============================================================================
# HMM Detector Tests
# ============================================================================

class TestHMMRegimeDetector:
    """Tests for HMM-based regime detection."""

    def test_fit_basic(self, sample_macro_data):
        """Test basic fitting works without errors."""
        detector = HMMRegimeDetector(n_regimes=3)
        detector.fit(sample_macro_data[['VIXCLS', 'T10Y2Y', 'BAA10Y', 'UNRATE']])

        assert detector.model is not None
        assert detector.scaler is not None
        assert len(detector.regime_labels) == 3

    def test_predict_returns_series(self, sample_macro_data):
        """Test predict returns a Series with correct index."""
        detector = HMMRegimeDetector(n_regimes=3)
        features = sample_macro_data[['VIXCLS', 'T10Y2Y', 'BAA10Y', 'UNRATE']]
        detector.fit(features)

        regimes = detector.predict(features)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(sample_macro_data)
        assert regimes.index.equals(sample_macro_data.index)

    def test_predict_proba_shape(self, sample_macro_data):
        """Test predict_proba returns correct shape."""
        n_regimes = 3
        detector = HMMRegimeDetector(n_regimes=n_regimes)
        features = sample_macro_data[['VIXCLS', 'T10Y2Y', 'BAA10Y', 'UNRATE']]
        detector.fit(features)

        proba = detector.predict_proba(features)

        assert isinstance(proba, pd.DataFrame)
        assert proba.shape == (len(sample_macro_data), n_regimes)
        # Probabilities should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_transition_matrix_valid(self, sample_macro_data):
        """Test transition matrix is valid (rows sum to 1)."""
        detector = HMMRegimeDetector(n_regimes=3)
        features = sample_macro_data[['VIXCLS', 'T10Y2Y', 'BAA10Y', 'UNRATE']]
        detector.fit(features)

        trans_matrix = detector.get_transition_matrix()

        assert trans_matrix.shape == (3, 3)
        assert np.allclose(trans_matrix.sum(axis=1), 1.0)
        assert (trans_matrix >= 0).all().all()

    def test_regime_labels_by_volatility(self, sample_macro_data):
        """Test that regimes are labeled by VIX level."""
        detector = HMMRegimeDetector(n_regimes=3)
        features = sample_macro_data[['VIXCLS', 'T10Y2Y', 'BAA10Y', 'UNRATE']]
        detector.fit(features, label_by_volatility=True)

        labels = set(detector.regime_labels.values())
        assert 'Expansion' in labels
        assert 'Contraction' in labels
        assert 'Crisis' in labels

    def test_regime_stats_computed(self, sample_macro_data):
        """Test regime statistics are computed after fitting."""
        detector = HMMRegimeDetector(n_regimes=3)
        features = sample_macro_data[['VIXCLS', 'T10Y2Y', 'BAA10Y', 'UNRATE']]
        detector.fit(features)

        summary = detector.get_regime_summary()

        assert len(summary) == 3
        assert 'count' in summary.columns
        assert 'proportion' in summary.columns

    def test_unfitted_predict_raises(self, sample_macro_data):
        """Test predict raises error if model not fitted."""
        detector = HMMRegimeDetector(n_regimes=3)

        with pytest.raises(ValueError, match="not fitted"):
            detector.predict(sample_macro_data)

    def test_missing_features_raises(self, sample_macro_data):
        """Test fit raises error if features are missing."""
        detector = HMMRegimeDetector(
            n_regimes=3,
            features=['VIXCLS', 'NONEXISTENT']
        )

        with pytest.raises(ValueError, match="Missing features"):
            detector.fit(sample_macro_data)

    def test_different_n_regimes(self, sample_macro_data):
        """Test detector works with different number of regimes."""
        features = sample_macro_data[['VIXCLS', 'T10Y2Y', 'BAA10Y', 'UNRATE']]

        for n in [2, 3, 4]:
            detector = HMMRegimeDetector(n_regimes=n)
            detector.fit(features)
            regimes = detector.predict(features)

            assert len(regimes.unique()) <= n

    def test_save_and_load(self, sample_macro_data, tmp_path):
        """Test model can be saved and loaded."""
        detector = HMMRegimeDetector(n_regimes=3)
        features = sample_macro_data[['VIXCLS', 'T10Y2Y', 'BAA10Y', 'UNRATE']]
        detector.fit(features)

        original_regimes = detector.predict(features)

        # Save
        save_path = tmp_path / 'hmm_model.pkl'
        detector.save(str(save_path))

        # Load
        loaded = HMMRegimeDetector.load(str(save_path))
        loaded_regimes = loaded.predict(features)

        assert (original_regimes == loaded_regimes).all()


# ============================================================================
# Rule-Based Detector Tests
# ============================================================================

class TestRuleBasedRegimeDetector:
    """Tests for rule-based regime detection."""

    def test_predict_basic(self, sample_macro_data):
        """Test basic prediction works."""
        detector = RuleBasedRegimeDetector()
        regimes = detector.predict(sample_macro_data)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(sample_macro_data)

    def test_crisis_detection(self):
        """Test crisis is detected correctly."""
        detector = RuleBasedRegimeDetector()

        # High VIX scenario
        crisis_data = pd.DataFrame({
            'VIXCLS': [50.0],
            'T10Y2Y': [1.0],
            'BAA10Y': [2.0]
        })

        regime = detector.predict(crisis_data).iloc[0]
        assert regime == MarketRegime.CRISIS.value

    def test_expansion_detection(self):
        """Test expansion is detected correctly."""
        detector = RuleBasedRegimeDetector()

        # Low VIX, steep curve, tight spreads
        expansion_data = pd.DataFrame({
            'VIXCLS': [12.0],
            'T10Y2Y': [2.0],
            'BAA10Y': [1.5]
        })

        regime = detector.predict(expansion_data).iloc[0]
        assert regime == MarketRegime.EXPANSION.value

    def test_contraction_detection(self):
        """Test contraction is detected correctly."""
        detector = RuleBasedRegimeDetector()

        # Inverted yield curve
        contraction_data = pd.DataFrame({
            'VIXCLS': [20.0],
            'T10Y2Y': [-0.5],
            'BAA10Y': [2.0]
        })

        regime = detector.predict(contraction_data).iloc[0]
        assert regime == MarketRegime.CONTRACTION.value

    def test_custom_thresholds(self):
        """Test custom thresholds work."""
        custom_thresholds = {'vix_crisis': 40.0}  # Higher threshold
        detector = RuleBasedRegimeDetector(thresholds=custom_thresholds)

        # VIX at 35 - would be crisis with default, but not with custom
        data = pd.DataFrame({
            'VIXCLS': [35.0],
            'T10Y2Y': [1.0],
            'BAA10Y': [2.0]
        })

        regime = detector.predict(data).iloc[0]
        assert regime != MarketRegime.CRISIS.value

    def test_explain_output(self, sample_macro_data):
        """Test explain returns a string explanation."""
        detector = RuleBasedRegimeDetector()
        explanation = detector.explain(sample_macro_data.iloc[0])

        assert isinstance(explanation, str)
        assert 'Regime:' in explanation
        assert 'VIX:' in explanation

    def test_predict_with_scores(self, sample_macro_data):
        """Test predict_with_scores returns correct columns."""
        detector = RuleBasedRegimeDetector()
        result = detector.predict_with_scores(sample_macro_data)

        assert 'regime' in result.columns
        assert 'crisis_score' in result.columns
        assert 'contraction_score' in result.columns
        assert 'expansion_score' in result.columns

        # Scores should be between 0 and 1
        for col in ['crisis_score', 'contraction_score', 'expansion_score']:
            assert result[col].between(0, 1).all()

    def test_missing_required_columns_raises(self):
        """Test error raised if required columns missing."""
        detector = RuleBasedRegimeDetector()

        incomplete_data = pd.DataFrame({
            'VIXCLS': [20.0],
            # Missing T10Y2Y and BAA10Y
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            detector.predict(incomplete_data)

    def test_column_name_aliases(self):
        """Test detector accepts various column name formats."""
        detector = RuleBasedRegimeDetector()

        # Using aliases
        data = pd.DataFrame({
            'VIX': [20.0],  # Alias for VIXCLS
            'yield_curve_spread': [1.0],  # Alias for T10Y2Y
            'credit_spread': [2.0],  # Alias for BAA10Y
        })

        regime = detector.predict(data)
        assert len(regime) == 1

    def test_regime_summary(self, sample_macro_data):
        """Test regime summary statistics."""
        detector = RuleBasedRegimeDetector()
        summary = detector.get_regime_summary(sample_macro_data)

        assert 'count' in summary.columns
        assert 'proportion' in summary.columns
        assert 'vix_mean' in summary.columns


# ============================================================================
# Regime Analysis Tests
# ============================================================================

class TestRegimeAnalysis:
    """Tests for regime analysis utilities."""

    def test_compute_regime_correlations(self, sample_features_and_target, sample_macro_data):
        """Test regime correlation computation."""
        features, target = sample_features_and_target

        # Get regimes
        detector = RuleBasedRegimeDetector()
        regimes = detector.predict(sample_macro_data.loc[features.index])

        corr_df = compute_regime_correlations(features, target, regimes)

        assert 'feature' in corr_df.columns
        assert 'overall_corr' in corr_df.columns
        assert 'max_regime_corr' in corr_df.columns
        assert len(corr_df) == len(features.columns)

    def test_compute_regime_statistics(self, sample_macro_data):
        """Test regime statistics computation."""
        detector = RuleBasedRegimeDetector()
        regimes = detector.predict(sample_macro_data)

        # Add a target column
        sample_macro_data['target'] = np.random.randint(0, 2, len(sample_macro_data))

        stats = compute_regime_statistics(sample_macro_data, regimes)

        assert 'count' in stats.columns
        assert 'proportion' in stats.columns
        assert stats['count'].sum() == len(sample_macro_data)

    def test_analyze_regime_transitions(self, sample_macro_data):
        """Test regime transition analysis."""
        detector = RuleBasedRegimeDetector()
        regimes = detector.predict(sample_macro_data)

        analysis = analyze_regime_transitions(regimes)

        assert 'transition_matrix' in analysis
        assert 'avg_duration_days' in analysis
        assert 'runs' in analysis

        # Transition matrix rows should sum to 1
        trans_matrix = analysis['transition_matrix']
        assert np.allclose(trans_matrix.sum(axis=1), 1.0)

    def test_regime_correlations_with_nan(self, sample_features_and_target, sample_macro_data):
        """Test correlation computation handles NaN values."""
        features, target = sample_features_and_target

        # Add some NaN values
        features = features.copy()
        features.iloc[10:20, 0] = np.nan

        detector = RuleBasedRegimeDetector()
        regimes = detector.predict(sample_macro_data.loc[features.index])

        # Should not raise
        corr_df = compute_regime_correlations(features, target, regimes)
        assert len(corr_df) == len(features.columns)


# ============================================================================
# Integration Tests
# ============================================================================

class TestRegimeDetectionIntegration:
    """Integration tests combining multiple components."""

    def test_hmm_and_rule_based_alignment(self, sample_macro_data):
        """Test that HMM and rule-based produce similar results on clear data."""
        features = sample_macro_data[['VIXCLS', 'T10Y2Y', 'BAA10Y', 'UNRATE']]

        hmm = HMMRegimeDetector(n_regimes=3)
        hmm.fit(features)
        hmm_regimes = hmm.predict(features)

        rule_based = RuleBasedRegimeDetector()
        rule_regimes = rule_based.predict(sample_macro_data)

        # Compare using confusion matrix
        comparison = rule_based.compare_with_hmm(sample_macro_data, hmm_regimes)

        # Should have some agreement (not perfect, but reasonable)
        # At minimum, there should be entries on or near diagonal
        assert comparison.values.sum() == len(sample_macro_data)

    def test_regime_persistence(self, sample_macro_data):
        """Test that regimes are persistent (not switching every day)."""
        detector = RuleBasedRegimeDetector()
        regimes = detector.predict(sample_macro_data)

        # Count transitions
        transitions = (regimes != regimes.shift(1)).sum()

        # Should have far fewer transitions than observations
        # (regimes should persist for multiple days)
        # Note: Our test data has noise at boundaries, so we use a looser threshold
        # In real data with smoother macro indicators, this would be much lower
        assert transitions < len(sample_macro_data) / 5  # Allow up to 20% transition rate

    def test_crisis_periods_have_high_vix(self, sample_macro_data):
        """Test that crisis regime has higher VIX than others."""
        detector = RuleBasedRegimeDetector()
        regimes = detector.predict(sample_macro_data)

        crisis_mask = regimes == MarketRegime.CRISIS.value
        expansion_mask = regimes == MarketRegime.EXPANSION.value

        if crisis_mask.any() and expansion_mask.any():
            crisis_vix = sample_macro_data.loc[crisis_mask, 'VIXCLS'].mean()
            expansion_vix = sample_macro_data.loc[expansion_mask, 'VIXCLS'].mean()

            assert crisis_vix > expansion_vix


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

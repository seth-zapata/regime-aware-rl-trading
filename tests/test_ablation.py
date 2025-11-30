"""Tests for ablation study framework."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
sys.path.insert(0, 'src')


@pytest.fixture
def sample_returns():
    """Sample returns for statistical tests."""
    np.random.seed(42)
    return {
        'strategy_a': np.random.normal(0.02, 0.05, 20).tolist(),
        'strategy_b': np.random.normal(0.01, 0.05, 20).tolist(),
        'baseline': np.random.normal(0.015, 0.04, 20).tolist(),
    }


@pytest.fixture
def sample_data():
    """Create sample data for ablation study."""
    np.random.seed(42)
    n_days = 500

    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    df = pd.DataFrame({
        'price_Close': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n_days)),
        'price_return_1d': np.random.normal(0, 0.01, n_days),
        'price_return_5d': np.random.normal(0, 0.02, n_days),
        'price_volatility_21d': np.abs(np.random.normal(0.15, 0.05, n_days)),
        'price_rsi_14': np.random.uniform(30, 70, n_days),
        'macro_VIX': np.random.uniform(15, 35, n_days),
        'macro_yield': np.random.normal(0.01, 0.005, n_days),
        'regime_numeric': np.random.choice([0, 1, 2], n_days),
    }, index=dates)

    return df


class TestStatisticalTests:
    """Tests for statistical comparison functions."""

    def test_compare_returns(self, sample_returns):
        """Test return comparison."""
        from ablation.statistical_tests import compare_returns

        result = compare_returns(
            sample_returns['strategy_a'],
            sample_returns['strategy_b']
        )

        assert 'mean_a' in result
        assert 'mean_b' in result
        assert 'mean_diff' in result
        assert 'cohens_d' in result
        assert 'effect_size' in result
        assert 'paired_p_value' in result
        assert 'ind_p_value' in result

    def test_compare_returns_effect_size(self, sample_returns):
        """Test effect size calculation."""
        from ablation.statistical_tests import compare_returns

        result = compare_returns(
            sample_returns['strategy_a'],
            sample_returns['strategy_b']
        )

        # Effect size should be one of the categories
        assert result['effect_size'] in ['negligible', 'small', 'medium', 'large']

    def test_paired_t_test(self, sample_returns):
        """Test paired t-test."""
        from ablation.statistical_tests import paired_t_test

        t_stat, p_value, significant = paired_t_test(
            sample_returns['strategy_a'],
            sample_returns['strategy_b']
        )

        assert isinstance(t_stat, float)
        assert 0 <= p_value <= 1
        assert significant in [True, False]  # numpy bool or Python bool

    def test_bootstrap_confidence_interval(self, sample_returns):
        """Test bootstrap CI calculation."""
        from ablation.statistical_tests import bootstrap_confidence_interval

        point, lower, upper = bootstrap_confidence_interval(
            sample_returns['strategy_a'],
            statistic='mean',
            n_bootstrap=1000
        )

        assert lower <= point <= upper
        assert lower < upper  # CI should have width

    def test_bootstrap_sharpe(self, sample_returns):
        """Test bootstrap CI for Sharpe ratio."""
        from ablation.statistical_tests import bootstrap_confidence_interval

        point, lower, upper = bootstrap_confidence_interval(
            sample_returns['strategy_a'],
            statistic='sharpe',
            n_bootstrap=1000
        )

        assert isinstance(point, float)
        assert lower <= upper

    def test_calculate_effect_size(self, sample_returns):
        """Test effect size measures."""
        from ablation.statistical_tests import calculate_effect_size

        result = calculate_effect_size(
            sample_returns['strategy_a'],
            sample_returns['strategy_b']
        )

        assert 'cohens_d' in result
        assert 'hedges_g' in result
        assert 'cles' in result
        assert 0 <= result['cles'] <= 1

    def test_multiple_comparison_correction(self):
        """Test p-value correction methods."""
        from ablation.statistical_tests import multiple_comparison_correction

        p_values = [0.01, 0.03, 0.05, 0.10]

        # Bonferroni
        corrected = multiple_comparison_correction(p_values, method='bonferroni')
        assert len(corrected) == len(p_values)
        assert all(c >= p for c, p in zip(corrected, p_values))

        # Holm
        corrected_holm = multiple_comparison_correction(p_values, method='holm')
        assert len(corrected_holm) == len(p_values)

        # FDR
        corrected_fdr = multiple_comparison_correction(p_values, method='fdr')
        assert len(corrected_fdr) == len(p_values)


class TestAblationConfig:
    """Tests for AblationConfig."""

    def test_config_creation(self):
        """Test basic config creation."""
        from ablation.study import AblationConfig

        config = AblationConfig(
            name='test_config',
            feature_columns=['feature1', 'feature2'],
            use_regime=False
        )

        assert config.name == 'test_config'
        assert len(config.feature_columns) == 2
        assert config.use_regime == False

    def test_config_with_regime(self):
        """Test config with regime."""
        from ablation.study import AblationConfig

        config = AblationConfig(
            name='regime_config',
            feature_columns=['feature1'],
            use_regime=True,
            regime_column='regime_numeric'
        )

        assert config.use_regime == True
        assert config.regime_column == 'regime_numeric'

    def test_config_regime_requires_column(self):
        """Test that regime config requires column."""
        from ablation.study import AblationConfig

        with pytest.raises(ValueError):
            AblationConfig(
                name='bad_config',
                feature_columns=['feature1'],
                use_regime=True,
                # Missing regime_column
            )


class TestAblationResult:
    """Tests for AblationResult."""

    def test_result_properties(self):
        """Test result property calculations."""
        from ablation.study import AblationConfig, AblationResult

        config = AblationConfig(
            name='test',
            feature_columns=['f1'],
            use_regime=False
        )

        result = AblationResult(
            config=config,
            train_returns=[0.1, 0.2, 0.15],
            test_returns=[0.05, -0.02, 0.03],
            train_sharpes=[1.0, 1.5, 1.2],
            test_sharpes=[0.5, -0.2, 0.3],
            test_drawdowns=[0.05, 0.08, 0.06],
            num_trades=[10, 12, 8],
            training_time_sec=60.0
        )

        assert result.mean_test_return == pytest.approx(0.02, rel=0.01)
        assert result.mean_test_sharpe == pytest.approx(0.2, rel=0.01)
        assert result.win_rate == pytest.approx(2/3, rel=0.01)

    def test_result_summary_dict(self):
        """Test summary dictionary generation."""
        from ablation.study import AblationConfig, AblationResult

        config = AblationConfig(
            name='test',
            feature_columns=['f1', 'f2'],
            use_regime=True,
            regime_column='regime'
        )

        result = AblationResult(
            config=config,
            train_returns=[0.1],
            test_returns=[0.05],
            train_sharpes=[1.0],
            test_sharpes=[0.5],
            test_drawdowns=[0.05],
            num_trades=[10],
            training_time_sec=30.0
        )

        summary = result.summary_dict()

        assert summary['name'] == 'test'
        assert summary['n_features'] == 2
        assert summary['use_regime'] == True


class TestAblationStudy:
    """Tests for AblationStudy class."""

    def test_study_initialization(self, sample_data):
        """Test study initialization."""
        from ablation.study import AblationStudy

        study = AblationStudy(
            df=sample_data,
            price_column='price_Close',
            verbose=0
        )

        assert len(study.configs) == 0
        assert len(study.results) == 0

    def test_add_config(self, sample_data):
        """Test adding configurations."""
        from ablation.study import AblationStudy, AblationConfig

        study = AblationStudy(df=sample_data, verbose=0)

        config = AblationConfig(
            name='test',
            feature_columns=['price_return_1d', 'price_volatility_21d'],
            use_regime=False
        )

        study.add_config(config)
        assert len(study.configs) == 1

    def test_add_config_validates_features(self, sample_data):
        """Test that adding config validates features exist."""
        from ablation.study import AblationStudy, AblationConfig

        study = AblationStudy(df=sample_data, verbose=0)

        config = AblationConfig(
            name='bad',
            feature_columns=['nonexistent_feature'],
            use_regime=False
        )

        with pytest.raises(ValueError, match="Missing features"):
            study.add_config(config)

    def test_get_summary_df(self, sample_data):
        """Test summary DataFrame generation."""
        from ablation.study import AblationStudy, AblationConfig, AblationResult

        study = AblationStudy(df=sample_data, verbose=0)

        # Manually add a result
        config = AblationConfig(
            name='test',
            feature_columns=['price_return_1d'],
            use_regime=False
        )

        result = AblationResult(
            config=config,
            train_returns=[0.1],
            test_returns=[0.05],
            train_sharpes=[1.0],
            test_sharpes=[0.5],
            test_drawdowns=[0.05],
            num_trades=[10],
            training_time_sec=30.0
        )

        study.results.append(result)
        study.baseline_returns = [0.03]

        df = study.get_summary_df()

        assert len(df) == 2  # result + baseline
        assert 'name' in df.columns
        assert 'mean_test_return' in df.columns


class TestCreateStandardConfigs:
    """Tests for standard config creation."""

    def test_create_configs(self, sample_data):
        """Test standard config creation."""
        from ablation.study import create_standard_configs

        price_features = ['price_return_1d', 'price_return_5d', 'price_volatility_21d']
        macro_features = ['macro_VIX', 'macro_yield']

        configs = create_standard_configs(
            df=sample_data,
            price_features=price_features,
            macro_features=macro_features,
            regime_column='regime_numeric'
        )

        # Should have at least price_only and price_regime
        assert len(configs) >= 2

        # Check names
        names = [c.name for c in configs]
        assert 'price_only' in names
        assert 'price_regime' in names

    def test_create_configs_missing_features(self, sample_data):
        """Test config creation with missing features."""
        from ablation.study import create_standard_configs

        # Include features that don't exist
        price_features = ['price_return_1d', 'nonexistent']
        macro_features = ['macro_VIX', 'also_nonexistent']

        configs = create_standard_configs(
            df=sample_data,
            price_features=price_features,
            macro_features=macro_features,
            regime_column='regime_numeric'
        )

        # Should still create configs with available features
        assert len(configs) >= 1

        # price_only should only have the valid feature
        price_only = [c for c in configs if c.name == 'price_only'][0]
        assert 'nonexistent' not in price_only.feature_columns


class TestAblationIntegration:
    """Integration tests for ablation study."""

    @pytest.fixture
    def integration_data(self):
        """Create data suitable for integration test."""
        np.random.seed(42)
        n_days = 400  # Enough for walk-forward

        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        df = pd.DataFrame({
            'price_Close': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n_days)),
            'price_return_1d': np.random.normal(0, 0.01, n_days),
            'price_return_5d': np.random.normal(0, 0.02, n_days),
            'price_volatility_21d': np.abs(np.random.normal(0.15, 0.05, n_days)),
            'price_rsi_14': np.random.uniform(30, 70, n_days),
            'regime_numeric': np.random.choice([0, 1, 2], n_days),
        }, index=dates)

        return df

    def test_minimal_ablation_run(self, integration_data):
        """Test minimal ablation study run."""
        from ablation.study import AblationStudy, AblationConfig

        study = AblationStudy(
            df=integration_data,
            price_column='price_Close',
            verbose=0
        )

        # Add minimal config
        study.add_config(AblationConfig(
            name='minimal',
            feature_columns=['price_return_1d', 'price_volatility_21d'],
            use_regime=False
        ))

        # Run with minimal settings
        results = study.run(
            total_timesteps=1000,  # Very small for speed
            n_windows=2,
            seeds=[42],
            min_train_days=100,
            min_test_days=30
        )

        assert 'results' in results
        assert len(results['results']) == 1
        assert results['results'][0].config.name == 'minimal'

"""
Unit tests for data loaders.

Tests cover:
- PriceLoader: Yahoo Finance data fetching and technical features
- FREDLoader: FRED API data fetching and derived features
- EDGARLoader: SEC filing fetching and parsing
- DataPipeline: Data alignment and integration
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.price_loader import PriceLoader
from data.fred_loader import FREDLoader, FRED_SERIES, DEFAULT_REGIME_SERIES
from data.edgar_loader import EDGARLoader, Filing
from data.data_pipeline import DataPipeline


# =============================================================================
# PriceLoader Tests
# =============================================================================

class TestPriceLoader:
    """Tests for PriceLoader class."""

    @pytest.fixture
    def loader(self, tmp_path):
        """Create PriceLoader with temp cache directory."""
        return PriceLoader(cache_dir=tmp_path / 'price')

    def test_init_creates_cache_dir(self, tmp_path):
        """Test that initialization creates cache directory."""
        cache_dir = tmp_path / 'price_cache'
        loader = PriceLoader(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_get_price_data_returns_dataframe(self, loader):
        """Test that price data is returned as DataFrame."""
        df = loader.get_price_data(
            'SPY',
            start_date='2024-01-01',
            end_date='2024-01-31',
            use_cache=False
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_get_price_data_has_required_columns(self, loader):
        """Test that price data has required OHLCV columns."""
        df = loader.get_price_data(
            'SPY',
            start_date='2024-01-01',
            end_date='2024-01-31',
            use_cache=False
        )

        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_get_price_data_caches_data(self, loader, tmp_path):
        """Test that data is cached after first fetch."""
        df1 = loader.get_price_data(
            'SPY',
            start_date='2024-01-01',
            end_date='2024-01-15',
            use_cache=True
        )

        # Check cache file exists
        cache_files = list((tmp_path / 'price').glob('*.parquet'))
        assert len(cache_files) > 0

    def test_add_technical_features_adds_returns(self, loader):
        """Test that technical features include return calculations."""
        df = loader.get_price_data('SPY', start_date='2024-01-01', end_date='2024-03-01')
        df_features = loader.add_technical_features(df)

        assert 'return_1d' in df_features.columns
        assert 'return_5d' in df_features.columns
        assert 'log_return_1d' in df_features.columns

    def test_add_technical_features_adds_volatility(self, loader):
        """Test that technical features include volatility measures."""
        df = loader.get_price_data('SPY', start_date='2023-01-01', end_date='2024-01-01')
        df_features = loader.add_technical_features(df)

        assert 'volatility_21d' in df_features.columns
        assert 'volatility_63d' in df_features.columns

    def test_add_technical_features_adds_momentum(self, loader):
        """Test that technical features include momentum indicators."""
        df = loader.get_price_data('SPY', start_date='2023-01-01', end_date='2024-01-01')
        df_features = loader.add_technical_features(df)

        assert 'sma_20' in df_features.columns
        assert 'sma_50' in df_features.columns
        assert 'rsi_14' in df_features.columns
        assert 'macd' in df_features.columns

    def test_create_target_direction(self, loader):
        """Test that direction target is binary."""
        df = loader.get_price_data('SPY', start_date='2024-01-01', end_date='2024-03-01')
        df_target = loader.create_target(df, horizon=1, target_type='direction')

        assert 'target' in df_target.columns
        # Target should be 0 or 1 (excluding NaN)
        valid_targets = df_target['target'].dropna().unique()
        assert set(valid_targets).issubset({0, 1})

    def test_create_target_return(self, loader):
        """Test that return target is continuous."""
        df = loader.get_price_data('SPY', start_date='2024-01-01', end_date='2024-03-01')
        df_target = loader.create_target(df, horizon=1, target_type='return')

        assert 'target' in df_target.columns
        # Return target should be continuous (floats)
        assert df_target['target'].dtype == float

    def test_get_price_data_invalid_symbol(self, loader):
        """Test that invalid symbol raises error."""
        with pytest.raises(Exception):
            loader.get_price_data(
                'INVALID_SYMBOL_123456',
                start_date='2024-01-01',
                end_date='2024-01-31',
                use_cache=False
            )


# =============================================================================
# FREDLoader Tests
# =============================================================================

class TestFREDLoader:
    """Tests for FREDLoader class."""

    @pytest.fixture
    def loader(self, tmp_path):
        """Create FREDLoader with temp cache directory."""
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            pytest.skip("FRED_API_KEY not set")
        return FREDLoader(api_key=api_key, cache_dir=tmp_path / 'fred')

    def test_init_without_api_key_raises_error(self, tmp_path):
        """Test that missing API key raises ValueError."""
        # Temporarily unset env var
        original = os.environ.pop('FRED_API_KEY', None)
        try:
            with pytest.raises(ValueError, match="FRED API key required"):
                FREDLoader(api_key=None, cache_dir=tmp_path)
        finally:
            if original:
                os.environ['FRED_API_KEY'] = original

    def test_get_series_returns_series(self, loader):
        """Test that single series returns pd.Series."""
        data = loader.get_series(
            'T10Y2Y',
            start_date='2024-01-01',
            end_date='2024-03-01',
            use_cache=False
        )

        assert isinstance(data, pd.Series)
        assert len(data) > 0

    def test_get_macro_indicators_returns_dataframe(self, loader):
        """Test that multiple indicators return DataFrame."""
        df = loader.get_macro_indicators(
            series=['T10Y2Y', 'VIXCLS'],
            start_date='2024-01-01',
            end_date='2024-03-01',
            use_cache=False
        )

        assert isinstance(df, pd.DataFrame)
        assert 'T10Y2Y' in df.columns
        assert 'VIXCLS' in df.columns

    def test_get_macro_indicators_daily_alignment(self, loader):
        """Test that data is aligned to daily frequency."""
        df = loader.get_macro_indicators(
            series=['T10Y2Y', 'UNRATE'],  # Daily and monthly series
            start_date='2024-01-01',
            end_date='2024-03-01',
            align_to_daily=True
        )

        # Should have daily frequency (business days + some gaps filled)
        date_diff = df.index.to_series().diff().dropna()
        # Most diffs should be 1 day (allowing for weekends)
        assert (date_diff <= pd.Timedelta(days=3)).mean() > 0.9

    def test_calculate_derived_features(self, loader):
        """Test that derived features are calculated correctly."""
        df = loader.get_macro_indicators(
            series=['T10Y2Y', 'VIXCLS', 'UNRATE'],
            start_date='2023-01-01',
            end_date='2024-01-01'
        )
        df_features = loader.calculate_derived_features(df)

        # Check derived features exist
        assert 'T10Y2Y_inverted' in df_features.columns
        assert 'VIX_spike' in df_features.columns

    def test_invalid_series_raises_error(self, loader):
        """Test that invalid series ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown FRED series"):
            loader.get_macro_indicators(
                series=['INVALID_SERIES'],
                start_date='2024-01-01',
                end_date='2024-03-01'
            )

    def test_list_available_series(self):
        """Test that available series are listed."""
        series = FREDLoader.list_available_series()
        assert isinstance(series, dict)
        assert 'T10Y2Y' in series
        assert 'VIXCLS' in series

    def test_default_regime_series_are_valid(self):
        """Test that default regime series are in FRED_SERIES."""
        for series_id in DEFAULT_REGIME_SERIES:
            assert series_id in FRED_SERIES


# =============================================================================
# EDGARLoader Tests
# =============================================================================

class TestEDGARLoader:
    """Tests for EDGARLoader class."""

    @pytest.fixture
    def loader(self, tmp_path):
        """Create EDGARLoader with temp cache directory."""
        return EDGARLoader(
            email="test@example.com",
            cache_dir=tmp_path / 'edgar'
        )

    def test_init_creates_cache_dir(self, tmp_path):
        """Test that initialization creates cache directory."""
        cache_dir = tmp_path / 'edgar_cache'
        loader = EDGARLoader(email="test@example.com", cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_get_cik_returns_string(self, loader):
        """Test that CIK lookup returns valid string."""
        cik = loader.get_cik('AAPL')
        assert isinstance(cik, str)
        assert len(cik) == 10  # CIK is 10-digit zero-padded
        assert cik.isdigit()

    def test_get_cik_invalid_ticker(self, loader):
        """Test that invalid ticker raises error."""
        with pytest.raises(ValueError, match="Could not find CIK"):
            loader.get_cik('INVALID_TICKER_123456')

    def test_get_filings_returns_list(self, loader):
        """Test that filings are returned as list."""
        filings = loader.get_filings('AAPL', filing_type='10-K', count=2)

        assert isinstance(filings, list)
        assert len(filings) <= 2

    def test_get_filings_returns_filing_objects(self, loader):
        """Test that filings list contains Filing objects."""
        filings = loader.get_filings('AAPL', filing_type='10-K', count=1)

        if filings:  # May be empty if rate limited
            filing = filings[0]
            assert isinstance(filing, Filing)
            assert filing.filing_type == '10-K'
            assert filing.cik is not None

    def test_get_filings_dataframe(self, loader):
        """Test that filings can be returned as DataFrame."""
        df = loader.get_filings_dataframe(
            'AAPL',
            filing_types=['10-K'],
            extract_text=False
        )

        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert 'filing_date' in df.columns
            assert 'filing_type' in df.columns

    def test_filing_date_filter(self, loader):
        """Test that date filtering works."""
        filings = loader.get_filings(
            'AAPL',
            filing_type='10-K',
            start_date='2020-01-01',
            end_date='2023-01-01'
        )

        for filing in filings:
            assert filing.filing_date >= '2020-01-01'
            assert filing.filing_date <= '2023-01-01'


# =============================================================================
# DataPipeline Tests
# =============================================================================

class TestDataPipeline:
    """Tests for DataPipeline class."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create DataPipeline with temp cache directory."""
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            pytest.skip("FRED_API_KEY not set")
        return DataPipeline(
            fred_api_key=api_key,
            cache_dir=tmp_path
        )

    def test_load_price_data(self, pipeline):
        """Test loading price data through pipeline."""
        df = pipeline.load_price_data(
            'SPY',
            start_date='2024-01-01',
            end_date='2024-02-01',
            add_features=True
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_macro_data(self, pipeline):
        """Test loading macro data through pipeline."""
        df = pipeline.load_macro_data(
            start_date='2024-01-01',
            end_date='2024-02-01',
            add_features=True
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_aligned_data(self, pipeline):
        """Test loading and aligning multiple data sources."""
        df = pipeline.load_aligned_data(
            symbol='SPY',
            start_date='2024-01-01',
            end_date='2024-02-01',
            include_price=True,
            include_macro=True,
            include_filings=False
        )

        assert isinstance(df, pd.DataFrame)
        # Should have both price and macro columns
        price_cols = [c for c in df.columns if c.startswith('price_')]
        macro_cols = [c for c in df.columns if c.startswith('macro_')]
        assert len(price_cols) > 0
        assert len(macro_cols) > 0

    def test_create_modeling_dataset(self, pipeline):
        """Test creating modeling dataset with features and target."""
        X, y = pipeline.create_modeling_dataset(
            symbol='SPY',
            start_date='2023-01-01',
            end_date='2024-01-01',
            target_horizon=1,
            target_type='direction'
        )

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        # No NaN after dropna
        assert X.isna().sum().sum() == 0
        assert y.isna().sum() == 0

    def test_train_test_split_temporal_order(self, pipeline):
        """Test that train/test split maintains temporal order."""
        X, y = pipeline.create_modeling_dataset(
            symbol='SPY',
            start_date='2020-01-01',
            end_date='2024-01-01'
        )

        splits = pipeline.get_train_test_split(X, y)

        # Ensure we have data in each split
        assert len(splits['train'][0]) > 0, "Train set is empty"
        assert len(splits['val'][0]) > 0, "Val set is empty"
        assert len(splits['test'][0]) > 0, "Test set is empty"

        # Train should be before val
        assert splits['train'][0].index.max() < splits['val'][0].index.min()
        # Val should be before test
        assert splits['val'][0].index.max() < splits['test'][0].index.min()

    def test_train_test_split_no_overlap(self, pipeline):
        """Test that train/test split has no data overlap."""
        X, y = pipeline.create_modeling_dataset(
            symbol='SPY',
            start_date='2023-01-01',
            end_date='2024-01-01'
        )

        splits = pipeline.get_train_test_split(X, y)

        train_dates = set(splits['train'][0].index)
        val_dates = set(splits['val'][0].index)
        test_dates = set(splits['test'][0].index)

        assert len(train_dates & val_dates) == 0
        assert len(train_dates & test_dates) == 0
        assert len(val_dates & test_dates) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full data pipeline."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create DataPipeline for integration tests."""
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            pytest.skip("FRED_API_KEY not set")
        return DataPipeline(fred_api_key=api_key, cache_dir=tmp_path)

    def test_full_pipeline_no_lookahead_bias(self, pipeline):
        """Test that pipeline doesn't introduce look-ahead bias."""
        X, y = pipeline.create_modeling_dataset(
            symbol='SPY',
            start_date='2020-01-01',
            end_date='2024-01-01',
            target_horizon=1
        )

        # Target at time t should be based on price at t+1
        # Features at time t should only use data up to t
        # This is validated by the fact that target is shifted properly

        # Check that target is not perfectly correlated with same-day features
        # (which would indicate leakage)
        # Only check numeric features with variance
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        checked = 0
        for col in numeric_cols:
            if X[col].std() > 0:  # Skip constant columns
                corr = X[col].corr(y)
                if not np.isnan(corr):
                    # Correlation should be imperfect (not 1.0 or -1.0)
                    assert abs(corr) < 0.99, f"Suspiciously high correlation for {col}"
                    checked += 1
                    if checked >= 5:
                        break

    def test_macro_data_forward_filled(self, pipeline):
        """Test that macro data is forward-filled correctly."""
        df = pipeline.load_aligned_data(
            symbol='SPY',
            start_date='2020-01-01',  # Longer period to ensure data available
            end_date='2024-01-01',
            include_price=True,
            include_macro=True
        )

        # Macro columns should not have many NaN values after forward-fill
        # Only check base macro indicators, not derived features which may have NaN
        base_macro_cols = [c for c in df.columns
                          if c.startswith('macro_') and
                          not any(x in c for x in ['change', 'inverted', 'spike', 'rising', 'percentile', 'stress'])]

        for col in base_macro_cols:
            # Drop first 30 days to allow for data availability lag
            data = df[col].iloc[30:]
            nan_ratio = data.isna().mean()
            # Allow some NaN but most should be filled
            assert nan_ratio < 0.2, f"Too many NaN in {col}: {nan_ratio:.2%}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

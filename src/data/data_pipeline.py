"""
Data Pipeline

Orchestrates data loading from multiple sources and aligns them to a common
daily frequency for use in regime detection and trading models.

Key responsibilities:
- Load and align price, FRED, and EDGAR data
- Handle different data frequencies (daily, weekly, monthly, quarterly)
- Prevent look-ahead bias in data alignment
- Create combined feature sets for modeling
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .price_loader import PriceLoader
from .fred_loader import FREDLoader, DEFAULT_REGIME_SERIES
from .edgar_loader import EDGARLoader


class DataPipeline:
    """
    Unified data pipeline for multi-source data loading and alignment.

    Combines price data, FRED macro indicators, and SEC filings into a
    single aligned DataFrame for modeling.

    Example:
        pipeline = DataPipeline()
        data = pipeline.load_aligned_data(
            symbol='SPY',
            start_date='2015-01-01',
            end_date='2024-01-01'
        )
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        edgar_email: str = "research@example.com",
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize data pipeline with all loaders.

        Args:
            fred_api_key: FRED API key (reads from env if not provided)
            edgar_email: Email for SEC EDGAR requests
            cache_dir: Base cache directory
        """
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / 'data'

        self.cache_dir = Path(cache_dir)

        # Initialize loaders
        self.price_loader = PriceLoader(cache_dir=cache_dir / 'raw' / 'price')
        self.fred_loader = FREDLoader(
            api_key=fred_api_key,
            cache_dir=cache_dir / 'raw' / 'fred'
        )
        self.edgar_loader = EDGARLoader(
            email=edgar_email,
            cache_dir=cache_dir / 'raw' / 'edgar'
        )

    def load_price_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Load and process price data.

        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            add_features: Whether to add technical features

        Returns:
            DataFrame with price data and optional features
        """
        df = self.price_loader.get_price_data(
            symbol,
            start_date=start_date,
            end_date=end_date
        )

        if add_features:
            df = self.price_loader.add_technical_features(df)

        return df

    def load_macro_data(
        self,
        start_date: str,
        end_date: str,
        series: Optional[List[str]] = None,
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Load and process FRED macro data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            series: List of FRED series to fetch
            add_features: Whether to add derived features

        Returns:
            DataFrame with macro indicators aligned to daily frequency
        """
        df = self.fred_loader.get_macro_indicators(
            series=series,
            start_date=start_date,
            end_date=end_date,
            align_to_daily=True
        )

        if add_features:
            df = self.fred_loader.calculate_derived_features(df)

        return df

    def load_filing_sentiment(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        filing_types: List[str] = ['10-K', '10-Q']
    ) -> pd.DataFrame:
        """
        Load SEC filings and create sentiment features.

        Note: This returns filing-level data that needs to be aligned
        with daily price data using forward-fill (filing sentiment
        persists until the next filing).

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            filing_types: Types of filings to fetch

        Returns:
            DataFrame with filing dates and text for sentiment analysis
        """
        df = self.edgar_loader.get_filings_dataframe(
            ticker,
            filing_types=filing_types,
            start_date=start_date,
            end_date=end_date,
            extract_text=True
        )

        return df

    def load_aligned_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        include_price: bool = True,
        include_macro: bool = True,
        include_filings: bool = False,
        price_features: bool = True,
        macro_features: bool = True,
        macro_series: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load and align data from all sources.

        This is the main entry point for getting a unified dataset.

        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_price: Include price data
            include_macro: Include FRED macro data
            include_filings: Include SEC filing data (slower)
            price_features: Add technical features to price data
            macro_features: Add derived features to macro data
            macro_series: Specific FRED series to include

        Returns:
            DataFrame aligned to daily frequency with all requested data
        """
        dfs_to_merge = []

        # Load price data (base for alignment)
        if include_price:
            price_df = self.load_price_data(
                symbol,
                start_date=start_date,
                end_date=end_date,
                add_features=price_features
            )
            # Add prefix to avoid column name conflicts
            price_df = price_df.add_prefix('price_')
            dfs_to_merge.append(price_df)

        # Load macro data
        if include_macro:
            macro_df = self.load_macro_data(
                start_date=start_date,
                end_date=end_date,
                series=macro_series,
                add_features=macro_features
            )
            macro_df = macro_df.add_prefix('macro_')
            dfs_to_merge.append(macro_df)

        # Load filing data
        if include_filings:
            filing_df = self.load_filing_sentiment(
                ticker=symbol,
                start_date=start_date,
                end_date=end_date
            )
            if not filing_df.empty:
                # Convert to daily by forward-filling filing sentiment
                filing_daily = self._align_filings_to_daily(
                    filing_df,
                    start_date,
                    end_date
                )
                filing_daily = filing_daily.add_prefix('filing_')
                dfs_to_merge.append(filing_daily)

        if not dfs_to_merge:
            raise ValueError("No data sources selected")

        # Merge all DataFrames
        result = dfs_to_merge[0]
        for df in dfs_to_merge[1:]:
            result = result.join(df, how='outer')

        # Forward-fill macro data (released with lag, value persists)
        macro_cols = [c for c in result.columns if c.startswith('macro_')]
        result[macro_cols] = result[macro_cols].ffill()

        # Sort by date
        result = result.sort_index()

        # Filter to requested date range (in case of edge effects)
        result = result[start_date:end_date]

        return result

    def _align_filings_to_daily(
        self,
        filing_df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Align filing data to daily frequency.

        Filing sentiment persists from filing date until the next filing.
        This prevents look-ahead bias - we only know about a filing after
        it's been filed.

        Args:
            filing_df: DataFrame with filing-level data
            start_date: Start date for daily range
            end_date: End date for daily range

        Returns:
            DataFrame with daily frequency, forward-filled from filing dates
        """
        if filing_df.empty:
            return pd.DataFrame()

        # Create daily date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Set filing_date as index
        filing_df = filing_df.set_index('filing_date')

        # Select relevant columns (exclude URLs and raw text)
        cols_to_keep = ['filing_type']

        # Add text length features (useful proxy for sentiment complexity)
        if 'risk_factors' in filing_df.columns:
            filing_df['risk_factors_length'] = filing_df['risk_factors'].str.len().fillna(0)
            cols_to_keep.append('risk_factors_length')

        if 'mda' in filing_df.columns:
            filing_df['mda_length'] = filing_df['mda'].str.len().fillna(0)
            cols_to_keep.append('mda_length')

        # Create filing type indicators
        filing_df['is_10k'] = (filing_df['filing_type'] == '10-K').astype(int)
        filing_df['is_10q'] = (filing_df['filing_type'] == '10-Q').astype(int)
        cols_to_keep.extend(['is_10k', 'is_10q'])

        # Reindex to daily and forward-fill
        daily_df = filing_df[cols_to_keep].reindex(date_range)
        daily_df = daily_df.ffill()

        # Add days since last filing
        filing_dates = filing_df.index
        daily_df['days_since_filing'] = pd.Series(
            [(d - filing_dates[filing_dates <= d].max()).days
             if any(filing_dates <= d) else np.nan
             for d in date_range],
            index=date_range
        )

        return daily_df

    def create_modeling_dataset(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        target_horizon: int = 1,
        target_type: str = 'direction',
        include_macro: bool = True,
        dropna: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create a dataset ready for model training.

        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            target_horizon: Days ahead to predict
            target_type: 'direction' or 'return'
            include_macro: Include macro indicators
            dropna: Drop rows with NaN values

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Load aligned data
        df = self.load_aligned_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            include_price=True,
            include_macro=include_macro,
            include_filings=False,  # Skip for speed, can add later
            price_features=True,
            macro_features=True
        )

        # Create target
        close_col = 'price_Adj_Close' if 'price_Adj_Close' in df.columns else 'price_Close'
        # Calculate future return (use periods parameter to avoid deprecation warning)
        future_return = (df[close_col] / df[close_col].shift(target_horizon) - 1).shift(-target_horizon)

        if target_type == 'direction':
            target = (future_return > 0).astype(int)
        else:
            target = future_return

        target.name = 'target'

        # Remove target-related columns and raw OHLCV (keep only features)
        feature_cols = [c for c in df.columns if not any(
            x in c.lower() for x in ['open', 'high', 'low', 'close', 'volume', 'dividends', 'splits']
        ) or 'sma' in c.lower() or 'return' in c.lower() or 'vol' in c.lower()]

        # Also keep derived price features
        feature_cols = [c for c in df.columns if c in feature_cols or
                       any(x in c for x in ['return', 'volatility', 'sma', 'rsi', 'macd', 'obv', 'percentile'])]

        features = df[feature_cols]

        if dropna:
            # Align features and target, then drop NaN
            combined = pd.concat([features, target], axis=1)
            combined = combined.dropna()
            features = combined.drop('target', axis=1)
            target = combined['target']

        return features, target

    def get_train_test_split(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Create time-based train/validation/test split.

        IMPORTANT: Uses time-based split to prevent look-ahead bias.
        Never shuffles data - maintains temporal order.

        Args:
            features: Feature DataFrame
            target: Target Series
            test_ratio: Proportion for test set
            val_ratio: Proportion for validation set

        Returns:
            Dict with 'train', 'val', 'test' keys, each containing (X, y) tuple
        """
        n = len(features)
        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)
        train_size = n - test_size - val_size

        # Split maintaining temporal order
        X_train = features.iloc[:train_size]
        y_train = target.iloc[:train_size]

        X_val = features.iloc[train_size:train_size + val_size]
        y_val = target.iloc[train_size:train_size + val_size]

        X_test = features.iloc[train_size + val_size:]
        y_test = target.iloc[train_size + val_size:]

        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }


def main():
    """Example usage of DataPipeline."""
    pipeline = DataPipeline()

    print("Loading aligned data for SPY...")
    df = pipeline.load_aligned_data(
        symbol='SPY',
        start_date='2020-01-01',
        end_date='2024-01-01',
        include_price=True,
        include_macro=True,
        include_filings=False  # Skip for speed in example
    )

    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nColumns ({len(df.columns)}):")

    # Group columns by prefix
    prefixes = set(c.split('_')[0] for c in df.columns)
    for prefix in sorted(prefixes):
        cols = [c for c in df.columns if c.startswith(prefix)]
        print(f"  {prefix}: {len(cols)} columns")

    # Create modeling dataset
    print("\nCreating modeling dataset...")
    X, y = pipeline.create_modeling_dataset(
        symbol='SPY',
        start_date='2020-01-01',
        end_date='2024-01-01',
        target_horizon=1,
        target_type='direction'
    )

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # Create train/test split
    splits = pipeline.get_train_test_split(X, y)
    for name, (X_split, y_split) in splits.items():
        print(f"\n{name.upper()}:")
        print(f"  Samples: {len(X_split)}")
        print(f"  Date range: {X_split.index.min()} to {X_split.index.max()}")


if __name__ == '__main__':
    main()

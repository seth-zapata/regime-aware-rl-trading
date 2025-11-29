"""
FRED Data Loader

Fetches macroeconomic indicators from the Federal Reserve Economic Data (FRED) API.
These indicators are used for regime detection and as features for trading models.

Key indicators:
- Yield curve (10Y-2Y spread): Recession predictor
- VIX: Market volatility/fear gauge
- Unemployment rate: Economic health
- Fed funds rate: Monetary policy
- Credit spreads: Financial stress
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# FRED series IDs and descriptions for macro indicators
FRED_SERIES = {
    # Yield curve indicators
    'DGS10': '10-Year Treasury Constant Maturity Rate',
    'DGS2': '2-Year Treasury Constant Maturity Rate',
    'T10Y2Y': '10-Year Treasury Minus 2-Year Treasury (Yield Curve)',
    'T10Y3M': '10-Year Treasury Minus 3-Month Treasury',

    # Volatility
    'VIXCLS': 'CBOE Volatility Index (VIX)',

    # Employment
    'UNRATE': 'Unemployment Rate',
    'ICSA': 'Initial Claims (Weekly Unemployment Claims)',

    # Monetary policy
    'FEDFUNDS': 'Federal Funds Effective Rate',
    'WALCL': 'Federal Reserve Total Assets (Balance Sheet)',

    # Credit conditions
    'BAA10Y': 'Moody\'s BAA Corporate Bond Yield Minus 10-Year Treasury',
    'TEDRATE': 'TED Spread (3-Month LIBOR minus 3-Month T-Bill)',

    # Economic activity
    'INDPRO': 'Industrial Production Index',
    'RSAFS': 'Retail Sales: Total',

    # Inflation
    'CPIAUCSL': 'Consumer Price Index (All Urban Consumers)',
    'PCEPI': 'Personal Consumption Expenditures Price Index',
}

# Default series for regime detection (most important indicators)
DEFAULT_REGIME_SERIES = [
    'T10Y2Y',    # Yield curve - best recession predictor
    'VIXCLS',    # Volatility - risk sentiment
    'UNRATE',    # Unemployment - economic health
    'FEDFUNDS',  # Fed policy
    'BAA10Y',    # Credit spreads - financial stress
]


class FREDLoader:
    """
    Load macroeconomic data from FRED API.

    Handles API calls, caching, and data alignment for use in regime detection
    and trading models.

    Example:
        loader = FREDLoader()
        data = loader.get_macro_indicators(
            series=['T10Y2Y', 'VIXCLS', 'UNRATE'],
            start_date='2010-01-01',
            end_date='2024-01-01'
        )
    """

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        """
        Initialize FRED loader.

        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env variable.
            cache_dir: Directory for caching data. Defaults to data/raw/fred/
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key parameter. Get a free key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        # Initialize FRED API client
        try:
            from fredapi import Fred
            self.fred = Fred(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "fredapi package required. Install with: pip install fredapi"
            )

        # Set up cache directory
        if cache_dir is None:
            # Default to project data directory
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / 'data' / 'raw' / 'fred'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.Series:
        """
        Fetch a single FRED series.

        Args:
            series_id: FRED series ID (e.g., 'T10Y2Y', 'VIXCLS')
            start_date: Start date as 'YYYY-MM-DD'. Defaults to 10 years ago.
            end_date: End date as 'YYYY-MM-DD'. Defaults to today.
            use_cache: Whether to use cached data if available.

        Returns:
            pd.Series with DatetimeIndex and series values
        """
        # Set default dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')

        # Check cache
        cache_file = self.cache_dir / f"{series_id}_{start_date}_{end_date}.parquet"
        if use_cache and cache_file.exists():
            df = pd.read_parquet(cache_file)
            return df[series_id]

        # Fetch from FRED API
        try:
            data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            data.name = series_id

            # Cache the data
            if use_cache:
                df = data.to_frame()
                df.to_parquet(cache_file)

            return data

        except Exception as e:
            raise RuntimeError(f"Failed to fetch FRED series {series_id}: {e}")

    def get_macro_indicators(
        self,
        series: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        align_to_daily: bool = True
    ) -> pd.DataFrame:
        """
        Fetch multiple macro indicators and combine into a DataFrame.

        Args:
            series: List of FRED series IDs. Defaults to DEFAULT_REGIME_SERIES.
            start_date: Start date as 'YYYY-MM-DD'.
            end_date: End date as 'YYYY-MM-DD'.
            use_cache: Whether to use cached data.
            align_to_daily: Forward-fill to daily frequency for alignment with price data.

        Returns:
            pd.DataFrame with DatetimeIndex and columns for each indicator
        """
        if series is None:
            series = DEFAULT_REGIME_SERIES

        # Validate series IDs
        invalid_series = [s for s in series if s not in FRED_SERIES]
        if invalid_series:
            available = ', '.join(FRED_SERIES.keys())
            raise ValueError(
                f"Unknown FRED series: {invalid_series}. "
                f"Available series: {available}"
            )

        # Fetch each series
        data_dict = {}
        for series_id in series:
            try:
                data_dict[series_id] = self.get_series(
                    series_id,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache
                )
            except Exception as e:
                print(f"Warning: Failed to fetch {series_id}: {e}")
                continue

        if not data_dict:
            raise RuntimeError("Failed to fetch any FRED series")

        # Combine into DataFrame
        df = pd.DataFrame(data_dict)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Align to daily frequency if requested
        if align_to_daily:
            df = self._align_to_daily(df)

        return df

    def _align_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align data to daily frequency using forward-fill.

        FRED data has different frequencies:
        - Daily: VIX, Treasury rates
        - Weekly: Initial claims
        - Monthly: Unemployment, CPI

        We forward-fill to create daily values, which is valid because:
        - We only use data that was available at the time (no look-ahead)
        - Monthly unemployment on Jan 31 is the value you'd know in February

        Args:
            df: DataFrame with mixed-frequency data

        Returns:
            DataFrame resampled to daily frequency with forward-fill
        """
        # Create daily date range
        date_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='D'
        )

        # Reindex and forward-fill
        df_daily = df.reindex(date_range)
        df_daily = df_daily.ffill()

        # Drop weekends (markets closed) - optional, can be configured
        # For now, keep all days for flexibility

        return df_daily

    def get_series_info(self, series_id: str) -> Dict:
        """
        Get metadata about a FRED series.

        Args:
            series_id: FRED series ID

        Returns:
            Dict with series metadata (title, frequency, units, etc.)
        """
        try:
            info = self.fred.get_series_info(series_id)
            return info.to_dict()
        except Exception as e:
            raise RuntimeError(f"Failed to get info for {series_id}: {e}")

    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features from raw macro indicators.

        Features:
        - Yield curve slope changes
        - VIX percentile rank
        - Unemployment rate changes
        - Rate of change indicators

        Args:
            df: DataFrame with raw macro indicators

        Returns:
            DataFrame with original and derived features
        """
        df = df.copy()

        # Yield curve changes (if available)
        if 'T10Y2Y' in df.columns:
            df['T10Y2Y_change_1m'] = df['T10Y2Y'].diff(21)  # ~1 month
            df['T10Y2Y_inverted'] = (df['T10Y2Y'] < 0).astype(int)

        # VIX features (if available)
        if 'VIXCLS' in df.columns:
            # Rolling percentile (where does current VIX rank historically)
            df['VIX_percentile'] = df['VIXCLS'].rolling(252).apply(
                lambda x: (x.iloc[-1] > x).mean() if len(x) > 0 else 0.5
            )
            # VIX spike indicator
            df['VIX_spike'] = (df['VIXCLS'] > df['VIXCLS'].rolling(63).mean() +
                               2 * df['VIXCLS'].rolling(63).std()).astype(int)

        # Unemployment changes (if available)
        if 'UNRATE' in df.columns:
            df['UNRATE_change_3m'] = df['UNRATE'].diff(63)  # ~3 months
            df['UNRATE_rising'] = (df['UNRATE_change_3m'] > 0.3).astype(int)

        # Fed funds rate changes (if available)
        if 'FEDFUNDS' in df.columns:
            df['FEDFUNDS_change_6m'] = df['FEDFUNDS'].diff(126)  # ~6 months

        # Credit spread changes (if available)
        if 'BAA10Y' in df.columns:
            df['BAA10Y_change_1m'] = df['BAA10Y'].diff(21)
            df['BAA10Y_stress'] = (df['BAA10Y'] > 3.0).astype(int)  # Elevated spread

        return df

    @staticmethod
    def list_available_series() -> Dict[str, str]:
        """Return dictionary of available FRED series IDs and descriptions."""
        return FRED_SERIES.copy()


def main():
    """Example usage of FREDLoader."""
    # Initialize loader
    loader = FREDLoader()

    # Fetch default regime indicators
    print("Fetching macro indicators...")
    df = loader.get_macro_indicators(
        start_date='2015-01-01',
        end_date='2024-01-01'
    )

    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample data:\n{df.tail()}")

    # Calculate derived features
    print("\nCalculating derived features...")
    df_features = loader.calculate_derived_features(df)
    print(f"Total features: {len(df_features.columns)}")
    print(f"New columns: {[c for c in df_features.columns if c not in df.columns]}")


if __name__ == '__main__':
    main()

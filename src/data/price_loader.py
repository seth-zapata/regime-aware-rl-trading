"""
Price Data Loader

Fetches OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.
Provides standardized interface for price data used in trading models.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance package required. Install with: pip install yfinance")


class PriceLoader:
    """
    Load price data from Yahoo Finance.

    Provides OHLCV data with optional technical indicators and feature engineering.

    Example:
        loader = PriceLoader()
        data = loader.get_price_data('SPY', start_date='2020-01-01')
        data_with_features = loader.add_technical_features(data)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize price loader.

        Args:
            cache_dir: Directory for caching data. Defaults to data/raw/price/
        """
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / 'data' / 'raw' / 'price'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_price_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'SPY', 'AAPL')
            start_date: Start date as 'YYYY-MM-DD'. Defaults to 10 years ago.
            end_date: End date as 'YYYY-MM-DD'. Defaults to today.
            use_cache: Whether to use cached data.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
        """
        # Set default dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')

        # Check cache
        cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.parquet"
        if use_cache and cache_file.exists():
            return pd.read_parquet(cache_file)

        # Fetch from Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Standardize column names
            df.columns = [col.title().replace(' ', '_') for col in df.columns]

            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Remove timezone info for consistency
            df.index = df.index.tz_localize(None)

            # Cache the data
            if use_cache:
                df.to_parquet(cache_file)

            return df

        except Exception as e:
            raise RuntimeError(f"Failed to fetch price data for {symbol}: {e}")

    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        price_col: str = 'Adj_Close',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch price data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            price_col: Which price column to use for multi-symbol DataFrame
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns for each symbol (using price_col values)
        """
        data_dict = {}
        for symbol in symbols:
            try:
                df = self.get_price_data(
                    symbol,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache
                )
                data_dict[symbol] = df[price_col]
            except Exception as e:
                print(f"Warning: Failed to fetch {symbol}: {e}")
                continue

        if not data_dict:
            raise RuntimeError("Failed to fetch any symbols")

        return pd.DataFrame(data_dict)

    def add_technical_features(
        self,
        df: pd.DataFrame,
        include_returns: bool = True,
        include_volatility: bool = True,
        include_momentum: bool = True,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Add technical features to price data.

        Args:
            df: DataFrame with OHLCV data
            include_returns: Add return features
            include_volatility: Add volatility features
            include_momentum: Add momentum indicators
            include_volume: Add volume features

        Returns:
            DataFrame with additional feature columns
        """
        df = df.copy()

        # Use Adj_Close if available, else Close
        close_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
        close = df[close_col]

        if include_returns:
            df = self._add_return_features(df, close)

        if include_volatility:
            df = self._add_volatility_features(df, close)

        if include_momentum:
            df = self._add_momentum_features(df, close)

        if include_volume and 'Volume' in df.columns:
            df = self._add_volume_features(df)

        return df

    def _add_return_features(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add return-based features."""
        # Daily returns
        df['return_1d'] = close.pct_change()

        # Multi-day returns
        df['return_5d'] = close.pct_change(5)
        df['return_21d'] = close.pct_change(21)  # ~1 month
        df['return_63d'] = close.pct_change(63)  # ~3 months

        # Log returns (more stable for modeling)
        df['log_return_1d'] = np.log(close / close.shift(1))

        return df

    def _add_volatility_features(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add volatility features."""
        returns = close.pct_change()

        # Rolling volatility (annualized)
        df['volatility_21d'] = returns.rolling(21).std() * np.sqrt(252)
        df['volatility_63d'] = returns.rolling(63).std() * np.sqrt(252)

        # Realized volatility using high-low range (Parkinson)
        if 'High' in df.columns and 'Low' in df.columns:
            log_hl = np.log(df['High'] / df['Low'])
            df['parkinson_vol_21d'] = np.sqrt(
                (log_hl ** 2).rolling(21).mean() / (4 * np.log(2))
            ) * np.sqrt(252)

        # Volatility percentile (where does current vol rank)
        df['vol_percentile'] = df['volatility_21d'].rolling(252).apply(
            lambda x: (x.iloc[-1] > x).mean() if len(x) > 0 else 0.5
        )

        return df

    def _add_momentum_features(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add momentum indicators."""
        # Simple Moving Averages
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()
        df['sma_200'] = close.rolling(200).mean()

        # Price relative to moving averages
        df['close_to_sma20'] = close / df['sma_20'] - 1
        df['close_to_sma50'] = close / df['sma_50'] - 1
        df['close_to_sma200'] = close / df['sma_200'] - 1

        # Moving average crossover signals
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['sma_50_200_cross'] = (df['sma_50'] > df['sma_200']).astype(int)

        # RSI (Relative Strength Index)
        df['rsi_14'] = self._calculate_rsi(close, 14)

        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        volume = df['Volume']

        # Volume moving averages
        df['volume_sma_20'] = volume.rolling(20).mean()

        # Relative volume
        df['relative_volume'] = volume / df['volume_sma_20']

        # Volume trend
        df['volume_change_5d'] = volume.pct_change(5)

        # On-balance volume indicator (simplified)
        close_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
        df['obv'] = (np.sign(df[close_col].diff()) * volume).cumsum()

        return df

    @staticmethod
    def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def create_target(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        target_type: str = 'direction'
    ) -> pd.DataFrame:
        """
        Create prediction target variable.

        Args:
            df: DataFrame with price data
            horizon: Number of days ahead to predict
            target_type: 'direction' (binary up/down) or 'return' (continuous)

        Returns:
            DataFrame with target column added
        """
        df = df.copy()
        close_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'

        # Future return (shifted back to align with features at time t)
        future_return = df[close_col].pct_change(horizon).shift(-horizon)

        if target_type == 'direction':
            df['target'] = (future_return > 0).astype(int)
        elif target_type == 'return':
            df['target'] = future_return
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        return df


def main():
    """Example usage of PriceLoader."""
    loader = PriceLoader()

    # Fetch SPY data
    print("Fetching SPY price data...")
    df = loader.get_price_data('SPY', start_date='2020-01-01')

    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nColumns: {list(df.columns)}")

    # Add technical features
    print("\nAdding technical features...")
    df_features = loader.add_technical_features(df)
    print(f"Total columns: {len(df_features.columns)}")

    # Create target
    df_target = loader.create_target(df_features, horizon=1, target_type='direction')
    print(f"\nTarget distribution:\n{df_target['target'].value_counts(normalize=True)}")


if __name__ == '__main__':
    main()

"""
SEC Filing Sentiment Pipeline.

This module combines EDGAR filing downloads with FinBERT sentiment analysis
to create time-series sentiment features from 10-K and 10-Q filings.

Key design decisions:
--------------------
1. Analyze specific sections (Risk Factors, MD&A) rather than full filing
   - Risk Factors: Forward-looking concerns
   - MD&A: Management's view on recent performance

2. Create multiple sentiment features:
   - Raw sentiment scores (positive, negative, neutral)
   - Compound score (positive - negative)
   - Sentiment change (vs. previous filing)
   - Sentiment surprise (vs. trailing average)

3. Handle filing dates carefully:
   - Use filing date (not period date) to prevent look-ahead bias
   - A 10-K for Q4 might be filed in February - use February date

4. Forward-fill sentiment to daily frequency:
   - Sentiment persists until next filing
   - This simulates real-world information availability
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.edgar_loader import EDGARLoader, Filing
from sentiment.finbert_analyzer import FinBERTAnalyzer


class FilingSentimentPipeline:
    """
    Pipeline for extracting sentiment from SEC filings.

    This class handles the full workflow:
    1. Download filings from EDGAR
    2. Extract relevant text sections
    3. Analyze sentiment with FinBERT
    4. Create time-series features

    Attributes:
        edgar_loader: EDGARLoader for downloading filings
        analyzer: FinBERTAnalyzer for sentiment scoring
        sections: Which filing sections to analyze

    Example:
        >>> pipeline = FilingSentimentPipeline()
        >>> sentiment_df = pipeline.get_sentiment_features(
        ...     ticker='AAPL',
        ...     start_date='2020-01-01',
        ...     end_date='2024-01-01'
        ... )
    """

    # Default sections to analyze
    DEFAULT_SECTIONS = ['item_1a', 'item_7']  # Risk Factors, MD&A

    def __init__(
        self,
        email: str = "research@example.com",
        cache_dir: Optional[Path] = None,
        sections: Optional[List[str]] = None
    ):
        """
        Initialize sentiment pipeline.

        Args:
            email: Email for SEC EDGAR requests.
            cache_dir: Base cache directory.
            sections: Filing sections to analyze. Defaults to Risk Factors and MD&A.
        """
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / 'data'

        self.cache_dir = Path(cache_dir)
        self.sections = sections or self.DEFAULT_SECTIONS

        # Initialize loaders
        self.edgar_loader = EDGARLoader(
            email=email,
            cache_dir=self.cache_dir / 'raw' / 'edgar'
        )
        self.analyzer = FinBERTAnalyzer(
            cache_dir=self.cache_dir / 'processed' / 'sentiment'
        )

    def get_filing_sentiment(
        self,
        ticker: str,
        filing_types: List[str] = ['10-K', '10-Q'],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_filings: int = 40
    ) -> pd.DataFrame:
        """
        Get sentiment scores for a company's filings.

        Args:
            ticker: Stock ticker symbol.
            filing_types: Types of filings to analyze.
            start_date: Start date filter (YYYY-MM-DD).
            end_date: End date filter (YYYY-MM-DD).
            max_filings: Maximum filings per type.

        Returns:
            DataFrame with filing dates and sentiment scores.
        """
        all_results = []

        for filing_type in filing_types:
            try:
                filings = self.edgar_loader.get_filings(
                    ticker=ticker,
                    filing_type=filing_type,
                    count=max_filings,
                    start_date=start_date,
                    end_date=end_date
                )

                for filing in filings:
                    result = self._analyze_filing(filing)
                    if result:
                        result['ticker'] = ticker
                        result['filing_type'] = filing_type
                        result['filing_date'] = filing.filing_date
                        all_results.append(result)

            except Exception as e:
                warnings.warn(f"Error fetching {filing_type} filings for {ticker}: {e}")
                continue

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        df = df.sort_values('filing_date')

        return df

    def _analyze_filing(self, filing: Filing) -> Optional[Dict]:
        """
        Analyze sentiment of a single filing.

        Extracts text sections and analyzes each separately,
        then combines into overall filing sentiment.

        Args:
            filing: Filing object from EDGAR.

        Returns:
            Dict with sentiment scores or None if extraction fails.
        """
        try:
            # Extract text sections
            text_sections = self.edgar_loader.extract_filing_text(
                filing,
                sections=self.sections
            )

            if not text_sections:
                return None

            # Analyze each section
            section_sentiments = {}
            for section, text in text_sections.items():
                if text and len(text) > 100:
                    sentiment = self.analyzer.analyze(text)
                    section_sentiments[section] = sentiment

            if not section_sentiments:
                return None

            # Aggregate across sections
            result = self._aggregate_section_sentiments(section_sentiments)

            # Add section-specific scores
            for section, sent in section_sentiments.items():
                result[f'{section}_positive'] = sent['positive']
                result[f'{section}_negative'] = sent['negative']
                result[f'{section}_compound'] = sent['compound']

            return result

        except Exception as e:
            warnings.warn(f"Error analyzing filing {filing.accession_number}: {e}")
            return None

    def _aggregate_section_sentiments(
        self,
        section_sentiments: Dict[str, Dict]
    ) -> Dict:
        """
        Aggregate sentiment across filing sections.

        Uses simple average across sections. Could be extended
        to weight sections differently (e.g., Risk Factors more important).

        Args:
            section_sentiments: Dict mapping section names to sentiment dicts.

        Returns:
            Aggregated sentiment dict.
        """
        if not section_sentiments:
            return {}

        # Average across sections
        n = len(section_sentiments)
        result = {
            'overall_positive': sum(s['positive'] for s in section_sentiments.values()) / n,
            'overall_negative': sum(s['negative'] for s in section_sentiments.values()) / n,
            'overall_neutral': sum(s['neutral'] for s in section_sentiments.values()) / n,
            'overall_compound': sum(s['compound'] for s in section_sentiments.values()) / n,
        }

        # Determine overall label
        probs = [result['overall_positive'], result['overall_negative'], result['overall_neutral']]
        labels = ['positive', 'negative', 'neutral']
        result['overall_label'] = labels[np.argmax(probs)]

        return result

    def get_sentiment_features(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        filing_types: List[str] = ['10-K', '10-Q'],
        include_changes: bool = True
    ) -> pd.DataFrame:
        """
        Get sentiment features aligned to daily frequency.

        This is the main method for creating features for modeling.
        It returns daily sentiment features that can be merged with
        price data.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            filing_types: Types of filings to include.
            include_changes: Whether to include change features.

        Returns:
            DataFrame with daily sentiment features, forward-filled.
        """
        # Get filing-level sentiment
        filing_df = self.get_filing_sentiment(
            ticker=ticker,
            filing_types=filing_types,
            start_date=start_date,
            end_date=end_date
        )

        if filing_df.empty:
            return pd.DataFrame()

        # Create daily date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Set filing_date as index
        filing_df = filing_df.set_index('filing_date')

        # Select sentiment columns
        sentiment_cols = [c for c in filing_df.columns if any(
            x in c for x in ['positive', 'negative', 'neutral', 'compound']
        )]

        # Reindex to daily and forward-fill
        daily_df = filing_df[sentiment_cols].reindex(dates)
        daily_df = daily_df.ffill()

        # Add derived features
        if include_changes:
            daily_df = self._add_change_features(daily_df, filing_df)

        # Add days since filing
        filing_dates = filing_df.index
        daily_df['days_since_filing'] = pd.Series(
            [self._days_since_last_filing(d, filing_dates) for d in dates],
            index=dates
        )

        # Add filing indicator (1 on filing days)
        daily_df['is_filing_day'] = daily_df.index.isin(filing_dates).astype(int)

        return daily_df

    def _add_change_features(
        self,
        daily_df: pd.DataFrame,
        filing_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add sentiment change features."""
        # Sentiment change from previous filing
        for col in ['overall_compound', 'overall_positive', 'overall_negative']:
            if col in filing_df.columns:
                # Get filing-level changes
                filing_changes = filing_df[col].diff()

                # Map to daily
                change_col = f'{col}_change'
                daily_df[change_col] = np.nan
                for date in filing_df.index:
                    if date in filing_changes.index:
                        daily_df.loc[date, change_col] = filing_changes.loc[date]

                # Forward-fill
                daily_df[change_col] = daily_df[change_col].ffill()

        # Sentiment vs trailing average (surprise)
        if 'overall_compound' in filing_df.columns:
            rolling_mean = filing_df['overall_compound'].rolling(4, min_periods=1).mean()

            surprise = filing_df['overall_compound'] - rolling_mean.shift(1)
            daily_df['sentiment_surprise'] = np.nan
            for date in filing_df.index:
                if date in surprise.index:
                    daily_df.loc[date, 'sentiment_surprise'] = surprise.loc[date]
            daily_df['sentiment_surprise'] = daily_df['sentiment_surprise'].ffill()

        return daily_df

    def _days_since_last_filing(
        self,
        date: pd.Timestamp,
        filing_dates: pd.DatetimeIndex
    ) -> float:
        """Calculate days since most recent filing."""
        prior_filings = filing_dates[filing_dates <= date]
        if len(prior_filings) == 0:
            return np.nan
        return (date - prior_filings.max()).days

    def get_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Get sentiment features for multiple tickers.

        Args:
            tickers: List of ticker symbols.
            start_date: Start date.
            end_date: End date.
            show_progress: Whether to show progress.

        Returns:
            DataFrame with sentiment for all tickers, indexed by (date, ticker).
        """
        all_dfs = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(tickers, desc="Processing tickers")
            except ImportError:
                iterator = tickers
        else:
            iterator = tickers

        for ticker in iterator:
            try:
                df = self.get_sentiment_features(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date
                )
                if not df.empty:
                    df['ticker'] = ticker
                    all_dfs.append(df)
            except Exception as e:
                warnings.warn(f"Error processing {ticker}: {e}")
                continue

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs)
        combined = combined.reset_index().rename(columns={'index': 'date'})
        combined = combined.set_index(['date', 'ticker'])

        return combined


def main():
    """Example usage of FilingSentimentPipeline."""
    pipeline = FilingSentimentPipeline()

    print("Fetching sentiment for AAPL filings...")

    # Get filing-level sentiment
    filing_df = pipeline.get_filing_sentiment(
        ticker='AAPL',
        filing_types=['10-K', '10-Q'],
        start_date='2022-01-01',
        end_date='2024-01-01'
    )

    if not filing_df.empty:
        print(f"\nFound {len(filing_df)} filings:")
        print(filing_df[['filing_date', 'filing_type', 'overall_compound', 'overall_label']].head(10))

        # Get daily features
        print("\nCreating daily sentiment features...")
        daily_df = pipeline.get_sentiment_features(
            ticker='AAPL',
            start_date='2022-01-01',
            end_date='2024-01-01'
        )
        print(f"Daily features shape: {daily_df.shape}")
        print(daily_df.head())
    else:
        print("No filings found or error occurred.")


if __name__ == '__main__':
    main()

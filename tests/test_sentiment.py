"""
Unit tests for sentiment analysis module.

Tests cover:
1. FinBERT analyzer functionality
2. Filing sentiment pipeline
3. Text chunking and aggregation
4. Edge cases and error handling

Note: Some tests require the FinBERT model to be downloaded.
Tests that require the model are marked with @pytest.mark.slow
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_financial_texts():
    """Sample financial texts with known sentiment."""
    return {
        'positive': [
            "Revenue increased by 25% year over year, exceeding analyst expectations.",
            "The company reported strong earnings with significant margin expansion.",
            "We are optimistic about future growth opportunities in emerging markets.",
        ],
        'negative': [
            "Revenue declined 15% due to challenging market conditions.",
            "The company faces significant headwinds from increased competition.",
            "We expect continued pressure on margins throughout the fiscal year.",
        ],
        'neutral': [
            "The company filed its quarterly report with the SEC.",
            "Total assets were $500 million as of December 31.",
            "The board of directors held its regular meeting.",
        ]
    }


@pytest.fixture
def mock_filing():
    """Create a mock filing object."""
    from data.edgar_loader import Filing
    return Filing(
        accession_number='0001234567-24-000001',
        filing_type='10-K',
        filing_date='2024-01-15',
        company_name='Test Corp',
        cik='0001234567',
        document_url='https://www.sec.gov/test/filing.htm'
    )


# ============================================================================
# FinBERT Analyzer Tests (without model loading)
# ============================================================================

class TestFinBERTAnalyzerBasic:
    """Tests that don't require model loading."""

    def test_empty_result_structure(self):
        """Test empty result has correct structure."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()
        result = analyzer._empty_result()

        assert 'label' in result
        assert 'positive' in result
        assert 'negative' in result
        assert 'neutral' in result
        assert 'compound' in result
        assert result['label'] == 'neutral'
        assert result['neutral'] == 1.0

    def test_cache_key_generation(self):
        """Test cache key is deterministic."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()

        text = "Test text for caching"
        key1 = analyzer._get_cache_key(text)
        key2 = analyzer._get_cache_key(text)

        assert key1 == key2
        assert len(key1) == 32  # MD5 hash length

    def test_different_texts_different_keys(self):
        """Test different texts produce different cache keys."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()

        key1 = analyzer._get_cache_key("Text one")
        key2 = analyzer._get_cache_key("Text two")

        assert key1 != key2

    def test_chunk_text_short(self):
        """Test short text is not chunked."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer(max_length=512)
        text = "This is a short text."

        chunks = analyzer._chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_long(self):
        """Test long text is chunked properly."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer(max_length=50)  # Very small for testing

        # Create long text with multiple sentences
        sentences = ["This is sentence number {}.".format(i) for i in range(100)]
        text = " ".join(sentences)

        chunks = analyzer._chunk_text(text)

        assert len(chunks) > 1
        # All chunks should be non-empty
        assert all(len(c) > 0 for c in chunks)

    def test_analyze_empty_text(self):
        """Test empty text returns neutral result."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()

        result = analyzer.analyze("")

        assert result['label'] == 'neutral'
        assert result['compound'] == 0.0

    def test_analyze_whitespace_only(self):
        """Test whitespace-only text returns neutral result."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()

        result = analyzer.analyze("   \n\t  ")

        assert result['label'] == 'neutral'


# ============================================================================
# FinBERT Analyzer Tests (with mocked model)
# ============================================================================

class TestFinBERTAnalyzerMocked:
    """Tests with mocked model to avoid downloads in CI."""

    def test_aggregate_results_single(self):
        """Test aggregation with single result."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()

        results = [{'positive': 0.8, 'negative': 0.1, 'neutral': 0.1, 'compound': 0.7, 'label': 'positive'}]
        chunks = ["Single chunk"]

        aggregated = analyzer._aggregate_results(results, chunks)

        assert aggregated == results[0]

    def test_aggregate_results_multiple(self):
        """Test aggregation with multiple results."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()

        results = [
            {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1, 'compound': 0.7, 'label': 'positive'},
            {'positive': 0.2, 'negative': 0.7, 'neutral': 0.1, 'compound': -0.5, 'label': 'negative'},
        ]
        chunks = ["Chunk one", "Chunk two"]  # Equal length

        aggregated = analyzer._aggregate_results(results, chunks)

        # Should be average with equal weights
        assert aggregated['positive'] == pytest.approx(0.5, rel=0.1)
        assert aggregated['negative'] == pytest.approx(0.4, rel=0.1)
        assert 'label' in aggregated

    def test_aggregate_results_weighted(self):
        """Test aggregation weights by chunk length."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()

        results = [
            {'positive': 1.0, 'negative': 0.0, 'neutral': 0.0, 'compound': 1.0, 'label': 'positive'},
            {'positive': 0.0, 'negative': 1.0, 'neutral': 0.0, 'compound': -1.0, 'label': 'negative'},
        ]
        # First chunk is 3x longer
        chunks = ["A" * 300, "B" * 100]

        aggregated = analyzer._aggregate_results(results, chunks)

        # Should be weighted toward first (positive) result
        assert aggregated['positive'] > 0.5


# ============================================================================
# Filing Sentiment Pipeline Tests
# ============================================================================

class TestFilingSentimentPipeline:
    """Tests for the filing sentiment pipeline."""

    def test_pipeline_initialization(self, tmp_path):
        """Test pipeline initializes correctly."""
        from sentiment.filing_sentiment import FilingSentimentPipeline

        pipeline = FilingSentimentPipeline(
            email="test@example.com",
            cache_dir=tmp_path
        )

        assert pipeline.edgar_loader is not None
        assert pipeline.analyzer is not None
        assert pipeline.sections == ['item_1a', 'item_7']

    def test_aggregate_section_sentiments_empty(self):
        """Test aggregation with no sections."""
        from sentiment.filing_sentiment import FilingSentimentPipeline

        pipeline = FilingSentimentPipeline()

        result = pipeline._aggregate_section_sentiments({})

        assert result == {}

    def test_aggregate_section_sentiments_single(self):
        """Test aggregation with single section."""
        from sentiment.filing_sentiment import FilingSentimentPipeline

        pipeline = FilingSentimentPipeline()

        sections = {
            'item_1a': {'positive': 0.3, 'negative': 0.5, 'neutral': 0.2, 'compound': -0.2}
        }

        result = pipeline._aggregate_section_sentiments(sections)

        assert result['overall_positive'] == 0.3
        assert result['overall_negative'] == 0.5
        assert result['overall_compound'] == -0.2
        assert result['overall_label'] == 'negative'

    def test_aggregate_section_sentiments_multiple(self):
        """Test aggregation with multiple sections."""
        from sentiment.filing_sentiment import FilingSentimentPipeline

        pipeline = FilingSentimentPipeline()

        sections = {
            'item_1a': {'positive': 0.2, 'negative': 0.6, 'neutral': 0.2, 'compound': -0.4},
            'item_7': {'positive': 0.6, 'negative': 0.2, 'neutral': 0.2, 'compound': 0.4},
        }

        result = pipeline._aggregate_section_sentiments(sections)

        # Should average
        assert result['overall_positive'] == pytest.approx(0.4, rel=0.01)
        assert result['overall_negative'] == pytest.approx(0.4, rel=0.01)
        assert result['overall_compound'] == pytest.approx(0.0, rel=0.01)

    def test_days_since_last_filing(self):
        """Test days since filing calculation."""
        from sentiment.filing_sentiment import FilingSentimentPipeline

        pipeline = FilingSentimentPipeline()

        filing_dates = pd.DatetimeIndex(['2024-01-15', '2024-04-15'])
        test_date = pd.Timestamp('2024-02-15')

        days = pipeline._days_since_last_filing(test_date, filing_dates)

        assert days == 31  # Jan 15 to Feb 15

    def test_days_since_last_filing_no_prior(self):
        """Test days since filing with no prior filings."""
        from sentiment.filing_sentiment import FilingSentimentPipeline

        pipeline = FilingSentimentPipeline()

        filing_dates = pd.DatetimeIndex(['2024-04-15', '2024-07-15'])
        test_date = pd.Timestamp('2024-01-01')  # Before first filing

        days = pipeline._days_since_last_filing(test_date, filing_dates)

        assert np.isnan(days)


# ============================================================================
# Integration Tests (with mocked external calls)
# ============================================================================

class TestSentimentIntegration:
    """Integration tests with mocked external dependencies."""

    def test_analyze_filing_mock(self, mock_filing, tmp_path):
        """Test filing analysis with mocked components."""
        from sentiment.filing_sentiment import FilingSentimentPipeline

        pipeline = FilingSentimentPipeline(cache_dir=tmp_path)

        # Mock EDGAR loader to return sample text (>100 chars required)
        long_risk_text = "Risk factors text with some negative language about challenges. " * 10
        long_mda_text = "Management discussion with optimistic outlook for growth. " * 10

        pipeline.edgar_loader.extract_filing_text = Mock(return_value={
            'item_1a': long_risk_text,
            'item_7': long_mda_text
        })

        # Mock analyzer to return predictable results
        def mock_analyze(text):
            if 'negative' in text or 'challenges' in text:
                return {'positive': 0.2, 'negative': 0.6, 'neutral': 0.2, 'compound': -0.4, 'label': 'negative'}
            else:
                return {'positive': 0.6, 'negative': 0.2, 'neutral': 0.2, 'compound': 0.4, 'label': 'positive'}

        pipeline.analyzer.analyze = mock_analyze

        result = pipeline._analyze_filing(mock_filing)

        assert result is not None
        assert 'overall_compound' in result
        assert 'item_1a_compound' in result
        assert 'item_7_compound' in result

    def test_get_sentiment_features_structure(self, tmp_path):
        """Test that daily features have correct structure."""
        from sentiment.filing_sentiment import FilingSentimentPipeline

        pipeline = FilingSentimentPipeline(cache_dir=tmp_path)

        # Mock get_filing_sentiment to return sample data
        filing_data = pd.DataFrame({
            'filing_date': pd.to_datetime(['2024-01-15', '2024-04-15']),
            'filing_type': ['10-K', '10-Q'],
            'overall_positive': [0.3, 0.4],
            'overall_negative': [0.5, 0.3],
            'overall_neutral': [0.2, 0.3],
            'overall_compound': [-0.2, 0.1],
            'ticker': ['TEST', 'TEST']
        })

        pipeline.get_filing_sentiment = Mock(return_value=filing_data)

        daily_df = pipeline.get_sentiment_features(
            ticker='TEST',
            start_date='2024-01-01',
            end_date='2024-06-01'
        )

        assert 'overall_compound' in daily_df.columns
        assert 'days_since_filing' in daily_df.columns
        assert 'is_filing_day' in daily_df.columns
        assert len(daily_df) > 0


# ============================================================================
# Slow Tests (require model download)
# ============================================================================

@pytest.mark.slow
class TestFinBERTWithModel:
    """Tests that require the actual FinBERT model."""

    def test_analyze_positive_text(self, sample_financial_texts):
        """Test positive text is classified correctly."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()

        for text in sample_financial_texts['positive']:
            result = analyzer.analyze(text)

            # Should lean positive
            assert result['positive'] > result['negative'], f"Failed on: {text}"

    def test_analyze_negative_text(self, sample_financial_texts):
        """Test negative text is classified correctly."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()

        for text in sample_financial_texts['negative']:
            result = analyzer.analyze(text)

            # Should lean negative
            assert result['negative'] > result['positive'], f"Failed on: {text}"

    def test_compound_score_range(self, sample_financial_texts):
        """Test compound score is in valid range."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()

        all_texts = sum(sample_financial_texts.values(), [])
        for text in all_texts:
            result = analyzer.analyze(text)

            assert -1 <= result['compound'] <= 1

    def test_probabilities_sum_to_one(self, sample_financial_texts):
        """Test probabilities sum to approximately 1."""
        from sentiment.finbert_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()

        all_texts = sum(sample_financial_texts.values(), [])
        for text in all_texts:
            result = analyzer.analyze(text)

            prob_sum = result['positive'] + result['negative'] + result['neutral']
            assert prob_sum == pytest.approx(1.0, rel=0.01)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    # Run fast tests by default
    pytest.main([__file__, '-v', '-m', 'not slow'])

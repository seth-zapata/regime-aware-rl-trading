"""
FinBERT-based sentiment analysis for financial text.

FinBERT is a BERT model fine-tuned on financial communication text.
It classifies text as positive, negative, or neutral with associated
probabilities.

Why FinBERT over generic sentiment models?
-----------------------------------------
1. Trained on financial text (earnings calls, analyst reports)
2. Understands domain-specific language ("headwinds", "tailwinds")
3. Better calibrated for financial sentiment tasks
4. Captures nuance ("revenue declined but beat expectations" = mixed)

Design decisions:
-----------------
- We use the ProsusAI/finbert model (most popular, well-tested)
- Text is chunked to handle BERT's 512 token limit
- We return both class labels and probability distributions
- Caching is used to avoid re-processing the same text
"""

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

# Lazy imports for heavy dependencies
_transformers_loaded = False
_tokenizer = None
_model = None


def _load_finbert():
    """Lazy load FinBERT model and tokenizer."""
    global _transformers_loaded, _tokenizer, _model

    if _transformers_loaded:
        return _tokenizer, _model

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    print("Loading FinBERT model (this may take a moment)...")

    model_name = "ProsusAI/finbert"

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Move to GPU if available
    if torch.cuda.is_available():
        _model = _model.cuda()
        print("FinBERT loaded on GPU")
    else:
        print("FinBERT loaded on CPU")

    _model.eval()  # Set to evaluation mode
    _transformers_loaded = True

    return _tokenizer, _model


class FinBERTAnalyzer:
    """
    Analyze financial text sentiment using FinBERT.

    FinBERT outputs three classes:
    - positive: Optimistic, bullish language
    - negative: Pessimistic, bearish language
    - neutral: Factual, balanced language

    Attributes:
        max_length: Maximum tokens per chunk (BERT limit is 512)
        cache_dir: Directory for caching sentiment results
        labels: Class labels in model output order

    Example:
        >>> analyzer = FinBERTAnalyzer()
        >>> result = analyzer.analyze("Revenue increased 20% year over year.")
        >>> print(result)
        {'label': 'positive', 'positive': 0.92, 'negative': 0.03, 'neutral': 0.05}
    """

    LABELS = ['positive', 'negative', 'neutral']

    def __init__(
        self,
        max_length: int = 512,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True
    ):
        """
        Initialize FinBERT analyzer.

        Args:
            max_length: Maximum tokens per text chunk.
            cache_dir: Directory for caching results. Defaults to data/processed/sentiment/
            use_cache: Whether to cache and reuse sentiment results.
        """
        self.max_length = max_length
        self.use_cache = use_cache

        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / 'data' / 'processed' / 'sentiment'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model loaded lazily on first use
        self._tokenizer = None
        self._model = None

    def _ensure_model_loaded(self):
        """Load model if not already loaded."""
        if self._tokenizer is None:
            self._tokenizer, self._model = _load_finbert()

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached result if available."""
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, cache_key: str, result: Dict):
        """Save result to cache."""
        if not self.use_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.

        For long texts (>512 tokens), the text is chunked and
        sentiments are averaged across chunks.

        Args:
            text: Text to analyze.

        Returns:
            Dict with:
            - 'label': Most likely sentiment class
            - 'positive': Probability of positive sentiment
            - 'negative': Probability of negative sentiment
            - 'neutral': Probability of neutral sentiment
            - 'compound': Composite score (positive - negative)
        """
        if not text or len(text.strip()) < 10:
            return self._empty_result()

        # Check cache
        cache_key = self._get_cache_key(text)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        # Load model
        self._ensure_model_loaded()

        # Chunk text if needed
        chunks = self._chunk_text(text)

        if not chunks:
            return self._empty_result()

        # Analyze each chunk
        chunk_results = []
        for chunk in chunks:
            try:
                result = self._analyze_chunk(chunk)
                chunk_results.append(result)
            except Exception as e:
                warnings.warn(f"Error analyzing chunk: {e}")
                continue

        if not chunk_results:
            return self._empty_result()

        # Aggregate results (weighted average by chunk length)
        result = self._aggregate_results(chunk_results, chunks)

        # Cache result
        self._save_to_cache(cache_key, result)

        return result

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks that fit within model's max length.

        We chunk by sentences to maintain semantic coherence.
        """
        import re

        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # Rough token estimate (words * 1.3)
            sentence_length = len(sentence.split()) * 1.3

            if current_length + sentence_length > self.max_length - 50:  # Buffer
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _analyze_chunk(self, text: str) -> Dict[str, float]:
        """Analyze a single text chunk."""
        import torch

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=True
        )

        # Move to same device as model
        if next(self._model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # Map to labels
        result = {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2]),
        }

        # Add label and compound score
        result['label'] = self.LABELS[np.argmax(probs)]
        result['compound'] = result['positive'] - result['negative']

        return result

    def _aggregate_results(
        self,
        results: List[Dict[str, float]],
        chunks: List[str]
    ) -> Dict[str, float]:
        """Aggregate results from multiple chunks (weighted by length)."""
        if len(results) == 1:
            return results[0]

        # Weight by chunk length
        weights = np.array([len(c) for c in chunks[:len(results)]])
        weights = weights / weights.sum()

        aggregated = {
            'positive': sum(r['positive'] * w for r, w in zip(results, weights)),
            'negative': sum(r['negative'] * w for r, w in zip(results, weights)),
            'neutral': sum(r['neutral'] * w for r, w in zip(results, weights)),
        }

        aggregated['compound'] = aggregated['positive'] - aggregated['negative']

        # Determine overall label
        probs = [aggregated['positive'], aggregated['negative'], aggregated['neutral']]
        aggregated['label'] = self.LABELS[np.argmax(probs)]

        return aggregated

    def _empty_result(self) -> Dict[str, float]:
        """Return empty/neutral result for invalid input."""
        return {
            'label': 'neutral',
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0,
            'compound': 0.0
        }

    def analyze_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, float]]:
        """
        Analyze multiple texts.

        Args:
            texts: List of texts to analyze.
            show_progress: Whether to show progress bar.

        Returns:
            List of sentiment results.
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Analyzing sentiment")
            except ImportError:
                iterator = texts
        else:
            iterator = texts

        for text in iterator:
            result = self.analyze(text)
            results.append(result)

        return results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        prefix: str = 'sentiment_'
    ) -> pd.DataFrame:
        """
        Add sentiment columns to a DataFrame.

        Args:
            df: DataFrame with text column.
            text_column: Name of column containing text.
            prefix: Prefix for new sentiment columns.

        Returns:
            DataFrame with added sentiment columns.
        """
        df = df.copy()

        # Analyze each row
        results = self.analyze_batch(df[text_column].fillna('').tolist())

        # Add result columns
        for key in ['positive', 'negative', 'neutral', 'compound', 'label']:
            df[f'{prefix}{key}'] = [r[key] for r in results]

        return df

"""
Hidden Markov Model (HMM) based regime detection.

This module implements unsupervised regime detection using Gaussian HMM
on macroeconomic indicators. The key insight is that market regimes
(expansion, contraction, crisis) have distinct statistical properties
that can be captured by hidden states.

Why HMM?
---------
1. Regimes are not directly observable - they're latent states
2. Regime transitions are probabilistic, not deterministic
3. HMM naturally handles the temporal persistence of regimes
4. We can interpret states post-hoc by examining their characteristics

Design Decisions:
-----------------
- We use macro indicators (not price) as observations because:
  - Macro data is less noisy than daily price moves
  - Regimes are fundamentally about economic conditions
  - This creates a regime signal independent of price (useful for trading)

- We standardize features before fitting because:
  - HMM assumes Gaussian emissions
  - Different indicators have different scales (VIX: 10-80, UNRATE: 3-15)

- We use 3 regimes by default because:
  - Maps to intuitive states: expansion, contraction, crisis
  - More regimes risk overfitting with limited macro history
  - Can be validated against known economic periods
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import pickle
import warnings

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class HMMRegimeDetector:
    """
    Detects market regimes using a Gaussian Hidden Markov Model.

    The detector fits an HMM to macroeconomic indicators and identifies
    hidden states that correspond to different market regimes.

    Attributes:
        n_regimes: Number of hidden states (regimes) to detect
        features: List of feature columns to use for regime detection
        model: Fitted GaussianHMM model
        scaler: StandardScaler for feature normalization
        regime_labels: Mapping from state index to interpretable label

    Example:
        >>> detector = HMMRegimeDetector(n_regimes=3)
        >>> detector.fit(macro_df[['T10Y2Y', 'VIXCLS', 'UNRATE']])
        >>> regimes = detector.predict(macro_df[['T10Y2Y', 'VIXCLS', 'UNRATE']])
    """

    # Default features for regime detection
    DEFAULT_FEATURES = ['T10Y2Y', 'VIXCLS', 'UNRATE', 'BAA10Y']

    def __init__(
        self,
        n_regimes: int = 3,
        features: Optional[List[str]] = None,
        covariance_type: str = 'full',
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize the HMM regime detector.

        Args:
            n_regimes: Number of hidden states to learn. Default 3 maps to
                      expansion/contraction/crisis paradigm.
            features: List of column names to use. If None, uses DEFAULT_FEATURES.
            covariance_type: Type of covariance matrix. 'full' allows correlated
                           features, 'diag' assumes independence.
            n_iter: Maximum number of EM iterations for training.
            random_state: Random seed for reproducibility.
        """
        self.n_regimes = n_regimes
        self.features = features or self.DEFAULT_FEATURES
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        # Will be set during fit
        self.model: Optional[GaussianHMM] = None
        self.scaler: Optional[StandardScaler] = None
        self.regime_labels: Dict[int, str] = {}
        self.regime_stats: Dict[int, Dict] = {}

    def fit(
        self,
        data: pd.DataFrame,
        label_by_volatility: bool = True
    ) -> 'HMMRegimeDetector':
        """
        Fit the HMM to macroeconomic data.

        The fitting process:
        1. Extract and validate features
        2. Standardize features (HMM assumes Gaussian)
        3. Fit Gaussian HMM using EM algorithm
        4. Optionally label regimes by their characteristics

        Args:
            data: DataFrame with datetime index and macro indicator columns.
                  Must contain all columns specified in self.features.
            label_by_volatility: If True, automatically label regimes based on
                               VIX levels (high VIX = crisis, low = expansion).

        Returns:
            self: Fitted detector instance.

        Raises:
            ValueError: If required features are missing from data.
        """
        # Validate features exist
        missing = set(self.features) - set(data.columns)
        if missing:
            raise ValueError(f"Missing features in data: {missing}")

        # Extract features and handle missing values
        X = data[self.features].copy()

        # Forward-fill then back-fill to handle any NaN
        # (should be minimal after proper data pipeline)
        X = X.ffill().bfill()

        if X.isna().any().any():
            raise ValueError("Data contains NaN values that couldn't be filled")

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit HMM
        # Suppress convergence warnings - we'll check convergence ourselves
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state
            )
            self.model.fit(X_scaled)

        # Check convergence
        if not self.model.monitor_.converged:
            warnings.warn(
                f"HMM did not converge after {self.n_iter} iterations. "
                "Consider increasing n_iter or reducing n_regimes."
            )

        # Compute regime statistics for interpretation
        self._compute_regime_stats(data)

        # Auto-label regimes if requested
        if label_by_volatility and 'VIXCLS' in self.features:
            self._label_regimes_by_volatility()
        else:
            # Default numeric labels
            self.regime_labels = {i: f"Regime {i}" for i in range(self.n_regimes)}

        return self

    def _compute_regime_stats(self, data: pd.DataFrame) -> None:
        """Compute summary statistics for each regime."""
        # Get regime assignments
        X_scaled = self.scaler.transform(data[self.features].ffill().bfill())
        regimes = self.model.predict(X_scaled)

        # Compute stats per regime
        self.regime_stats = {}
        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_data = data[self.features].iloc[mask]

            self.regime_stats[regime] = {
                'count': int(mask.sum()),
                'proportion': float(mask.mean()),
                'means': regime_data.mean().to_dict(),
                'stds': regime_data.std().to_dict()
            }

    def _label_regimes_by_volatility(self) -> None:
        """
        Automatically label regimes based on average VIX level.

        Logic:
        - Highest avg VIX → Crisis
        - Lowest avg VIX → Expansion
        - Middle → Contraction

        This heuristic works because VIX is elevated during market stress
        and compressed during calm/bullish periods.
        """
        # Get mean VIX for each regime
        vix_means = {
            regime: stats['means'].get('VIXCLS', 0)
            for regime, stats in self.regime_stats.items()
        }

        # Sort regimes by VIX level
        sorted_regimes = sorted(vix_means.keys(), key=lambda r: vix_means[r])

        # Assign labels
        labels = ['Expansion', 'Contraction', 'Crisis']
        if self.n_regimes == 2:
            labels = ['Expansion', 'Crisis']
        elif self.n_regimes > 3:
            labels = [f"Regime {i}" for i in range(self.n_regimes)]

        self.regime_labels = {
            regime: labels[i] for i, regime in enumerate(sorted_regimes)
        }

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict regime for each observation.

        Args:
            data: DataFrame with same features used for fitting.

        Returns:
            Series with regime labels, indexed same as input data.

        Raises:
            ValueError: If model hasn't been fitted.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare features
        X = data[self.features].ffill().bfill()
        X_scaled = self.scaler.transform(X)

        # Predict hidden states
        regime_indices = self.model.predict(X_scaled)

        # Map to labels
        regime_labels = pd.Series(
            [self.regime_labels[r] for r in regime_indices],
            index=data.index,
            name='regime'
        )

        return regime_labels

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get probability of each regime for each observation.

        Useful for:
        - Soft regime assignments in models
        - Measuring regime uncertainty
        - Detecting regime transitions

        Args:
            data: DataFrame with same features used for fitting.

        Returns:
            DataFrame with probability columns for each regime.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = data[self.features].ffill().bfill()
        X_scaled = self.scaler.transform(X)

        # Get state probabilities
        proba = self.model.predict_proba(X_scaled)

        # Create DataFrame with labeled columns
        proba_df = pd.DataFrame(
            proba,
            index=data.index,
            columns=[f"prob_{self.regime_labels[i]}" for i in range(self.n_regimes)]
        )

        return proba_df

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the regime transition probability matrix.

        The (i,j) entry represents P(regime_t+1 = j | regime_t = i).

        Useful for:
        - Understanding regime persistence
        - Identifying likely transition paths
        - Regime-based position sizing

        Returns:
            DataFrame with transition probabilities.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        labels = [self.regime_labels[i] for i in range(self.n_regimes)]

        return pd.DataFrame(
            self.model.transmat_,
            index=labels,
            columns=labels
        )

    def get_regime_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each regime.

        Returns:
            DataFrame with regime characteristics.
        """
        if not self.regime_stats:
            raise ValueError("Model not fitted. Call fit() first.")

        summary_data = []
        for regime, stats in self.regime_stats.items():
            row = {
                'regime': self.regime_labels[regime],
                'count': stats['count'],
                'proportion': f"{stats['proportion']:.1%}"
            }
            for feature in self.features:
                mean = stats['means'].get(feature, np.nan)
                std = stats['stds'].get(feature, np.nan)
                row[f"{feature}_mean"] = f"{mean:.2f}"
                row[f"{feature}_std"] = f"{std:.2f}"
            summary_data.append(row)

        return pd.DataFrame(summary_data).set_index('regime')

    def save(self, path: str) -> None:
        """Save fitted model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Call fit() first.")

        state = {
            'model': self.model,
            'scaler': self.scaler,
            'n_regimes': self.n_regimes,
            'features': self.features,
            'regime_labels': self.regime_labels,
            'regime_stats': self.regime_stats
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'HMMRegimeDetector':
        """Load fitted model from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        detector = cls(
            n_regimes=state['n_regimes'],
            features=state['features']
        )
        detector.model = state['model']
        detector.scaler = state['scaler']
        detector.regime_labels = state['regime_labels']
        detector.regime_stats = state['regime_stats']

        return detector

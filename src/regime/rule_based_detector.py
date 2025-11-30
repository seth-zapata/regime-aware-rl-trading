"""
Rule-based regime detection using economic thresholds.

This module implements a simple, interpretable regime classifier using
hand-crafted rules based on macroeconomic indicators. It serves as:

1. A baseline to compare against HMM (is ML adding value?)
2. An interpretable alternative when transparency is needed
3. A sanity check - HMM regimes should roughly align with rules

Why Rule-Based Detection?
-------------------------
- **Interpretability**: "Crisis = VIX > 30 AND yield curve inverted" is
  easy to explain to stakeholders
- **No training required**: Works out-of-the-box, no risk of overfitting
- **Domain knowledge**: Encodes economic expertise directly
- **Baseline**: If HMM doesn't beat simple rules, it's not adding value

Rule Design Philosophy:
----------------------
We use thresholds derived from historical norms and economic research:

- VIX thresholds:
  - < 15: Complacency (often precedes vol spikes)
  - 15-25: Normal
  - 25-35: Elevated fear
  - > 35: Crisis/panic

- Yield curve (T10Y2Y):
  - > 1.0: Steep (healthy, growth expectations)
  - 0 to 1.0: Flat (slowing growth)
  - < 0: Inverted (recession warning)

- Credit spreads (BAA10Y):
  - < 2.0: Tight (risk-on, credit confidence)
  - 2.0-3.0: Normal
  - > 3.0: Wide (credit stress)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from enum import Enum


class MarketRegime(Enum):
    """Enumeration of market regimes with clear definitions."""
    EXPANSION = "Expansion"      # Low vol, steep curve, tight spreads
    CONTRACTION = "Contraction"  # Elevated vol or flat/inverted curve
    CRISIS = "Crisis"            # High vol, wide spreads, stress signals


class RuleBasedRegimeDetector:
    """
    Detects market regimes using rule-based classification.

    Rules are based on macroeconomic thresholds that align with
    economic intuition and historical patterns.

    Attributes:
        thresholds: Dictionary of threshold values for each indicator
        regime_priority: Order of regime checks (crisis checked first)

    Example:
        >>> detector = RuleBasedRegimeDetector()
        >>> regimes = detector.predict(macro_df)
        >>> detector.explain(macro_df.iloc[-1])  # Explain latest classification
    """

    # Default thresholds based on historical analysis
    DEFAULT_THRESHOLDS = {
        # VIX thresholds
        'vix_crisis': 35.0,       # Above this = crisis
        'vix_elevated': 25.0,     # Above this = elevated fear
        'vix_low': 15.0,          # Below this = complacency

        # Yield curve thresholds (T10Y2Y)
        'curve_inverted': 0.0,    # Below this = inverted (recession signal)
        'curve_flat': 0.5,        # Below this = flat
        'curve_steep': 1.5,       # Above this = steep (healthy)

        # Credit spread thresholds (BAA10Y)
        'spread_wide': 3.0,       # Above this = credit stress
        'spread_normal': 2.0,     # Above this = slightly elevated

        # Unemployment thresholds
        'unrate_high': 6.0,       # Above this = elevated unemployment
        'unrate_spike': 1.0,      # 3-month change above this = rapid deterioration
    }

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the rule-based detector.

        Args:
            thresholds: Custom thresholds to override defaults.
                       Only specified thresholds are overridden.
        """
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Classify each observation into a regime.

        The classification logic (in priority order):

        1. CRISIS if ANY of:
           - VIX > 35
           - Credit spreads > 3.0 AND VIX > 25
           - Unemployment spike > 1% in 3 months

        2. CONTRACTION if ANY of:
           - VIX > 25
           - Yield curve inverted (< 0)
           - Credit spreads > 2.5 AND curve < 0.5

        3. EXPANSION otherwise

        Args:
            data: DataFrame with macro indicators. Expected columns:
                  - VIXCLS (or vix): VIX index
                  - T10Y2Y (or yield_curve): Yield curve spread
                  - BAA10Y (or credit_spread): Credit spread
                  - UNRATE (optional): Unemployment rate

        Returns:
            Series with regime labels.
        """
        # Standardize column names
        df = self._standardize_columns(data)

        # Initialize all as Expansion
        regimes = pd.Series(
            MarketRegime.EXPANSION.value,
            index=data.index,
            name='regime'
        )

        # Apply rules in reverse priority (expansion → contraction → crisis)
        # so higher priority (crisis) overwrites lower

        # Contraction conditions
        contraction_mask = (
            (df['vix'] > self.thresholds['vix_elevated']) |
            (df['yield_curve'] < self.thresholds['curve_inverted']) |
            (
                (df['credit_spread'] > self.thresholds['spread_normal']) &
                (df['yield_curve'] < self.thresholds['curve_flat'])
            )
        )
        regimes[contraction_mask] = MarketRegime.CONTRACTION.value

        # Crisis conditions (highest priority)
        crisis_mask = (
            (df['vix'] > self.thresholds['vix_crisis']) |
            (
                (df['credit_spread'] > self.thresholds['spread_wide']) &
                (df['vix'] > self.thresholds['vix_elevated'])
            )
        )

        # Add unemployment spike condition if available
        if 'unrate_change_3m' in df.columns:
            crisis_mask = crisis_mask | (
                df['unrate_change_3m'] > self.thresholds['unrate_spike']
            )

        regimes[crisis_mask] = MarketRegime.CRISIS.value

        return regimes

    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Map various column names to standard names."""
        df = data.copy()

        # Column name mappings
        column_map = {
            'VIXCLS': 'vix',
            'VIX': 'vix',
            'T10Y2Y': 'yield_curve',
            'yield_curve_spread': 'yield_curve',
            'BAA10Y': 'credit_spread',
            'credit_spread': 'credit_spread',
            'UNRATE': 'unrate',
            'UNRATE_change_3m': 'unrate_change_3m',
        }

        # Rename columns that exist
        rename_dict = {
            old: new for old, new in column_map.items()
            if old in df.columns
        }
        df = df.rename(columns=rename_dict)

        # Validate required columns
        required = ['vix', 'yield_curve', 'credit_spread']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def predict_with_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify regimes and return component scores.

        Returns a DataFrame with:
        - regime: The classified regime
        - crisis_score: How "crisis-like" (0-1 scale)
        - contraction_score: How "contraction-like" (0-1 scale)
        - expansion_score: How "expansion-like" (0-1 scale)

        These scores help identify borderline cases and regime transitions.
        """
        df = self._standardize_columns(data)
        regimes = self.predict(data)

        # Compute component scores (0-1 scale)
        # Higher = more indicative of that regime

        # Crisis score: high VIX and/or wide spreads
        vix_crisis_score = np.clip(
            (df['vix'] - self.thresholds['vix_elevated']) /
            (self.thresholds['vix_crisis'] - self.thresholds['vix_elevated']),
            0, 1
        )
        spread_crisis_score = np.clip(
            (df['credit_spread'] - self.thresholds['spread_normal']) /
            (self.thresholds['spread_wide'] - self.thresholds['spread_normal']),
            0, 1
        )
        crisis_score = (vix_crisis_score + spread_crisis_score) / 2

        # Contraction score: elevated VIX or inverted curve
        vix_contraction_score = np.clip(
            (df['vix'] - self.thresholds['vix_low']) /
            (self.thresholds['vix_elevated'] - self.thresholds['vix_low']),
            0, 1
        )
        curve_contraction_score = np.clip(
            (self.thresholds['curve_steep'] - df['yield_curve']) /
            (self.thresholds['curve_steep'] - self.thresholds['curve_inverted']),
            0, 1
        )
        contraction_score = (vix_contraction_score + curve_contraction_score) / 2

        # Expansion score: inverse of stress indicators
        expansion_score = 1 - (crisis_score + contraction_score) / 2

        return pd.DataFrame({
            'regime': regimes,
            'crisis_score': crisis_score,
            'contraction_score': contraction_score,
            'expansion_score': expansion_score
        }, index=data.index)

    def explain(self, row: pd.Series) -> str:
        """
        Explain the regime classification for a single observation.

        Useful for debugging and building intuition.

        Args:
            row: Single row of data with macro indicators.

        Returns:
            Human-readable explanation string.
        """
        # Convert to DataFrame for predict
        df = pd.DataFrame([row])
        regime = self.predict(df).iloc[0]
        std_df = self._standardize_columns(df)
        r = std_df.iloc[0]

        lines = [f"Regime: {regime}", "", "Indicator values:"]

        # VIX analysis
        vix = r['vix']
        if vix > self.thresholds['vix_crisis']:
            vix_status = "CRISIS level"
        elif vix > self.thresholds['vix_elevated']:
            vix_status = "ELEVATED"
        elif vix < self.thresholds['vix_low']:
            vix_status = "LOW (complacency)"
        else:
            vix_status = "NORMAL"
        lines.append(f"  VIX: {vix:.1f} ({vix_status})")

        # Yield curve analysis
        curve = r['yield_curve']
        if curve < self.thresholds['curve_inverted']:
            curve_status = "INVERTED (recession signal)"
        elif curve < self.thresholds['curve_flat']:
            curve_status = "FLAT (slowing growth)"
        elif curve > self.thresholds['curve_steep']:
            curve_status = "STEEP (healthy)"
        else:
            curve_status = "NORMAL"
        lines.append(f"  Yield Curve: {curve:.2f}% ({curve_status})")

        # Credit spread analysis
        spread = r['credit_spread']
        if spread > self.thresholds['spread_wide']:
            spread_status = "WIDE (credit stress)"
        elif spread > self.thresholds['spread_normal']:
            spread_status = "ELEVATED"
        else:
            spread_status = "TIGHT (risk-on)"
        lines.append(f"  Credit Spread: {spread:.2f}% ({spread_status})")

        # Reasoning
        lines.extend(["", "Classification reasoning:"])

        if regime == MarketRegime.CRISIS.value:
            reasons = []
            if vix > self.thresholds['vix_crisis']:
                reasons.append(f"VIX ({vix:.1f}) > crisis threshold ({self.thresholds['vix_crisis']})")
            if spread > self.thresholds['spread_wide'] and vix > self.thresholds['vix_elevated']:
                reasons.append("Wide spreads combined with elevated VIX")
            lines.append(f"  Crisis triggered by: {'; '.join(reasons)}")

        elif regime == MarketRegime.CONTRACTION.value:
            reasons = []
            if vix > self.thresholds['vix_elevated']:
                reasons.append(f"VIX ({vix:.1f}) > elevated threshold ({self.thresholds['vix_elevated']})")
            if curve < self.thresholds['curve_inverted']:
                reasons.append(f"Yield curve inverted ({curve:.2f}%)")
            if spread > self.thresholds['spread_normal'] and curve < self.thresholds['curve_flat']:
                reasons.append("Elevated spreads with flat curve")
            lines.append(f"  Contraction triggered by: {'; '.join(reasons)}")

        else:  # Expansion
            lines.append("  Expansion: No stress indicators triggered")
            lines.append(f"    VIX ({vix:.1f}) <= {self.thresholds['vix_elevated']}")
            lines.append(f"    Curve ({curve:.2f}%) >= {self.thresholds['curve_inverted']}")

        return "\n".join(lines)

    def get_regime_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each regime in the data.

        Returns:
            DataFrame with regime counts and characteristics.
        """
        regimes = self.predict(data)
        df = self._standardize_columns(data)
        df['regime'] = regimes

        summary = df.groupby('regime').agg({
            'vix': ['count', 'mean', 'std'],
            'yield_curve': ['mean', 'std'],
            'credit_spread': ['mean', 'std']
        }).round(2)

        # Flatten column names
        summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
        summary = summary.rename(columns={'vix_count': 'count'})

        # Add proportion
        summary['proportion'] = (summary['count'] / len(data) * 100).round(1).astype(str) + '%'

        return summary

    def compare_with_hmm(
        self,
        data: pd.DataFrame,
        hmm_regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Compare rule-based regimes with HMM regimes.

        Useful for validating that HMM is learning economically
        meaningful patterns.

        Args:
            data: DataFrame with macro indicators.
            hmm_regimes: Series of HMM-predicted regimes.

        Returns:
            Confusion matrix DataFrame.
        """
        rule_regimes = self.predict(data)

        # Create confusion matrix
        from collections import Counter
        pairs = list(zip(rule_regimes, hmm_regimes))
        counts = Counter(pairs)

        # Get unique regimes
        rule_unique = sorted(rule_regimes.unique())
        hmm_unique = sorted(hmm_regimes.unique())

        matrix = pd.DataFrame(
            0,
            index=rule_unique,
            columns=hmm_unique
        )

        for (rule, hmm), count in counts.items():
            matrix.loc[rule, hmm] = count

        matrix.index.name = 'Rule-Based'
        matrix.columns.name = 'HMM'

        return matrix

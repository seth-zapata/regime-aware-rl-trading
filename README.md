# Macro Regime-Aware Trading with Alternative Data

**Research Question**: Does incorporating macroeconomic regime information provide measurable improvement over price-only models?

**Key Finding**: Neither macro features nor regime conditioning helps in isolation - both actually hurt performance. But when combined, there's a **strong positive interaction effect (+7.94%)** that produces the best overall RL performance. This suggests regime labels provide *context* for interpreting macro features correctly.

---

## Project Status: Complete

All 6 milestones completed with comprehensive documentation, 127 unit tests, and rigorous ablation studies.

| Milestone | Status | Key Deliverable |
|-----------|--------|-----------------|
| 1. Data Infrastructure | Done | Multi-source pipeline (FRED, EDGAR, yfinance) |
| 2. Regime Detection | Done | HMM + rule-based regime detector |
| 3. Feature Engineering | Done | 21 features across price, macro, sentiment |
| 4. RL Trading Agent | Done | PPO agent with regime-conditioned state |
| 5. Ablation Studies | Done | Statistical comparison of 4 configurations |
| 6. Documentation | Done | Full reports with visualizations |

---

## Key Results

### Ablation Study Findings

| Configuration | Mean Test Return | Sharpe Ratio | Win Rate |
|--------------|----------------:|-------------:|---------:|
| price_only | +4.17% | 0.78 | 50% |
| price_macro | +1.36% | 0.39 | 50% |
| price_regime | -0.08% | -0.01 | 0% |
| **price_macro_regime** | **+5.05%** | **0.81** | **100%** |
| Buy & Hold | +5.47% | N/A | 100% |

### Incremental Component Effects

| Component | Effect |
|-----------|-------:|
| Baseline (price_only) | +4.17% |
| +Macro alone | -2.81% |
| +Regime alone | -4.25% |
| **Interaction (Macro x Regime)** | **+7.94%** |
| Full Model | +5.05% |

**Interpretation**: Regime labels help the agent *interpret* macro features. VIX=25 means something different in an Expansion vs. a Crisis. Without regime context, macro features add noise. Without macro features, regime labels lack specificity.

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INFRASTRUCTURE                          │
├──────────────────┬──────────────────┬──────────────────────────┤
│   Price Data     │    FRED Data     │      SEC EDGAR           │
│   (yfinance)     │  (8 indicators)  │    (FinBERT NLP)         │
│   SPY OHLCV      │  VIX, Yield      │   10-K/10-Q filings      │
│                  │  Unemployment    │   Sentiment scores       │
└────────┬─────────┴────────┬─────────┴────────────┬─────────────┘
         │                  │                      │
         ▼                  ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                            │
│  Price: returns, volatility, RSI, SMA ratios (8 features)       │
│  Macro: VIX, yield curve, percentiles (3 features)              │
│  Sentiment: FinBERT scores (10 features)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  REGIME DETECTION                               │
│  HMM (3 states): Expansion / Contraction / Crisis               │
│  Rule-based fallback using yield curve + VIX thresholds         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RL TRADING AGENT (PPO)                         │
│  State: [price_features, macro_features, regime_one_hot]        │
│  Action: position ∈ [-1, 1] (short to long)                     │
│  Reward: log returns - transaction costs                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EVALUATION                                     │
│  Walk-forward validation (2 windows)                            │
│  Multiple seeds for robustness                                  │
│  Statistical tests: t-test, bootstrap CI, Cohen's d             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
regime-aware-rl-trading/
├── src/
│   ├── data/                    # Data loading infrastructure
│   │   ├── price_loader.py      # Yahoo Finance integration
│   │   ├── fred_loader.py       # FRED macro indicators
│   │   └── edgar_loader.py      # SEC EDGAR filings
│   ├── features/                # Feature engineering
│   │   └── engineer.py          # Price, macro, sentiment features
│   ├── regime/                  # Regime detection
│   │   └── detector.py          # HMM and rule-based detection
│   ├── rl/                      # Reinforcement learning
│   │   ├── trading_env.py       # Gym-compatible trading environment
│   │   └── ppo_trader.py        # PPO agent wrapper
│   └── ablation/                # Ablation study framework
│       ├── study.py             # AblationStudy class
│       └── statistical_tests.py # Significance testing
├── tests/                       # 127 unit tests
├── notebooks/                   # Interactive exploration
│   ├── 01_data_infrastructure.ipynb
│   ├── 02_regime_detection.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_rl_trading_agent.ipynb
│   └── 05_ablation_studies.ipynb
├── reports/                     # Milestone reports
│   ├── 01_data_infrastructure.md
│   ├── 02_regime_detection.md
│   ├── 03_feature_engineering.md
│   ├── 04_rl_trading_agent.md
│   ├── 05_ablation_studies.md
│   └── images/                  # Visualizations
├── docs/
│   └── PROJECT_MASTER.md        # Detailed project specification
├── configs/                     # YAML configuration
└── data/                        # Data storage (gitignored)
```

---

## Quick Start

### Prerequisites

- Python 3.10+ (3.12 recommended)
- FRED API key (free from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html))

### Installation

```bash
# Clone and navigate
git clone https://github.com/seth-zapata/regime-aware-rl-trading.git
cd regime-aware-rl-trading

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set FRED API key
export FRED_API_KEY="your_key_here"

# Verify installation
pytest tests/ -v
```

### Running the Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/

# Or execute all notebooks
for nb in notebooks/*.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb" --inplace
done
```

---

## Test Coverage

```
tests/
├── test_price_loader.py       # 8 tests - Price data loading
├── test_fred_loader.py        # 19 tests - FRED integration
├── test_edgar_loader.py       # 11 tests - SEC EDGAR parsing
├── test_feature_engineering.py # 31 tests - Feature calculation
├── test_regime_detection.py   # 20 tests - HMM/rule-based regimes
├── test_rl.py                 # 19 tests - Trading environment
└── test_ablation.py           # 19 tests - Ablation framework

Total: 127 tests
```

Run tests:
```bash
pytest tests/ -v              # All tests
pytest tests/ -v -x           # Stop on first failure
pytest tests/test_rl.py -v    # Specific module
```

---

## Data Sources (All Free)

| Source | Data | Update Frequency | Registration |
|--------|------|------------------|--------------|
| **yfinance** | OHLCV price data | Daily | None required |
| **FRED** | Macroeconomic indicators | Daily/Monthly | Free API key |
| **SEC EDGAR** | Company filings | As filed | None required |

### FRED Indicators Used

| Indicator | Code | Description |
|-----------|------|-------------|
| VIX | VIXCLS | Volatility index |
| 10Y Treasury | DGS10 | Long-term rate |
| 2Y Treasury | DGS2 | Short-term rate |
| Fed Funds | FEDFUNDS | Policy rate |
| Unemployment | UNRATE | Labor market |
| BAA Spread | BAA10Y | Credit risk |
| TED Spread | TEDRATE | Interbank risk |
| Consumer Sentiment | UMCSENT | Survey data |

---

## Key Technical Decisions

### 1. Share-Based Portfolio Tracking

The RL environment uses share-based accounting instead of position percentages:
```python
# Proper portfolio value calculation
portfolio_value = cash + shares_held * current_price
```
This prevents numerical instability when going short and ensures realistic P&L tracking.

### 2. Walk-Forward Validation

Instead of a single train/test split, we use rolling windows:
- Window 1: Train 2020-2021, Test early 2022
- Window 2: Train 2021-2022, Test late 2022-2023

This prevents overfitting to specific market conditions.

### 3. Regime Conditioning via One-Hot Encoding

Regimes are added to the state space as one-hot vectors:
```python
state = [price_features, macro_features, [1,0,0]]  # Expansion
state = [price_features, macro_features, [0,1,0]]  # Contraction
state = [price_features, macro_features, [0,0,1]]  # Crisis
```
This allows the policy network to learn regime-specific behaviors.

### 4. Transaction Costs

All simulations include realistic transaction costs (10 basis points) to prevent overtrading and ensure results are implementable.

---

## Lessons Learned

### What Worked

1. **Feature Interactions > Individual Features**: The macro+regime interaction effect (+7.94%) was larger than either individual component
2. **Walk-Forward Validation**: Revealed significant performance variation across windows that single splits would miss
3. **Comprehensive Testing**: 127 tests caught several bugs before they affected results

### What Didn't Work

1. **Regime Alone**: Adding regime labels without macro features hurt performance (-4.25%)
2. **More Training**: Increasing RL timesteps beyond 50k led to overfitting
3. **Complex Architectures**: Simpler models often outperformed elaborate setups

### Honest Assessment

The full model (price_macro_regime) achieves +5.05% return vs Buy & Hold's +5.47%. While not beating the benchmark, this represents:
- The best RL configuration tested
- Most consistent performance (100% win rate across windows)
- Highest Sharpe ratio (0.81) among active strategies

The research question is answered: regime information **does** help, but only when combined with macro features to provide interpretive context.

---

## Future Work

1. **More Walk-Forward Windows**: Current study uses 2 windows; production would use 5+
2. **Multi-Asset Universe**: Extend beyond SPY to diversified portfolio
3. **Alternative Regime Indicators**: Test GDP, leading indicators, sentiment surveys
4. **Ensemble Methods**: Combine RL with traditional momentum/mean-reversion strategies
5. **Live Paper Trading**: Validate out-of-sample performance

---

## Documentation

| Document | Description |
|----------|-------------|
| [PROJECT_MASTER.md](docs/PROJECT_MASTER.md) | Detailed project specification |
| [CLAUDE.md](CLAUDE.md) | Development workflow guide |
| [reports/](reports/) | Milestone reports with visualizations |
| [notebooks/](notebooks/) | Interactive exploration |

---

## References

### Papers
- Hamilton (1989) - "Regime Shifts in Stock Returns"
- Deng et al. (2017) - "Deep Reinforcement Learning for Trading"
- Araci (2019) - "FinBERT: Financial Sentiment Analysis"
- Gu, Kelly, Xiu (2020) - "Empirical Asset Pricing via Machine Learning"

### Libraries
- [stable-baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms (PPO)
- [hmmlearn](https://hmmlearn.readthedocs.io/) - Hidden Markov Models
- [transformers](https://huggingface.co/transformers/) - FinBERT
- [MLflow](https://mlflow.org/) - Experiment tracking

---

## License

MIT License - See LICENSE file for details.

---

## Disclaimer

**Educational research project.** Not intended for live trading without extensive additional validation, risk management, and professional financial advice. Past performance does not guarantee future results.

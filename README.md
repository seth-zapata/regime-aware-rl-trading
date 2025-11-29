# Macro Regime-Aware Trading with Alternative Data

**Research Question**: Does incorporating macroeconomic regime information and SEC filing data provide measurable improvement over price-only models?

An advanced quantitative finance research project exploring regime-aware trading strategies using alternative data sources (FRED macroeconomic indicators, SEC EDGAR filings) and reinforcement learning. This project extends the [baseline quant-trading-ml project](https://github.com/seth-zapata/quant-trading-ml) with a focus on better problem formulation rather than just more complex models.

**Project Status**: Active Development

---

## Overview

### Why This Project Exists

The baseline project demonstrated that **daily stock prediction from technical indicators alone is fundamentally difficult** (F1=0.17, underperformed Buy & Hold). Instead of applying more complex models to the same problem, this project takes a different approach:

1. **Better problem formulation**: Predict market *regimes* (which persist for months) instead of daily direction (which is mostly noise)
2. **Alternative data**: Use FRED macro indicators and SEC filings, which provide fundamentally different signals than price history
3. **Direct policy learning**: Use reinforcement learning to learn trading policies conditioned on regime state

### Core Innovation: Regime-Aware Trading

Markets behave differently in different regimes:
- **Expansion**: Risk-on, momentum works, buy dips
- **Contraction**: Risk-off, defensive positioning, cash
- **High Volatility**: Mean reversion, smaller positions

This project detects regimes using macroeconomic indicators and adapts trading strategy accordingly.

---

## Key Features

### 1. Multi-Source Data Pipeline

| Source | Data Type | Use Case |
|--------|-----------|----------|
| **yfinance** | Price/Volume | Base market data |
| **FRED** | Macro indicators | Regime detection |
| **SEC EDGAR** | Company filings | Sentiment signals |

All data sources are **free** - no API costs required.

### 2. Regime Detection

Two complementary approaches:
- **Hidden Markov Model (HMM)**: Probabilistic regime switching based on macro indicators
- **Rule-Based**: Simple thresholds on yield curve, VIX, unemployment

Key indicators used:
- Yield curve slope (10Y - 2Y Treasury)
- VIX volatility index
- Unemployment rate
- Fed funds rate
- Credit spreads

### 3. SEC Filing Analysis

Extract sentiment from SEC filings:
- Download 10-K, 10-Q, 8-K filings via EDGAR API
- Parse and extract key sections (Risk Factors, MD&A)
- Score sentiment using pre-trained FinBERT
- Use as additional trading signal

### 4. Reinforcement Learning Trading Agent

PPO (Proximal Policy Optimization) agent that:
- Observes price features + regime state + filing sentiment
- Learns optimal trading policy (buy/hold/sell)
- Optimizes for risk-adjusted returns (Sharpe ratio)

### 5. Rigorous Evaluation

- **Walk-forward validation**: Test across multiple time periods
- **Ablation studies**: Prove each component adds value
- **Statistical significance**: p-values and confidence intervals
- **MLflow tracking**: All experiments logged and reproducible

---

## Project Structure

```
quant-trading-advanced/
├── src/
│   ├── data/
│   │   ├── price_loader.py      # Yahoo Finance integration
│   │   ├── fred_loader.py       # FRED macro data
│   │   ├── edgar_loader.py      # SEC filings
│   │   └── data_pipeline.py     # Alignment and feature engineering
│   ├── models/
│   │   ├── regime_detector.py   # HMM and rule-based regimes
│   │   ├── sentiment_model.py   # FinBERT for filing analysis
│   │   └── rl_agent.py          # PPO trading agent
│   ├── backtesting/
│   │   ├── environment.py       # Gym trading environment
│   │   ├── metrics.py           # Performance metrics
│   │   └── walk_forward.py      # Walk-forward validation
│   └── utils/
│       ├── config.py            # YAML config loading
│       └── visualization.py     # Plotting utilities
├── configs/                     # YAML configuration files
├── tests/                       # Comprehensive test suite
├── notebooks/                   # Research and EDA notebooks
├── data/                        # Data storage (gitignored)
├── mlruns/                      # MLflow tracking (gitignored)
└── results/                     # Experiment results
```

---

## Quick Start

### Prerequisites

- Python 3.10+ (3.12 recommended)
- Git
- FRED API key (free, get from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html))

### Installation

```bash
# Navigate to project
cd /path/to/quant-trading-advanced

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set FRED API key
export FRED_API_KEY="your_key_here"

# Run tests
pytest tests/ -v

# Start MLflow UI (optional)
mlflow ui  # Access at http://localhost:5000
```

---

## Technical Approach

### Regime Detection Pipeline

```
FRED Macro Data → Feature Engineering → HMM → Regime Labels
     ↓                                          ↓
[yield_curve,          [normalize,         [expansion,
 vix,                   smooth,             contraction,
 unemployment,          lag]                high_vol]
 fed_funds]
```

### Trading Agent Architecture

```
State = [price_features, regime_one_hot, filing_sentiment, portfolio_state]
           ↓
        PPO Policy Network
           ↓
Action = {buy, hold, sell} or position_size ∈ [-1, 1]
           ↓
        Trading Environment
           ↓
Reward = risk_adjusted_return - drawdown_penalty
```

### Evaluation Framework

```
Full Dataset
    ├── Train Window 1 → Test Window 1 → Metrics 1
    ├── Train Window 2 → Test Window 2 → Metrics 2
    └── Train Window 3 → Test Window 3 → Metrics 3
                              ↓
                    Aggregate Statistics
                    (mean, std, significance tests)
```

---

## Success Criteria

### Primary Goals

| Metric | Target | Description |
|--------|--------|-------------|
| **Ablation Significance** | p < 0.05 | At least one alt data source shows significant improvement |
| **Walk-Forward Windows** | 3+ | Validated across multiple time periods |
| **Documentation** | Complete | Honest analysis of what worked and what didn't |

### Secondary Goals

| Metric | Target | Description |
|--------|--------|-------------|
| **Unit Tests** | 50+ | Comprehensive coverage of critical paths |
| **Reproducibility** | 100% | All experiments reproducible via configs |
| **Code Quality** | Clean | Modular, documented, follows best practices |

### Stretch Goals

| Metric | Target | Description |
|--------|--------|-------------|
| **Beat Buy & Hold** | 2/3 windows | Outperform in majority of test periods |
| **Sharpe Ratio** | > 1.0 | Consistent risk-adjusted returns |

---

## Key Differences from Baseline Project

| Aspect | Baseline | Advanced |
|--------|----------|----------|
| **Problem** | Daily direction | Regime detection |
| **Data** | Price only | Price + FRED + EDGAR |
| **Approach** | Supervised (LSTM) | RL + HMM + NLP |
| **Models** | LSTM, Transformer | PPO, HMM, FinBERT |
| **Infrastructure** | Manual logging | MLflow, YAML configs |
| **Success Metric** | Beat market | Prove alt data value |

---

## What This Project Demonstrates

Even if the model doesn't beat the market, this project showcases:

1. **Problem Formulation Skills**: Choosing regime detection over daily prediction shows understanding of market dynamics
2. **Alternative Data Expertise**: Working with SEC filings and FRED data - real institutional data sources
3. **Modern ML Techniques**: RL (PPO), unsupervised learning (HMM), NLP (FinBERT)
4. **Research Rigor**: Ablation studies, statistical testing, walk-forward validation
5. **Software Engineering**: Clean code, comprehensive testing, experiment tracking

These are exactly what quantitative finance firms look for in candidates.

---

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Development workflow and AI collaboration instructions
- **[docs/PROJECT_MASTER.md](docs/PROJECT_MASTER.md)** - Detailed project specification

---

## Related Projects

- [Baseline Quant Trading ML Project](https://github.com/seth-zapata/quant-trading-ml) - Foundation project demonstrating core workflow and limitations of price-only prediction

---

## References

### Key Papers
- Hamilton (1989) - "Regime Shifts in Stock Returns" (foundational HMM)
- Deng et al. (2017) - "Deep Reinforcement Learning for Trading"
- Araci (2019) - "FinBERT: Financial Sentiment Analysis"
- Gu, Kelly, Xiu (2020) - "Empirical Asset Pricing via Machine Learning"

### Libraries
- [stable-baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [hmmlearn](https://hmmlearn.readthedocs.io/) - Hidden Markov Models
- [transformers](https://huggingface.co/transformers/) - FinBERT
- [MLflow](https://mlflow.org/) - Experiment tracking

### Data Sources
- [FRED](https://fred.stlouisfed.org/) - Federal Reserve Economic Data
- [SEC EDGAR](https://www.sec.gov/edgar/) - SEC filings database
- [Yahoo Finance](https://finance.yahoo.com/) - Price data

---

## License

MIT License - See LICENSE file for details

---

## Disclaimer

**This is an educational research project.** Not intended for live trading without:
- Extensive additional testing and validation
- Proper risk management systems
- Professional financial advice
- Understanding of regulatory requirements

Past performance does not guarantee future results. Trading involves substantial risk of loss.

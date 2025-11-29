# Advanced Quantitative Trading Research - Master Project Document

**Project Goal**: Build a macro regime-aware trading system using alternative data (SEC EDGAR filings, FRED macroeconomic indicators) and reinforcement learning to demonstrate that incorporating fundamental and macro signals provides measurable improvement over price-only models.

**Context**: This is a follow-up to the completed baseline project (`quant-trading-ml/`), which proved that daily stock prediction from technical indicators alone is fundamentally difficult. This advanced project tackles the problem with institutional-level methods and a more realistic problem formulation.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Core Research Question](#core-research-question)
- [Technical Approaches](#technical-approaches)
- [Data Infrastructure](#data-infrastructure)
- [Development Workflow](#development-workflow)
- [Milestone Structure](#milestone-structure)
- [Success Criteria](#success-criteria)

---

## Project Overview

### What We Learned from Baseline Project

**Proven Challenges**:
- Daily stock direction prediction from technical indicators alone: F1=0.1745 (weak)
- Architecture changes (LSTM → Transformer) don't help when signal is weak
- Walk-forward validation shows high variance (CV=0.900) across time periods
- Post-processing (confidence filtering) can't fix fundamentally weak predictions

**Best Baseline Result**:
- LSTM with Weighted BCE Loss: 2.06% return vs 19.05% Buy & Hold
- **Conclusion**: Need stronger signal sources beyond price history

### Why This Advanced Project is Different

**Key Insight**: Instead of trying to predict daily price movements (extremely noisy), we focus on:
1. **Macro regime detection** - Identifying market regimes (expansion, contraction, high/low volatility) that persist for months
2. **Regime-conditional strategies** - Different trading behavior in different regimes
3. **Alternative data integration** - SEC filings and macro indicators provide fundamentally different signals than price

**Why This Matters for Career**:
- Hedge funds explicitly model regimes (risk-on/risk-off frameworks)
- Shows macro-financial understanding, not just ML pattern matching
- Uses institutionally-relevant data sources (EDGAR, FRED)
- Demonstrates ability to formulate better problems, not just apply more complex models

---

## Core Research Question

> **Does incorporating macroeconomic regime information and SEC filing data provide statistically significant, measurable improvement over price-only models in walk-forward validation?**

This is deliberately achievable and defensible:
- Success = proving alternative data adds value (with statistical rigor)
- NOT necessarily beating Buy & Hold (which may be impossible in efficient markets)
- Focus on **methodology and analysis quality** over absolute performance

---

## Technical Approaches

### Priority 1: Macro Regime Detection ⭐

**Core Idea**: Markets behave differently in different regimes. Detect the regime, adapt the strategy.

**Regime Indicators (from FRED)**:
- **Yield curve slope** (10Y - 2Y Treasury): Inverted = recession warning
- **VIX** (volatility index): High = risk-off, Low = risk-on
- **Unemployment rate**: Rising = contraction, Falling = expansion
- **Fed funds rate**: Tightening vs easing cycles
- **Credit spreads** (BAA - AAA): Widening = stress

**Detection Methods**:
1. **Hidden Markov Model (HMM)**: Probabilistic regime switching
2. **K-Means/Gaussian Mixture**: Cluster macro states
3. **Rule-based**: Simple thresholds (e.g., inverted yield curve = defensive)

**Implementation**:
```python
from hmmlearn import hmm

class RegimeDetector:
    """Detect market regimes from macro indicators."""

    def __init__(self, n_regimes=3):
        self.model = hmm.GaussianHMM(n_components=n_regimes)
        self.regime_names = ['expansion', 'contraction', 'high_volatility']

    def fit(self, macro_data: pd.DataFrame) -> None:
        """Fit HMM to historical macro indicators."""
        features = self._prepare_features(macro_data)
        self.model.fit(features)

    def predict_regime(self, macro_data: pd.DataFrame) -> str:
        """Predict current regime."""
        features = self._prepare_features(macro_data)
        regime_id = self.model.predict(features)[-1]
        return self.regime_names[regime_id]
```

### Priority 2: SEC EDGAR Integration 📄

**Why EDGAR**:
- **Free and reliable** - Government-maintained, no API costs
- **Rich signal** - 10-K/10-Q filings contain forward-looking statements, risk factors
- **Less crowded** - Fewer retail traders process SEC filings systematically

**Data to Extract**:
- **10-K Annual Reports**: Business description, risk factors, MD&A
- **10-Q Quarterly Reports**: Quarterly financials, management discussion
- **8-K Current Reports**: Material events (earnings, acquisitions, management changes)

**NLP Pipeline**:
1. Download filings via SEC EDGAR API
2. Parse HTML/XML to extract text sections
3. Apply pre-trained FinBERT for sentiment scoring
4. Aggregate to company-level sentiment signal

**Implementation**:
```python
import requests
from bs4 import BeautifulSoup

class EDGARLoader:
    """Load and parse SEC EDGAR filings."""

    BASE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

    def get_filings(self, ticker: str, filing_type: str = "10-K",
                    count: int = 10) -> List[Dict]:
        """Fetch recent filings for a company."""
        # SEC requires User-Agent header
        headers = {"User-Agent": "Research Project contact@example.com"}
        # ... implementation

    def extract_risk_factors(self, filing_html: str) -> str:
        """Extract Item 1A: Risk Factors section."""
        soup = BeautifulSoup(filing_html, 'html.parser')
        # ... parse and extract
```

### Priority 3: Reinforcement Learning 🤖

**Why RL for Regime-Aware Trading**:
- Learn optimal policy conditioned on current regime
- Directly optimize for risk-adjusted returns (Sharpe/Sortino)
- Handle non-stationarity better than supervised learning

**State Space**:
- Price features (returns, volatility, momentum)
- Current regime (from HMM)
- Portfolio state (position, cash, unrealized P&L)
- Macro indicators (yield curve, VIX, etc.)

**Action Space**:
- Discrete: Buy, Hold, Sell (simpler, more stable)
- Or continuous: Position size [-1, 1] (more expressive)

**Reward Function**:
```python
def calculate_reward(self, portfolio_return: float,
                     volatility: float) -> float:
    """Risk-adjusted reward (differential Sharpe ratio)."""
    # Penalize volatility, reward returns
    risk_free_rate = 0.0
    sharpe_component = (portfolio_return - risk_free_rate) / (volatility + 1e-8)

    # Add drawdown penalty
    drawdown_penalty = -0.5 * max(0, self.max_drawdown - 0.1)

    return sharpe_component + drawdown_penalty
```

**Algorithms**:
- **PPO (Proximal Policy Optimization)**: Most stable, good default
- **DQN**: For discrete actions, easier to interpret
- **A2C**: Faster training, more sample efficient

### Priority 4: Ablation Studies 🔬

**Critical for Research Credibility**:

Every component must prove its value:

| Experiment | Description | Hypothesis |
|------------|-------------|------------|
| Price Only | Baseline LSTM with just price data | Control group |
| + Regime | Add regime state to model | Regime improves consistency |
| + EDGAR | Add SEC filing sentiment | Filing sentiment adds signal |
| + FRED | Add macro indicators directly | Macro data reduces variance |
| Full Model | All components together | Combined > individual |

**Statistical Rigor**:
- Multiple train/test splits (walk-forward validation)
- Statistical significance tests (paired t-test, Wilcoxon)
- Confidence intervals on all metrics
- Document negative results honestly

---

## Data Infrastructure

### Data Sources (All Free)

| Source | Data Type | Access Method | Rate Limits |
|--------|-----------|---------------|-------------|
| **yfinance** | Price/Volume | Python library | None (reasonable use) |
| **FRED** | Macro indicators | API (free key) | 120 requests/min |
| **SEC EDGAR** | SEC filings | REST API | 10 requests/sec |

### FRED Data Series

Key macro indicators to fetch:
```python
FRED_SERIES = {
    # Yield curve
    'DGS10': '10-Year Treasury Rate',
    'DGS2': '2-Year Treasury Rate',
    'T10Y2Y': '10Y-2Y Spread (pre-calculated)',

    # Volatility
    'VIXCLS': 'VIX Volatility Index',

    # Employment
    'UNRATE': 'Unemployment Rate',
    'ICSA': 'Initial Jobless Claims',

    # Monetary policy
    'FEDFUNDS': 'Federal Funds Rate',

    # Credit conditions
    'BAA10Y': 'BAA Corporate Bond Spread',

    # Economic activity
    'INDPRO': 'Industrial Production Index',
}
```

### Data Pipeline Structure

```
data/
├── raw/                    # Never modify (immutable)
│   ├── price/              # OHLCV from yfinance
│   ├── fred/               # Macro indicators
│   └── edgar/              # SEC filings (text)
├── processed/              # Cleaned, aligned data
│   ├── features/           # Engineered features
│   ├── regimes/            # Regime labels
│   └── sentiment/          # Filing sentiment scores
└── metadata/               # Schemas, data dictionaries
```

### Data Alignment Considerations

**Critical**: Different data sources have different frequencies:
- Price: Daily
- FRED macro: Monthly (some weekly)
- SEC filings: Quarterly (with random filing dates)

**Solution**: Forward-fill macro/filing data to daily frequency, being careful about look-ahead bias (use only data available at decision time).

---

## Development Workflow

### Core Principles (Preserved from Baseline)

These worked extremely well and are preserved:

#### 1. Autonomous Execution

Claude Code executes commands autonomously without asking for permission unless absolutely necessary.

**Execute Automatically (NO permission needed)**:
- File operations: mkdir, cp, mv, rm (on project files)
- Running tests: pytest, python -m pytest
- Data processing: scripts that read/write to data/ directory
- Model training: running training scripts
- Git operations: add, commit, push (following git safety protocol)
- Building/installing: pip install
- Analysis/visualization: matplotlib, seaborn plots

**Ask for Permission (human intervention required)**:
- Destructive git operations: git reset --hard, git push --force
- System-level changes: apt-get, brew install
- Deleting large amounts of data (>100MB)
- Running commands that require sudo/root access

#### 2. Feature Implementation Cycle

1. **Implement Feature** - Write production code with proper error handling
2. **Write Tests** - Comprehensive coverage for critical paths
3. **Verify Functionality** - Run tests and manual verification
4. **Commit & Push** - Immediately after verification passes

#### 3. Checkpoints

**Decision-Making Pattern**:
1. Complete current feature (implement → test → verify → commit)
2. Determine logical next step based on dependencies
3. **Briefly state** what was completed and what's next
4. **PAUSE and wait for human's explicit go-ahead** before proceeding

### New Workflow Elements

#### Experiment Tracking with MLflow

```python
import mlflow

with mlflow.start_run(run_name="regime_ppo_v1"):
    mlflow.log_params({
        'model': 'PPO',
        'n_regimes': 3,
        'data_sources': ['price', 'fred', 'edgar'],
        'reward_function': 'sharpe'
    })

    # Training loop
    mlflow.log_metrics({
        'train_reward': train_reward,
        'val_sharpe': val_sharpe,
        'val_max_drawdown': val_mdd
    })

    mlflow.pytorch.log_model(model, "model")
```

#### Configuration Management

```yaml
# configs/regime_ppo.yaml
model:
  type: ppo
  policy_network:
    hidden_sizes: [128, 64]
  learning_rate: 0.0003

regime_detector:
  type: hmm
  n_regimes: 3
  features: ['yield_curve', 'vix', 'unemployment']

data:
  sources: ['price', 'fred', 'edgar']
  lookback_days: 60
  symbols: ['SPY']

training:
  total_timesteps: 100000
  eval_frequency: 1000
```

---

## Milestone Structure

**Note**: Internal development phases are not exposed in Git history. Commits should describe features, not timeline.

### Milestone 1: Data Infrastructure
- FRED data loader with caching
- SEC EDGAR filing downloader and parser
- Price data integration (from baseline project)
- Data alignment pipeline (daily frequency)
- Unit tests for all data loaders

### Milestone 2: Regime Detection
- HMM-based regime detector
- Rule-based regime detector (for comparison)
- Regime visualization and analysis
- Backtest regime transitions against market events
- Tests for regime consistency

### Milestone 3: SEC Filing Analysis
- EDGAR filing text extraction
- FinBERT sentiment scoring
- Sentiment aggregation pipeline
- Correlation analysis with price movements
- Tests for NLP pipeline

### Milestone 4: RL Trading Agent
- Custom Gym trading environment
- PPO agent implementation (via Stable-Baselines3)
- Regime-conditioned state space
- Multiple reward function experiments
- Walk-forward validation framework

### Milestone 5: Ablation Studies & Analysis
- Systematic component ablation
- Statistical significance testing
- Performance visualization
- Comparison with baseline project
- Final documentation

### Milestone 6: Polish & Documentation
- Code cleanup and refactoring
- Comprehensive README
- Results summary with visualizations
- Model cards for each approach
- Future work documentation

---

## Success Criteria

### Primary Success (Research Quality)

**Demonstrate measurable improvement from alternative data**:
- [ ] Ablation study shows statistically significant improvement (p < 0.05) from at least one alternative data source
- [ ] Walk-forward validation across 3+ time windows with documented variance
- [ ] Honest documentation of what worked and what didn't

### Secondary Success (Technical Quality)

**Production-quality implementation**:
- [ ] 50+ unit tests covering critical paths
- [ ] MLflow tracking for all experiments
- [ ] Reproducible via configuration files
- [ ] Clean, documented codebase

### Stretch Goals

**If time permits**:
- [ ] Beat Buy & Hold in 2/3 walk-forward windows
- [ ] Sharpe ratio > 1.0 consistently
- [ ] Docker containerization

### What "Success" Means for Career

Even if the model doesn't beat the market, this project demonstrates:
1. **Problem formulation skills** - Choosing regime detection over daily prediction
2. **Alternative data expertise** - Working with SEC filings, FRED
3. **Modern ML techniques** - RL, HMM, NLP
4. **Research rigor** - Ablation studies, statistical testing, walk-forward validation
5. **Software engineering** - Clean code, testing, experiment tracking

These are exactly what quant firms look for in candidates.

---

## Key Differences from Baseline Project

| Aspect | Baseline Project | Advanced Project |
|--------|------------------|------------------|
| **Problem** | Daily direction prediction | Regime detection + regime-conditional trading |
| **Data** | Price (OHLCV) only | Price + FRED macro + SEC filings |
| **Approach** | Supervised learning | RL + unsupervised (HMM) + supervised (sentiment) |
| **Models** | LSTM, Transformer | HMM, PPO, FinBERT |
| **Infrastructure** | Manual logging | MLflow, YAML configs |
| **Budget** | $0 | $0 (all free data sources) |
| **Success Metric** | Beat market | Prove alternative data adds value |

---

## Critical Warnings (Lessons from Baseline)

### 1. Look-Ahead Bias (Still Critical!)
- FRED data is released with a lag - use publication dates, not reference dates
- SEC filings have filing dates - only use after filing date
- Always validate with dedicated tests

### 2. Data Quality > Data Quantity
- Baseline showed more features can make things worse
- Each data source must prove its value via ablation
- Start simple, add complexity only if it helps

### 3. Regime Detection Pitfalls
- HMM can overfit to historical regimes
- Regimes are only clear in hindsight
- Use proper out-of-sample testing

### 4. RL Training Instability
- RL is notoriously unstable
- Use multiple random seeds
- Log training curves carefully
- Have fallback to simpler methods

---

## References & Resources

### Key Papers
1. "Regime Shifts in Stock Returns" (Hamilton, 1989) - Foundational HMM for finance
2. "Deep Reinforcement Learning for Trading" (Deng et al., 2017)
3. "FinBERT: Financial Sentiment Analysis" (Araci, 2019)
4. "Empirical Asset Pricing via Machine Learning" (Gu, Kelly, Xiu, 2020)

### Libraries
- **ML**: PyTorch, stable-baselines3, hmmlearn
- **NLP**: transformers (HuggingFace), beautifulsoup4
- **Data**: yfinance, fredapi, pandas
- **MLOps**: mlflow, pyyaml

### Data Sources
- [FRED](https://fred.stlouisfed.org/) - Federal Reserve Economic Data
- [SEC EDGAR](https://www.sec.gov/edgar/) - SEC filings database
- [Yahoo Finance](https://finance.yahoo.com/) - Price data

---

## Development Notes

### Getting Started

```bash
# Clone and navigate to project
git clone https://github.com/seth-zapata/quant-trading-advanced.git
cd quant-trading-advanced

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up FRED API key (free, get from https://fred.stlouisfed.org/docs/api/api_key.html)
export FRED_API_KEY="your_key_here"

# Run tests
pytest tests/ -v

# Start MLflow UI
mlflow ui  # Access at http://localhost:5000
```

### Project Structure

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

**This document should be read at the start of EVERY session** to maintain context and consistency.

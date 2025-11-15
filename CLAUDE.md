# Advanced Quantitative Trading Research - Master Project Document

**Project Goal**: Implement sophisticated quantitative trading approaches using alternative data, reinforcement learning, and multi-modal models to move beyond the limitations of technical-indicator-only prediction.

**Context**: This is a follow-up to the completed baseline project (`quant-trading-ml/`), which proved that daily stock prediction from technical indicators alone is fundamentally difficult. This advanced project tackles the problem with institutional-level methods.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Development Workflow](#development-workflow)
- [Technical Approaches](#technical-approaches)
- [Data Infrastructure](#data-infrastructure)
- [8-Week Timeline](#8-week-timeline)
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

### Goals for Advanced Project

**Primary Goal**: Achieve risk-adjusted returns **consistently better** than Buy & Hold across multiple market regimes

**Technical Goals**:
1. Integrate alternative data (sentiment, fundamentals, macro indicators)
2. Implement reinforcement learning for dynamic policy optimization
3. Build multi-modal models (price + text + fundamentals)
4. Establish robust MLOps infrastructure (experiment tracking, model versioning)
5. Test across multiple assets and time periods

**Career Goal**: Build research-level portfolio project demonstrating:
- Ability to work with heterogeneous data sources
- Modern RL techniques (PPO, A2C, DQN)
- Production ML infrastructure (MLflow, Docker, APIs)
- Systematic experimental methodology at scale

---

## Development Workflow

### Core Principles (Carried Over from Baseline Project)

These principles worked extremely well in the baseline project and should be preserved:

#### 1. **Autonomous Execution**

Claude Code should execute commands autonomously without asking for permission unless absolutely necessary.

**Execute Automatically (NO permission needed)**:
- File operations: chmod, mkdir, cp, mv, rm (on project files)
- Running tests: pytest, python -m pytest
- Data processing: scripts that read/write to data/ directory
- Model training: running training scripts
- Git operations: add, commit, push (following git safety protocol)
- Building/installing: pip install, python setup.py
- Analysis/visualization: matplotlib, seaborn plots

**Ask for Permission (human intervention required)**:
- Destructive git operations: git reset --hard, git push --force
- System-level changes: apt-get, brew install
- Deleting large amounts of data (>100MB)
- Making API calls that cost money (ONLY on first use)
- Running commands that require sudo/root access

**Communication Style**:
- Instead of: "Should I run pytest to verify the tests?"
- Just do: Run pytest, then report "✓ All 24 tests passing"

#### 2. **Feature Implementation Cycle**

**Standard workflow** (this worked perfectly):
1. **Implement Feature** - Write production code with proper error handling
2. **Write Tests** - Comprehensive coverage for critical paths
3. **Verify Functionality** - Run tests and manual verification
4. **Commit & Push** - Immediately after verification passes

**No commit without passing these checks**:
- ✅ All tests pass
- ✅ No obvious bugs or crashes
- ✅ Code follows project patterns
- ✅ Critical functions have docstrings

#### 3. **Git Workflow**

**What worked well**:
- Direct commits to master for incremental changes (acceptable for solo project)
- One commit per logical feature (bundled implementation + tests)
- Detailed commit messages with context
- Regular pushes (never let uncommitted work pile up)

**Commit granularity**:
- ✅ Bundle feature + tests in one commit
- ✅ Separate commits for docs vs functional changes
- ✅ Atomic commits that tell a story of progression

#### 4. **Claude Code Autonomy & Checkpoints**

**Decision-Making Pattern** (this worked great):
1. Complete current feature (implement → test → verify → commit)
2. Determine logical next step based on project timeline and dependencies
3. **Briefly state** what was completed and what's next
4. **PAUSE and wait for human's explicit go-ahead** before proceeding

**Example**:
- ✅ "Alternative data ingestion complete. Next: Sentiment model training. Should I proceed?"
- ❌ "Alternative data complete. Moving to sentiment model next." (doesn't wait)

#### 5. **Testing Standards**

**Test coverage guidelines**:
- Test critical paths thoroughly (data pipeline, model training, backtesting)
- Edge cases: Missing data, empty DataFrames, invalid inputs
- Temporal correctness: Validate no look-ahead bias
- Performance: Flag potential bottlenecks

**Testing approach that worked**:
- Write tests AFTER feature implementation
- Focus on critical functionality, not 100% coverage
- Integration tests for module interactions

#### 6. **Documentation Strategy**

**What worked well**:
- Inline documentation (docstrings) same commit as feature
- External documentation (README, guides) separate commits after functional work
- Progressive documentation (don't wait until end)

**New additions for advanced project**:
- Experiment tracking logs (MLflow metadata)
- Data pipeline documentation (schemas, sources, update frequency)
- Model cards (architecture, performance, limitations)

### New Workflow Elements for Advanced Project

#### 7. **Experiment Tracking**

Use MLflow from day one (learned from baseline project pain points):

```python
import mlflow

# Track every experiment
with mlflow.start_run(run_name="sentiment_lstm_v1"):
    mlflow.log_params({
        'model': 'LSTM',
        'hidden_size': 128,
        'data_sources': ['price', 'sentiment'],
        'sequence_length': 30
    })

    # Training loop
    mlflow.log_metrics({
        'train_loss': train_loss,
        'val_f1': val_f1,
        'val_sharpe': val_sharpe
    })

    # Save model
    mlflow.pytorch.log_model(model, "model")
```

**Benefits**:
- No manual logging in markdown files
- Easy comparison across experiments
- Model versioning built-in

#### 8. **Data Pipeline Management**

**Because we're using multiple data sources**, establish clear pipeline:

```
data/
├── raw/                    # Never modify (immutable)
│   ├── price/              # OHLCV from APIs
│   ├── sentiment/          # Scraped news/tweets
│   ├── fundamentals/       # Financial statements
│   └── macro/              # Fed data, yields
├── processed/              # Cleaned, aligned data
│   ├── features/           # Engineered features
│   └── sequences/          # Model-ready sequences
└── metadata/               # Schemas, update logs
```

**Pipeline pattern**:
1. Fetch → `data/raw/{source}/{date}.parquet` (immutable)
2. Process → `data/processed/features/{date}.parquet`
3. Generate sequences → `data/processed/sequences/{split}/{date}.pkl`

#### 9. **Configuration Management**

Use YAML configs (learned this lesson from baseline):

```yaml
# configs/sentiment_lstm.yaml
model:
  type: multimodal_lstm
  price_encoder:
    hidden_size: 128
    num_layers: 2
  text_encoder:
    model_name: "ProsusAI/finbert"
    max_length: 512
  fusion:
    hidden_size: 256

data:
  sources: ['price', 'sentiment_finbert']
  lookback: 30
  batch_size: 64

training:
  epochs: 50
  lr: 0.001
  early_stopping_patience: 10
```

**Benefits**:
- Easy hyperparameter tracking
- Reproducible experiments
- No hardcoded values

---

## Technical Approaches

### Priority 1: Alternative Data Integration ⭐

#### 1.1 Sentiment Analysis

**Data Sources**:
- **Free**: Reddit API (r/wallstreetbets, r/stocks), Twitter API (academic), EDGAR filings
- **Paid** (if budget allows): Bloomberg, Refinitiv, RavenPack

**Implementation Steps**:
1. Build scrapers for news/social media
2. Fine-tune FinBERT for sentiment classification
3. Create multi-modal model (price LSTM + sentiment BERT)
4. Backtest with sentiment as additional feature

**Key Papers**:
- "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"
- "Listening to Chaotic Whispers" (news-oriented stock prediction)

**Libraries**:
- `transformers` (HuggingFace)
- `praw` (Reddit API)
- `tweepy` (Twitter API)

#### 1.2 Fundamental Data

**Sources**:
- Free: Financial Modeling Prep API, Alpha Vantage
- Paid: Quandl, Polygon.io

**Features**:
- P/E ratio, EPS, revenue growth, profit margins
- Debt-to-equity, cash flow, book value

**Approach**: Combine fundamentals with technical indicators

#### 1.3 Macroeconomic Indicators

**Sources** (Free):
- FRED API (Federal Reserve Economic Data)
- Bureau of Labor Statistics
- Treasury.gov (yield curves)

**Features**:
- Fed funds rate, yield curve slope, VIX
- GDP growth, unemployment rate, inflation (CPI)

**Use Case**: Regime detection (bull/bear/sideways markets)

### Priority 2: Reinforcement Learning 🤖

**Why RL**: Instead of predicting price direction, learn optimal trading policy directly

**Algorithms to Try**:
1. **PPO** (Proximal Policy Optimization) - stable, good for continuous action spaces
2. **A2C** (Advantage Actor-Critic) - faster training than PPO
3. **DQN** (Deep Q-Network) - discrete actions (buy/sell/hold)

**Implementation** (using Stable-Baselines3):

```python
from stable_baselines3 import PPO
from gym import Env

class TradingEnv(Env):
    """Custom gym environment for stock trading."""

    def __init__(self, data, initial_balance=100000):
        self.data = data
        self.balance = initial_balance
        # Define action space: [position_size, hold_days]
        self.action_space = Box(low=-1, high=1, shape=(2,))
        # State: price history + portfolio state
        self.observation_space = Box(...)

    def step(self, action):
        # Execute trade, update portfolio
        reward = self.calculate_reward()  # Sharpe, return, etc.
        return obs, reward, done, info

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

**Reward Functions to Try**:
- Sharpe ratio (risk-adjusted)
- Sortino ratio (downside risk)
- Total return - max_drawdown (balance return and risk)

**Key Papers**:
- "Deep Reinforcement Learning for Trading" (Deng et al.)
- "Practical Deep Reinforcement Learning Approach for Stock Trading" (Xiong et al.)

### Priority 3: Multi-Modal Models 🔀

**Combine heterogeneous data sources**:

```python
class MultiModalPredictor(nn.Module):
    def __init__(self):
        # Price encoder
        self.price_encoder = nn.LSTM(price_features, 128, 2)

        # Text encoder (FinBERT)
        self.text_encoder = BertModel.from_pretrained('ProsusAI/finbert')

        # Fundamental encoder
        self.fundamental_encoder = nn.Linear(fundamental_features, 64)

        # Attention fusion
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4)

        # Classifier
        self.classifier = nn.Linear(256, 1)

    def forward(self, price, text, fundamentals):
        price_emb = self.price_encoder(price)[0][:, -1, :]
        text_emb = self.text_encoder(text).pooler_output
        fund_emb = self.fundamental_encoder(fundamentals)

        # Stack embeddings
        combined = torch.stack([price_emb, text_emb, fund_emb], dim=0)

        # Attention fusion
        fused, _ = self.attention(combined, combined, combined)

        # Prediction
        return self.classifier(fused.mean(dim=0))
```

**Fusion Strategies**:
- Early fusion (concatenate features)
- Late fusion (separate predictions, then ensemble)
- Attention fusion (learn which modality matters when)

### Priority 4: Graph Neural Networks 🕸️

**Use Case**: Model relationships between stocks

**Data**: Stock correlation networks, sector relationships, supply chain connections

**Implementation** (using PyTorch Geometric):

```python
import torch_geometric as pyg

class StockGNN(nn.Module):
    def __init__(self, num_stocks, hidden_dim):
        self.conv1 = pyg.nn.GCNConv(price_features, hidden_dim)
        self.conv2 = pyg.nn.GCNConv(hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        # x: [num_stocks, features]
        # edge_index: [2, num_edges] (correlation graph)

        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return self.predictor(x)
```

**Graph Construction**:
- Nodes: Individual stocks
- Edges: Correlation > threshold, same sector, supply chain relationships

---

## Data Infrastructure

### Required APIs & Services

**Free Tier** (Start here):
- ✅ yfinance - Price data
- ✅ FRED API - Macroeconomic data
- ✅ Reddit API - Sentiment (with academic access)
- ✅ Alpha Vantage - Fundamentals (500 calls/day)
- ✅ EDGAR - SEC filings

**Paid Tier** (If budget $100-500/month):
- Polygon.io ($200/mo) - Real-time data + fundamentals
- Financial Modeling Prep ($50/mo) - Comprehensive fundamentals
- NewsAPI ($450/mo) - News headlines
- RavenPack (institutional) - Professional sentiment

**Compute Resources**:
- **Local**: CPU training is fine for prototyping
- **Cloud**: Google Colab Pro ($10/mo) for GPU training
- **Production**: AWS EC2 spot instances (~$50/mo)

### MLOps Stack

**Experiment Tracking**: MLflow (free, open-source)
**Model Registry**: MLflow Model Registry
**Monitoring**: TensorBoard (free)
**Orchestration**: Airflow or Prefect (for data pipelines)
**Containerization**: Docker (for reproducibility)

---

## 8-Week Timeline

### Week 1-2: Data Infrastructure & Sentiment Analysis
- Set up MLflow tracking
- Build news/Reddit scrapers
- Fine-tune FinBERT for sentiment
- Create multi-modal model (price + sentiment)
- Baseline: LSTM with sentiment features

### Week 3-4: Reinforcement Learning
- Implement custom trading gym environment
- Train PPO/A2C/DQN agents
- Experiment with reward functions
- Compare RL vs supervised learning

### Week 5-6: Multi-Modal Models & GNNs
- Integrate fundamental data
- Build attention fusion model
- Implement stock correlation GNN
- Test across multiple assets (SPY, QQQ, individual stocks)

### Week 7: Production Infrastructure
- Dockerize models
- Set up automated data pipelines
- Create monitoring dashboards
- Paper trading validation

### Week 8: Evaluation & Documentation
- Walk-forward validation across ALL models
- Performance comparison vs baseline project
- Comprehensive documentation
- Portfolio presentation materials

---

## Success Criteria

### Technical Success

**Minimum Bar** (Better than baseline):
- F1 > 0.20 (vs 0.1745 baseline)
- Sharpe > 1.5 consistently across time windows
- Low variance in walk-forward validation (CV < 0.5 vs 0.900 baseline)

**Stretch Goals**:
- Beat Buy & Hold in 2/3 time windows
- Sharpe > 2.0
- Positive returns in both bull and bear markets

### Research Success

Demonstrate **systematic improvement** over baseline:
- Quantify value of each data source (ablation studies)
- Compare RL vs supervised learning rigorously
- Prove multi-modal > single-modal with statistical tests

### Career Success

Build portfolio showcasing:
- Alternative data integration (news, sentiment, fundamentals)
- Modern RL techniques (PPO, DQN)
- Production ML infrastructure (MLflow, Docker, APIs)
- Research-level experimental rigor

---

## Key Differences from Baseline Project

| Aspect | Baseline Project | Advanced Project |
|--------|------------------|------------------|
| **Data** | Price (OHLCV) only | Price + sentiment + fundamentals + macro |
| **Approach** | Supervised learning (direction prediction) | RL (policy learning) + supervised + multi-modal |
| **Models** | LSTM, Transformer | LSTM + BERT + GNN + PPO/A2C/DQN |
| **Infrastructure** | Manual logging | MLflow, Docker, automated pipelines |
| **Timeline** | 8 weeks (solo project) | 8 weeks (more complex, more resources) |
| **Budget** | $0 (all free data) | $0-500 (optional paid data/compute) |
| **Outcome** | Proof of concept (F1=0.1745) | Production-ready (target Sharpe >1.5) |

---

## Critical Warnings & Lessons from Baseline

### 1. **Look-Ahead Bias** (Still critical!)
- Multi-modal data makes this HARDER (sentiment from news can leak future info)
- Always validate: sentiment at time t uses only news from t-1 or earlier
- Dedicated tests for every data source

### 2. **Data Quality > Data Quantity**
- Baseline showed more features (48 vs 26) made things worse
- Focus on **signal** not **volume**
- Ablation studies to validate each data source adds value

### 3. **Overfitting is Easier with More Data Sources**
- More data = more ways to overfit
- Strict train/val/test splits
- Walk-forward validation is MANDATORY

### 4. **Start Simple, Add Complexity**
- Don't build full multi-modal model day 1
- Baseline: Price + sentiment LSTM first
- Then add RL, then add GNN, then add fundamentals
- Each addition must prove its value

### 5. **Budget Time for Data Wrangling**
- Scraping, cleaning, aligning different data sources takes 40-50% of project time
- APIs fail, data has gaps, schemas change
- Build robust error handling and data validation

---

## References & Resources

### Essential Papers
1. **Sentiment**: "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"
2. **RL Trading**: "Deep Reinforcement Learning for Trading" (Deng et al., 2017)
3. **Multi-Modal**: "Multimodal Deep Learning for Finance" (Chen et al., 2020)
4. **GNNs**: "Temporal Graph Networks for Deep Learning on Dynamic Graphs"

### Libraries & Tools
- **ML**: PyTorch, transformers, stable-baselines3, pytorch-geometric
- **Data**: yfinance, praw, tweepy, fredapi, pandas-datareader
- **MLOps**: mlflow, docker, tensorboard, prefect
- **Backtesting**: (reuse from baseline project)

### Courses
- "Deep Reinforcement Learning" (UC Berkeley CS285)
- "Natural Language Processing with Deep Learning" (Stanford CS224N)
- "Machine Learning for Trading" (Georgia Tech CS7646)

---

## Development Best Practices (Preserved from Baseline)

### What Worked Extremely Well
1. ✅ **Modular code** (separate data, models, strategies, backtesting)
2. ✅ **Incremental development** (small commits, frequent testing)
3. ✅ **Autonomous execution** (Claude runs tests/commits without asking)
4. ✅ **Checkpoints** (pause after each feature for human approval on next step)
5. ✅ **Honest documentation** (negative results are valuable research)

### New Additions for Advanced Project
1. ✅ **MLflow from day 1** (don't manually log experiments)
2. ✅ **Config files** (YAML for all hyperparameters)
3. ✅ **Docker** (reproducible environments)
4. ✅ **Data pipelines** (automated, versioned, monitored)

---

## Closing Notes

This advanced project is **significantly more complex** than the baseline:
- More data sources (4+ vs 1)
- More model types (RL, multi-modal, GNN vs LSTM/Transformer)
- More infrastructure (MLflow, Docker, APIs vs manual logging)

**Budget extra time** for:
- Data pipeline debugging (APIs break, data has gaps)
- RL training (hyperparameter tuning is hard)
- Multi-modal integration (aligning different data frequencies)

**But the payoff is huge**:
- Research-level portfolio project
- Demonstrates institutional-grade skills
- Much higher chance of beating market
- Production ML infrastructure experience

Good luck! Start with Week 1-2 (sentiment analysis) and build from there.

---

**This document should be read at the start of EVERY session** to maintain context and consistency.

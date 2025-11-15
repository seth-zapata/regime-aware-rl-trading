# Advanced Quantitative Trading with Machine Learning

**Building on Baseline Project - Exploring Advanced ML Techniques**

An advanced quantitative finance research project exploring state-of-the-art machine learning approaches for algorithmic trading. This project extends the [baseline quant-trading-ml project](../quant-trading-ml/) with advanced techniques including reinforcement learning, alternative data integration, and multi-modal neural architectures.

**Project Status**: Week 1 - Initial Setup

---

## Overview

This advanced project addresses the key limitations discovered in the baseline project:

### Baseline Project Limitations
- **Weak predictive signal**: F1 score of 0.17 with LSTM
- **High variance**: Walk-forward validation showed CV=0.900
- **Limited data**: Only price/volume technical indicators
- **Binary classification**: Simple UP/DOWN next-day prediction
- **Underperformed Buy & Hold**: 2.06% vs 19.05% returns

### Advanced Project Goals
1. **Alternative data integration**: News sentiment, social media, fundamentals, macroeconomic indicators
2. **Advanced ML architectures**: Reinforcement learning (PPO, A2C), multi-modal models, Graph Neural Networks
3. **Better problem formulation**: Multi-day horizons, volatility prediction, regime detection
4. **Robust evaluation**: Comprehensive walk-forward validation, transaction cost modeling
5. **Production infrastructure**: MLflow experiment tracking, Docker deployment, YAML configs

---

## Key Innovations

### 1. Multi-Modal Architecture
Combine multiple data sources in a unified model:
- **Price encoder**: LSTM for price/volume sequences
- **Text encoder**: BERT for news sentiment
- **Fundamental encoder**: MLP for balance sheet data
- **Fusion layer**: Attention mechanism to combine modalities

### 2. Reinforcement Learning
Train agents to learn optimal trading policies:
- **PPO (Proximal Policy Optimization)** for stable training
- **DQN (Deep Q-Network)** for discrete action spaces
- **Custom trading environment** with realistic market simulation

### 3. Alternative Data Pipeline
Integrate diverse data sources:
- **News sentiment**: FinBERT on financial news headlines
- **Social media**: Reddit r/wallstreetbets, Twitter financial influencers
- **Fundamental data**: P/E ratios, earnings reports, balance sheets
- **Macro indicators**: Interest rates, GDP, unemployment, VIX

### 4. Graph Neural Networks
Model relationships between assets:
- Nodes: Individual stocks
- Edges: Correlation, sector membership, supply chain relationships
- **GCN (Graph Convolutional Network)** to propagate information

---

## Project Structure

```
quant-trading-advanced/
├── src/
│   ├── data/
│   │   ├── price_data.py         # Price/volume data (Yahoo, Alpha Vantage)
│   │   ├── sentiment_data.py     # News + social media sentiment
│   │   ├── fundamental_data.py   # Financial statements, ratios
│   │   ├── macro_data.py         # Economic indicators
│   │   └── pipeline.py           # Multi-source data orchestration
│   ├── models/
│   │   ├── multimodal/           # Price + text + fundamental fusion
│   │   ├── reinforcement/        # PPO, DQN, A2C agents
│   │   ├── graph/                # GNN architectures
│   │   └── baseline/             # Improved LSTM/Transformer baselines
│   ├── strategies/
│   │   ├── rl_strategy.py        # RL agent trading strategy
│   │   ├── multimodal_strategy.py# Multi-modal model strategy
│   │   └── ensemble_strategy.py  # Combine multiple models
│   ├── backtesting/
│   │   ├── backtest.py           # Enhanced backtesting engine
│   │   ├── transaction_costs.py  # Slippage, commissions, market impact
│   │   └── metrics.py            # Performance metrics
│   └── utils/
│       ├── experiment_tracking.py# MLflow integration
│       ├── config_loader.py      # YAML config management
│       └── visualization.py      # Plotting utilities
├── configs/                      # YAML configuration files
│   ├── models/
│   │   ├── multimodal_lstm.yaml
│   │   ├── ppo_agent.yaml
│   │   └── gnn.yaml
│   └── experiments/
│       ├── sentiment_experiment.yaml
│       └── rl_experiment.yaml
├── tests/                        # Comprehensive test suite
├── notebooks/                    # Research and EDA notebooks
├── scripts/                      # Training and evaluation scripts
├── docs/                         # Documentation
├── data/                         # Data storage (gitignored)
├── models/                       # Saved model checkpoints
├── results/                      # Experiment results
├── mlruns/                       # MLflow tracking data
├── CLAUDE.md                     # AI development workflow
└── README.md                     # This file
```

---

## Technical Stack

### Core Technologies
- **Python 3.12** - Primary language
- **PyTorch 2.0+** - Deep learning framework
- **MLflow** - Experiment tracking and model registry
- **Docker** - Containerization for reproducibility
- **Gym** - RL environment interface

### Machine Learning
- **Models**: LSTM, BERT, PPO, DQN, GCN, Transformers
- **Libraries**: transformers (Hugging Face), stable-baselines3, torch-geometric
- **Optimization**: Adam, learning rate scheduling, gradient clipping

### Data & Infrastructure
- **Data**: yfinance, NewsAPI, Reddit API, SEC EDGAR
- **Storage**: pandas, parquet files, SQLite
- **Visualization**: matplotlib, seaborn, plotly
- **Testing**: pytest, hypothesis (property-based testing)

---

## Quick Start

### Prerequisites
- Python 3.8+ (3.12 recommended)
- Git
- 5-10 GB free disk space (for data + models)
- API keys (optional): NewsAPI, Reddit, Twitter

### Installation

```bash
# Clone repository
cd /home/sethz/quant-research-2025/quant-trading-advanced

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up MLflow
mlflow ui  # Access at http://localhost:5000

# Run tests
pytest tests/ -v
```

---

## Development Roadmap

### 8-Week Timeline

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1 | Setup & Data Pipeline | Multi-source data ingestion, MLflow setup |
| 2 | Sentiment Integration | FinBERT sentiment, Reddit scraper, data fusion |
| 3 | Multi-Modal Model | Price+Text encoder, attention fusion, training |
| 4 | Reinforcement Learning | PPO agent, custom trading environment |
| 5 | Advanced Techniques | GNN for multi-asset, meta-learning |
| 6 | Comprehensive Backtesting | Walk-forward validation, transaction costs |
| 7 | Comparison & Analysis | Baseline vs RL vs Multi-modal vs GNN |
| 8 | Documentation & Deployment | Docker containerization, final report |

---

## Success Criteria

### Technical Metrics
- **F1 Score > 0.20** (baseline: 0.17)
- **Sharpe Ratio > 1.5** (baseline: 0.0)
- **Outperform Buy & Hold** in at least 50% of walk-forward windows
- **Low variance**: CV < 0.5 across time periods (baseline: 0.900)

### Research Quality
- Systematic hypothesis testing with MLflow tracking
- Comprehensive ablation studies (what contributes to performance?)
- Walk-forward validation across different market regimes
- Transaction cost sensitivity analysis

### Engineering Quality
- 100+ unit tests with >90% coverage
- All experiments reproducible via config files
- Docker image for deployment
- Professional documentation and visualizations

---

## Building on Baseline Project

### What Worked (Preserved)
✅ Modular architecture (separate data, models, strategies, backtesting)
✅ Strict temporal validation (no look-ahead bias)
✅ Comprehensive testing suite
✅ Git workflow with atomic commits
✅ Honest documentation of negative results

### What Didn't Work (Improved)
❌ Limited data → ✅ Multi-source alternative data
❌ Weak predictive signal → ✅ Better problem formulation
❌ High variance → ✅ Robust walk-forward validation
❌ Simple architectures → ✅ Advanced models (RL, GNN, multi-modal)
❌ Binary classification → ✅ Multi-day horizons, volatility prediction

---

## Documentation

- **CLAUDE.md** - Development workflow and AI collaboration instructions
- **ARCHITECTURE.md** - System design and technical reference (TBD)
- **EXPERIMENTS.md** - Detailed experiment log with MLflow tracking (TBD)
- **RESULTS.md** - Comprehensive performance analysis (TBD)

---

## Related Projects

- [Baseline Quant Trading ML Project](../quant-trading-ml/) - Foundation project demonstrating core workflow

---

## License

MIT License - See LICENSE file for details

---

## Contact

**Seth Zapata**
[GitHub](https://github.com/seth-zapata) | [LinkedIn](https://linkedin.com/in/sethzapata)

---

## Disclaimer

**This is an educational research project.** Not intended for live trading without:
- Extensive additional testing and validation
- Proper risk management systems
- Professional financial advice
- Understanding of regulatory requirements

Past performance does not guarantee future results. Trading involves substantial risk of loss.

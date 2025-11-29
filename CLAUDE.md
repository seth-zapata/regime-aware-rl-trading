# CLAUDE.md - Development Workflow for Advanced Quant Trading Project

## Session Startup Protocol

**At the start of EVERY session**, read the master document to understand project context:
```
Read /home/sethz/quant-research-2025/ADVANCED_PROJECT_MASTER.md
```

This provides:
- Project goals and research question
- Technical approaches (regime detection, EDGAR, RL)
- Data infrastructure details
- Success criteria
- Current milestone status

---

## Project Overview

**Research Question**: Does incorporating macroeconomic regime information and SEC filing data provide measurable improvement over price-only models?

**Key Technical Approaches**:
1. **Macro Regime Detection**: HMM on FRED indicators (yield curve, VIX, unemployment)
2. **SEC Filing Analysis**: FinBERT sentiment on 10-K/10-Q filings from EDGAR
3. **Reinforcement Learning**: PPO agent with regime-conditioned state space
4. **Ablation Studies**: Rigorous testing of each component's value

**Data Sources** (all free):
- yfinance: Price/volume data
- FRED API: Macroeconomic indicators
- SEC EDGAR: Company filings

---

## Development Principles

### 1. Autonomous Execution

Execute commands autonomously without asking for permission unless absolutely necessary.

**Execute Automatically (NO permission needed)**:
- File operations: mkdir, cp, mv, rm (on project files)
- Running tests: pytest, python -m pytest
- Data processing: scripts that read/write to data/ directory
- Model training: running training scripts
- Git operations: add, commit, push (following git safety protocol)
- Building/installing: pip install
- Analysis/visualization: matplotlib, seaborn plots
- MLflow operations: logging experiments, starting runs

**Ask for Permission (human intervention required)**:
- Destructive git operations: git reset --hard, git push --force
- System-level changes: apt-get, brew install
- Deleting large amounts of data (>100MB)
- Running commands that require sudo/root access

**Communication Style**:
- Instead of: "Should I run pytest to verify the tests?"
- Just do: Run pytest, then report "✓ All 24 tests passing"

### 2. Feature Implementation Cycle

**Standard workflow for each feature:**

1. **Implement Feature**
   - Write production code with proper error handling
   - Follow existing code patterns and architecture
   - Add comprehensive docstrings

2. **Write Tests**
   - Tests written AFTER feature implementation
   - Focus on critical paths: data loaders, model training, backtesting
   - Edge cases: Missing data, empty DataFrames, invalid inputs
   - Temporal correctness: Validate no look-ahead bias

3. **Verify Functionality**
   - Run all tests and ensure they pass
   - Manual verification on sample data
   - Check for common issues

4. **Commit & Push**
   - Commit immediately after verification passes
   - Use clear, descriptive commit messages
   - Push to GitHub

### 3. Git Workflow

**Commit Messages**:
- Describe the feature, not the timeline
- Use conventional commit format: `feat:`, `fix:`, `test:`, `docs:`
- Bundle related changes (implementation + tests) in one commit

**Examples**:
- ✅ `feat: implement FRED data loader with macro indicators`
- ✅ `feat: add HMM-based regime detection`
- ✅ `test: add unit tests for EDGAR filing parser`
- ❌ `Week 1: data infrastructure` (don't expose timeline)

**What to Commit**:
- Source code and tests
- Configuration files (YAML)
- Documentation (README, CLAUDE.md)
- Requirements.txt updates

**What NOT to Commit** (in .gitignore):
- Data files (data/)
- Model checkpoints (except final models)
- MLflow runs (mlruns/)
- Virtual environment (venv/)
- API keys and credentials

### 4. Checkpoints

**Decision-Making Pattern**:
1. Complete current feature (implement → test → verify → commit)
2. Determine logical next step based on dependencies
3. **Briefly state** what was completed and what's next
4. **PAUSE and wait for human's explicit go-ahead** before proceeding

**Example**:
- ✅ "FRED data loader complete with 8 macro indicators. Next: Regime detection module. Should I proceed?"
- ❌ "FRED loader done. Starting regime detection." (doesn't wait)

---

## Technical Guidelines

### Data Pipeline

**Directory Structure**:
```
data/
├── raw/                    # Immutable raw data
│   ├── price/              # OHLCV from yfinance
│   ├── fred/               # Macro indicators
│   └── edgar/              # SEC filings (text)
├── processed/              # Cleaned, aligned data
│   ├── features/           # Engineered features
│   ├── regimes/            # Regime labels
│   └── sentiment/          # Filing sentiment scores
└── metadata/               # Schemas, data dictionaries
```

**Data Alignment**:
- Price: Daily frequency
- FRED: Monthly (forward-fill to daily)
- EDGAR: Quarterly (use filing date, not period date)
- **Critical**: Always use publication/filing dates to avoid look-ahead bias

### Experiment Tracking

**Use MLflow for all experiments**:
```python
import mlflow

with mlflow.start_run(run_name="experiment_name"):
    mlflow.log_params({
        'model': 'PPO',
        'n_regimes': 3,
        'data_sources': ['price', 'fred', 'edgar']
    })

    # Training...

    mlflow.log_metrics({
        'val_sharpe': sharpe,
        'val_max_drawdown': mdd
    })
```

**Benefits**:
- No manual logging in markdown
- Easy comparison across experiments
- Reproducible via logged parameters

### Configuration Management

**Use YAML configs for all hyperparameters**:
```yaml
# configs/experiment.yaml
model:
  type: ppo
  hidden_sizes: [128, 64]
  learning_rate: 0.0003

data:
  sources: ['price', 'fred', 'edgar']
  lookback_days: 60

training:
  total_timesteps: 100000
```

**Load configs**:
```python
import yaml

with open('configs/experiment.yaml') as f:
    config = yaml.safe_load(f)
```

### Testing Standards

**Test Categories**:
- **Unit tests**: Individual functions (data loaders, feature engineering)
- **Integration tests**: Module interactions (data → model → backtest)
- **Temporal tests**: Verify no look-ahead bias

**Key Areas to Test**:
- Data loaders return correct schema
- Feature engineering respects temporal order
- Regime detection produces valid labels
- RL environment follows Gym interface
- Backtest metrics calculated correctly

---

## Project Milestones

**Note**: These are internal milestones. Git commits should describe features, not timeline.

### Milestone 1: Data Infrastructure
- [ ] FRED data loader with caching
- [ ] SEC EDGAR filing downloader and parser
- [ ] Price data integration
- [ ] Data alignment pipeline
- [ ] Unit tests for all data loaders

### Milestone 2: Regime Detection
- [ ] HMM-based regime detector
- [ ] Rule-based regime detector
- [ ] Regime visualization
- [ ] Tests for regime consistency

### Milestone 3: SEC Filing Analysis
- [ ] EDGAR filing text extraction
- [ ] FinBERT sentiment scoring
- [ ] Sentiment aggregation pipeline
- [ ] Tests for NLP pipeline

### Milestone 4: RL Trading Agent
- [ ] Custom Gym trading environment
- [ ] PPO agent implementation
- [ ] Regime-conditioned state space
- [ ] Walk-forward validation framework

### Milestone 5: Ablation Studies
- [ ] Systematic component ablation
- [ ] Statistical significance testing
- [ ] Performance visualization
- [ ] Comparison with baseline project

### Milestone 6: Documentation
- [ ] Code cleanup
- [ ] Comprehensive README
- [ ] Results summary
- [ ] Future work documentation

---

## Common Commands

```bash
# Activate environment
cd /home/sethz/quant-research-2025/quant-trading-advanced
source venv/bin/activate

# Run tests
pytest tests/ -v
pytest tests/test_fred_loader.py -v  # Specific test file

# MLflow UI
mlflow ui  # http://localhost:5000

# Install new dependency
pip install package_name
pip freeze > requirements.txt

# Git operations
git add .
git commit -m "feat: description"
git push origin master
```

---

## Critical Warnings

### Look-Ahead Bias Prevention
- FRED data has publication lag - use release dates
- SEC filings have filing dates - only use after filing date
- Always test with dedicated temporal validation

### Data Quality
- FRED API requires free API key
- SEC EDGAR has rate limits (10 req/sec)
- Handle missing data explicitly

### RL Training
- RL is unstable - use multiple random seeds
- Log training curves to detect divergence
- Have fallback to simpler methods

### Overfitting
- More data sources = more overfitting risk
- Each component must prove value via ablation
- Walk-forward validation is mandatory

---

## Environment Setup

```bash
# Required environment variable
export FRED_API_KEY="your_key_here"

# Python version
python --version  # Should be 3.10+

# Key dependencies
# - PyTorch: Deep learning
# - stable-baselines3: RL algorithms
# - hmmlearn: Hidden Markov Models
# - transformers: FinBERT
# - fredapi: FRED data access
# - mlflow: Experiment tracking
```

---

**This document should be read at the start of EVERY session** to maintain context and consistency.

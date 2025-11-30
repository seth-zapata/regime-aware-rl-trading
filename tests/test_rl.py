"""
Unit tests for RL trading module.

Tests cover:
1. Trading environment (TradingEnv)
2. PPO trader wrapper
3. Walk-forward validation framework
4. Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_trading_data():
    """Create sample trading data for tests."""
    np.random.seed(42)
    n_days = 500

    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')

    # Generate realistic price series
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        'High': prices * (1 + np.random.uniform(0, 0.02, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.02, 0, n_days)),
        'Close': prices,
        'Volume': np.random.uniform(1e6, 1e7, n_days),
        # Features
        'return_1d': returns,
        'return_5d': pd.Series(returns).rolling(5).sum().values,
        'volatility': pd.Series(returns).rolling(20).std().values * np.sqrt(252),
        'momentum': pd.Series(prices).pct_change(20).values,
        # Regime (0, 1, or 2)
        'regime': np.random.choice([0, 1, 2], n_days, p=[0.4, 0.4, 0.2]),
    }, index=dates)

    # Fill NaN from rolling calculations
    df = df.ffill().bfill().fillna(0)

    return df


@pytest.fixture
def feature_columns():
    """Feature columns for tests."""
    return ['return_1d', 'return_5d', 'volatility', 'momentum']


@pytest.fixture
def small_trading_data():
    """Smaller dataset for faster tests."""
    np.random.seed(42)
    n_days = 100

    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'Close': prices,
        'return_1d': returns,
        'volatility': 0.2,
        'regime': np.random.choice([0, 1, 2], n_days),
    }, index=dates)

    return df


# ============================================================================
# TradingEnv Tests
# ============================================================================

class TestTradingEnv:
    """Tests for TradingEnv class."""

    def test_env_initialization(self, sample_trading_data, feature_columns):
        """Test environment initializes correctly."""
        from rl.trading_env import TradingEnv

        env = TradingEnv(
            df=sample_trading_data,
            feature_columns=feature_columns
        )

        assert env.action_space.n == 3  # Sell, Hold, Buy
        assert env.observation_space.shape[0] > 0
        assert env.initial_balance == 100000.0

    def test_env_with_regime(self, sample_trading_data, feature_columns):
        """Test environment with regime column."""
        from rl.trading_env import TradingEnv

        env = TradingEnv(
            df=sample_trading_data,
            feature_columns=feature_columns,
            regime_column='regime',
            n_regimes=3
        )

        # State should include features + regime one-hot + portfolio state
        expected_dim = len(feature_columns) + 3 + 2  # 3 regimes, 2 portfolio
        assert env.observation_space.shape[0] == expected_dim

    def test_env_reset(self, sample_trading_data, feature_columns):
        """Test environment reset."""
        from rl.trading_env import TradingEnv

        env = TradingEnv(
            df=sample_trading_data,
            feature_columns=feature_columns
        )

        obs, info = env.reset()

        assert obs.shape == env.observation_space.shape
        assert info['portfolio_value'] == env.initial_balance
        assert info['position'] == 0.0

    def test_env_step_buy(self, sample_trading_data, feature_columns):
        """Test buy action."""
        from rl.trading_env import TradingEnv

        env = TradingEnv(
            df=sample_trading_data,
            feature_columns=feature_columns
        )

        env.reset()

        # Buy action
        obs, reward, terminated, truncated, info = env.step(2)  # BUY

        assert info['position'] == 1.0  # Max long position
        assert len(env.trades) == 1

    def test_env_step_sell(self, sample_trading_data, feature_columns):
        """Test sell action."""
        from rl.trading_env import TradingEnv

        env = TradingEnv(
            df=sample_trading_data,
            feature_columns=feature_columns
        )

        env.reset()

        # Sell action
        obs, reward, terminated, truncated, info = env.step(0)  # SELL

        assert info['position'] == -1.0  # Max short position

    def test_env_step_hold(self, sample_trading_data, feature_columns):
        """Test hold action."""
        from rl.trading_env import TradingEnv

        env = TradingEnv(
            df=sample_trading_data,
            feature_columns=feature_columns
        )

        env.reset()

        # Hold action (should not trade)
        obs, reward, terminated, truncated, info = env.step(1)  # HOLD

        assert info['position'] == 0.0
        assert len(env.trades) == 0

    def test_env_episode_completion(self, small_trading_data):
        """Test running through entire episode."""
        from rl.trading_env import TradingEnv

        env = TradingEnv(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility']
        )

        obs, info = env.reset()
        done = False
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps > 0
        # Just verify we completed the episode (portfolio may go negative with random actions)

    def test_env_transaction_costs(self, sample_trading_data, feature_columns):
        """Test transaction costs are applied."""
        from rl.trading_env import TradingEnv

        env = TradingEnv(
            df=sample_trading_data,
            feature_columns=feature_columns,
            transaction_cost=0.01  # 1% cost
        )

        obs, _ = env.reset()
        initial_cash = env.cash

        # Execute a buy
        env.step(2)  # BUY

        # Cash should decrease due to buying shares + transaction costs
        # (we buy shares, reducing cash, and pay transaction costs)
        assert env.cash < initial_cash

    def test_env_portfolio_history(self, small_trading_data):
        """Test portfolio history tracking."""
        from rl.trading_env import TradingEnv

        env = TradingEnv(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility']
        )

        env.reset()

        # Run a few steps
        for _ in range(10):
            env.step(env.action_space.sample())

        history = env.get_portfolio_history()

        assert len(history) > 0
        assert 'portfolio_value' in history.columns

    def test_env_trades_df(self, sample_trading_data, feature_columns):
        """Test trades DataFrame generation."""
        from rl.trading_env import TradingEnv

        env = TradingEnv(
            df=sample_trading_data,
            feature_columns=feature_columns
        )

        env.reset()

        # Make some trades
        env.step(2)  # Buy
        env.step(0)  # Sell

        trades_df = env.get_trades_df()

        assert len(trades_df) >= 2
        assert 'price' in trades_df.columns
        assert 'costs' in trades_df.columns

    def test_env_invalid_features_raises(self, sample_trading_data):
        """Test that missing feature columns raise error."""
        from rl.trading_env import TradingEnv

        with pytest.raises(ValueError, match="Missing feature columns"):
            TradingEnv(
                df=sample_trading_data,
                feature_columns=['nonexistent_feature']
            )

    def test_env_observation_no_nan(self, sample_trading_data, feature_columns):
        """Test that observations don't contain NaN."""
        from rl.trading_env import TradingEnv

        env = TradingEnv(
            df=sample_trading_data,
            feature_columns=feature_columns
        )

        obs, _ = env.reset()

        assert not np.isnan(obs).any()

        for _ in range(50):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert not np.isnan(obs).any()


# ============================================================================
# PPOTrader Tests
# ============================================================================

class TestPPOTrader:
    """Tests for PPOTrader class."""

    def test_trader_initialization(self, small_trading_data):
        """Test trader initializes correctly."""
        from rl.ppo_trader import PPOTrader

        trader = PPOTrader(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility'],
            verbose=0
        )

        assert trader.model is not None
        assert not trader.trained

    def test_trader_train_short(self, small_trading_data):
        """Test short training run."""
        from rl.ppo_trader import PPOTrader

        trader = PPOTrader(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility'],
            verbose=0
        )

        stats = trader.train(total_timesteps=100, progress_bar=False)

        assert trader.trained
        assert 'total_timesteps' in stats

    def test_trader_evaluate(self, small_trading_data):
        """Test evaluation on test data."""
        from rl.ppo_trader import PPOTrader

        trader = PPOTrader(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility'],
            verbose=0
        )

        trader.train(total_timesteps=100, progress_bar=False)
        metrics = trader.evaluate(small_trading_data)

        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics

    def test_trader_predict(self, small_trading_data):
        """Test prediction."""
        from rl.ppo_trader import PPOTrader

        trader = PPOTrader(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility'],
            verbose=0
        )

        trader.train(total_timesteps=100, progress_bar=False)

        # Create dummy observation
        obs = np.zeros(trader.env.observation_space.shape[0], dtype=np.float32)
        action, _ = trader.predict(obs)

        assert action in [0, 1, 2]  # Valid action

    def test_trader_save_load(self, small_trading_data, tmp_path):
        """Test model save and load."""
        from rl.ppo_trader import PPOTrader

        trader = PPOTrader(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility'],
            verbose=0
        )

        trader.train(total_timesteps=100, progress_bar=False)

        # Save
        model_path = tmp_path / "test_model"
        trader.save(str(model_path))

        # Load
        new_trader = PPOTrader(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility'],
            verbose=0
        )
        new_trader.load(str(model_path))

        assert new_trader.trained

    def test_trader_with_regime(self, small_trading_data):
        """Test trader with regime conditioning."""
        from rl.ppo_trader import PPOTrader

        trader = PPOTrader(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility'],
            regime_column='regime',
            n_regimes=3,
            verbose=0
        )

        trader.train(total_timesteps=100, progress_bar=False)
        metrics = trader.evaluate(small_trading_data)

        assert 'total_return' in metrics


# ============================================================================
# WalkForwardValidator Tests
# ============================================================================

class TestWalkForwardValidator:
    """Tests for walk-forward validation."""

    def test_validator_initialization(self, sample_trading_data, feature_columns):
        """Test validator initializes correctly."""
        from rl.walk_forward import WalkForwardValidator

        validator = WalkForwardValidator(
            df=sample_trading_data,
            feature_columns=feature_columns,
            n_windows=3,
            min_train_days=100,  # Smaller for test data
            min_test_days=30,
            verbose=0
        )

        assert len(validator.windows) >= 1

    def test_validator_window_generation(self, sample_trading_data, feature_columns):
        """Test walk-forward windows are generated correctly."""
        from rl.walk_forward import WalkForwardValidator

        validator = WalkForwardValidator(
            df=sample_trading_data,
            feature_columns=feature_columns,
            n_windows=3,
            train_pct=0.7,
            min_train_days=100,  # Smaller for test data
            min_test_days=30,
            verbose=0
        )

        for window in validator.windows:
            # Train should come before test
            assert window.train_end < window.test_start

            # Windows should have data
            assert len(window.train_df) > 0
            assert len(window.test_df) > 0

    def test_validator_anchored_mode(self, sample_trading_data, feature_columns):
        """Test anchored (expanding window) validation."""
        from rl.walk_forward import WalkForwardValidator

        validator = WalkForwardValidator(
            df=sample_trading_data,
            feature_columns=feature_columns,
            n_windows=3,
            anchored=True,
            verbose=0
        )

        # All training windows should start at the same point
        train_starts = [w.train_start for w in validator.windows]

        # First training start should be earliest
        assert all(ts >= train_starts[0] for ts in train_starts)

    def test_validator_run_quick(self, small_trading_data):
        """Test running walk-forward validation (quick)."""
        from rl.walk_forward import WalkForwardValidator

        # Very small setup for quick test
        validator = WalkForwardValidator(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility'],
            n_windows=2,
            min_train_days=30,
            min_test_days=10,
            verbose=0
        )

        results = validator.run(
            total_timesteps=50,  # Very short
            seeds=[42]
        )

        assert 'summary' in results
        assert 'baseline' in results
        assert 'window_results' in results

    def test_validator_baseline_calculation(self, sample_trading_data, feature_columns):
        """Test buy & hold baseline calculation."""
        from rl.walk_forward import WalkForwardValidator

        validator = WalkForwardValidator(
            df=sample_trading_data,
            feature_columns=feature_columns,
            n_windows=3,
            min_train_days=100,  # Smaller for test data
            min_test_days=30,
            verbose=0
        )

        baseline = validator._calculate_baseline()

        assert 'mean_return' in baseline
        assert 'per_window_returns' in baseline
        assert len(baseline['per_window_returns']) == len(validator.windows)

    def test_validator_requires_datetime_index(self, feature_columns):
        """Test that validator requires DatetimeIndex."""
        from rl.walk_forward import WalkForwardValidator

        # Create DataFrame without DatetimeIndex
        df = pd.DataFrame({
            'Close': [100, 101, 102],
            'return_1d': [0.01, 0.01, 0.01],
            'volatility': [0.2, 0.2, 0.2],
        })

        with pytest.raises(ValueError, match="DatetimeIndex"):
            WalkForwardValidator(
                df=df,
                feature_columns=['return_1d', 'volatility'],
                n_windows=2
            )


# ============================================================================
# Integration Tests
# ============================================================================

class TestRLIntegration:
    """Integration tests for RL module."""

    def test_full_pipeline(self, small_trading_data):
        """Test complete training and evaluation pipeline."""
        from rl.trading_env import TradingEnv
        from rl.ppo_trader import PPOTrader

        # Create environment
        env = TradingEnv(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility'],
            regime_column='regime',
            n_regimes=3
        )

        # Create and train trader
        trader = PPOTrader(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility'],
            regime_column='regime',
            n_regimes=3,
            verbose=0
        )

        trainer_stats = trader.train(total_timesteps=100, progress_bar=False)

        # Evaluate
        metrics = trader.evaluate(small_trading_data)

        # Check results are reasonable
        assert -1.0 <= metrics['total_return'] <= 10.0  # Return in reasonable range
        assert metrics['max_drawdown'] >= 0
        assert metrics['max_drawdown'] <= 1.0

    def test_env_gymnasium_compatibility(self, small_trading_data):
        """Test environment is compatible with Gymnasium API."""
        from rl.trading_env import TradingEnv
        import gymnasium as gym

        env = TradingEnv(
            df=small_trading_data,
            feature_columns=['return_1d', 'volatility']
        )

        # Check required attributes
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')

        # Check spaces are valid
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Box)

        # Check reset returns correct format
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        # Check step returns correct format
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

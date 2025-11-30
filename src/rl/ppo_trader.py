"""
PPO Trading Agent wrapper for Stable-Baselines3.

This module provides a high-level interface for training and using
PPO agents for trading with regime-conditioned state spaces.

Design decisions:
-----------------
1. PPO over other algorithms: Most stable for continuous observation spaces
2. Separate train/evaluate interface for walk-forward validation
3. Built-in support for multiple random seeds (RL is high-variance)
4. Logging and checkpointing for experiment tracking
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from .trading_env import TradingEnv, create_trading_env


class TradingCallback(BaseCallback):
    """
    Custom callback for logging trading-specific metrics during training.
    """

    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called after each step."""
        # Log episode info when available
        if 'episode' in self.locals.get('infos', [{}])[0]:
            episode_info = self.locals['infos'][0]['episode']
            self.episode_rewards.append(episode_info['r'])
            self.episode_lengths.append(episode_info['l'])

            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Mean Reward (last 10) = {mean_reward:.2f}")

        return True


class PPOTrader:
    """
    High-level wrapper for PPO trading agent.

    This class provides:
    - Easy training with sensible defaults
    - Multiple random seed support
    - Evaluation on separate test data
    - Model saving/loading
    - Performance metrics calculation

    Attributes:
        env: Training environment
        model: PPO model instance
        trained: Whether model has been trained

    Example:
        >>> trader = PPOTrader(train_df, feature_columns)
        >>> trader.train(total_timesteps=100000)
        >>> metrics = trader.evaluate(test_df)
        >>> trader.save('models/ppo_trader.zip')
    """

    # Default PPO hyperparameters (tuned for trading)
    DEFAULT_PPO_PARAMS = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,  # Encourage exploration
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }

    # Default network architecture
    DEFAULT_POLICY_KWARGS = {
        'net_arch': dict(pi=[64, 64], vf=[64, 64])
    }

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        regime_column: Optional[str] = None,
        n_regimes: int = 3,
        price_column: str = 'Close',
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        ppo_params: Optional[Dict] = None,
        policy_kwargs: Optional[Dict] = None,
        seed: int = 42,
        device: str = 'auto',
        verbose: int = 0
    ):
        """
        Initialize PPO trader.

        Args:
            df: Training data with features and prices
            feature_columns: List of feature column names
            regime_column: Optional column with regime labels
            n_regimes: Number of regimes for one-hot encoding
            price_column: Column name for prices
            initial_balance: Starting capital
            transaction_cost: Trading cost as fraction
            ppo_params: Override default PPO parameters
            policy_kwargs: Override default policy network
            seed: Random seed for reproducibility
            device: Device for training ('auto', 'cpu', 'cuda')
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        self.feature_columns = feature_columns
        self.regime_column = regime_column
        self.n_regimes = n_regimes
        self.price_column = price_column
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.seed = seed
        self.device = device
        self.verbose = verbose

        # Merge parameters with defaults
        self.ppo_params = {**self.DEFAULT_PPO_PARAMS, **(ppo_params or {})}
        self.policy_kwargs = {**self.DEFAULT_POLICY_KWARGS, **(policy_kwargs or {})}

        # Create environment
        self.train_df = df
        self.env = self._create_env(df)

        # Initialize model
        self.model = PPO(
            policy='MlpPolicy',
            env=self.env,
            seed=seed,
            device=device,
            verbose=verbose,
            policy_kwargs=self.policy_kwargs,
            **self.ppo_params
        )

        self.trained = False
        self.training_history = []

    def _create_env(
        self,
        df: pd.DataFrame,
        normalize: bool = False
    ) -> DummyVecEnv:
        """
        Create vectorized environment.

        Args:
            df: Data for environment
            normalize: Whether to apply observation normalization

        Returns:
            Vectorized environment
        """
        def make_env():
            env = TradingEnv(
                df=df,
                feature_columns=self.feature_columns,
                price_column=self.price_column,
                regime_column=self.regime_column,
                n_regimes=self.n_regimes,
                initial_balance=self.initial_balance,
                transaction_cost=self.transaction_cost
            )
            return Monitor(env)

        env = DummyVecEnv([make_env])

        if normalize:
            env = VecNormalize(env, norm_obs=True, norm_reward=True)

        return env

    def train(
        self,
        total_timesteps: int = 100000,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 100,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Train the PPO agent.

        Args:
            total_timesteps: Total training steps
            callback: Optional custom callback
            log_interval: Logging frequency
            progress_bar: Whether to show progress bar

        Returns:
            Training statistics
        """
        if callback is None:
            callback = TradingCallback(verbose=self.verbose)

        if self.verbose > 0:
            print(f"Training PPO for {total_timesteps} timesteps...")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            progress_bar=progress_bar
        )

        self.trained = True

        # Collect training stats
        stats = {
            'total_timesteps': total_timesteps,
            'episodes': len(callback.episode_rewards) if hasattr(callback, 'episode_rewards') else 0,
            'mean_reward': np.mean(callback.episode_rewards) if hasattr(callback, 'episode_rewards') and callback.episode_rewards else 0,
        }

        self.training_history.append(stats)

        return stats

    def evaluate(
        self,
        df: pd.DataFrame,
        n_episodes: int = 1,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate trained agent on test data.

        Args:
            df: Test data
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions

        Returns:
            Evaluation metrics
        """
        if not self.trained:
            warnings.warn("Model has not been trained yet!")

        # Create evaluation environment
        eval_env = TradingEnv(
            df=df,
            feature_columns=self.feature_columns,
            price_column=self.price_column,
            regime_column=self.regime_column,
            n_regimes=self.n_regimes,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )

        all_returns = []
        all_sharpes = []
        all_drawdowns = []
        all_trades = []
        portfolio_histories = []

        for episode in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated

            # Collect episode metrics
            portfolio_history = eval_env.get_portfolio_history()
            portfolio_histories.append(portfolio_history)

            total_return = (eval_env.portfolio_value - self.initial_balance) / self.initial_balance
            all_returns.append(total_return)

            # Calculate Sharpe ratio
            if len(eval_env.returns_history) > 1:
                returns = np.array(eval_env.returns_history)
                sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252) + 1e-8)
                all_sharpes.append(sharpe)

                # Max drawdown
                portfolio_values = np.array(eval_env.portfolio_history)
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (peak - portfolio_values) / peak
                all_drawdowns.append(drawdown.max())
            else:
                all_sharpes.append(0.0)
                all_drawdowns.append(0.0)

            all_trades.append(len(eval_env.trades))

        # Aggregate metrics
        metrics = {
            'total_return': np.mean(all_returns),
            'total_return_std': np.std(all_returns),
            'sharpe_ratio': np.mean(all_sharpes),
            'sharpe_std': np.std(all_sharpes),
            'max_drawdown': np.mean(all_drawdowns),
            'num_trades': np.mean(all_trades),
            'final_portfolio_value': eval_env.portfolio_value,
            'n_episodes': n_episodes,
        }

        # Annualize return
        n_days = len(df)
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (252 / n_days) - 1

        # Store last portfolio history for plotting
        metrics['portfolio_history'] = portfolio_histories[-1] if portfolio_histories else None
        metrics['trades'] = eval_env.get_trades_df()

        return metrics

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Predict action for given observation.

        Args:
            obs: Observation array
            deterministic: Whether to use deterministic action

        Returns:
            action: Predicted action
            state: Model state (None for PPO)
        """
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: str):
        """
        Save model to file.

        Args:
            path: Path to save model (will add .zip extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

        if self.verbose > 0:
            print(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load model from file.

        Args:
            path: Path to model file
        """
        self.model = PPO.load(path, env=self.env, device=self.device)
        self.trained = True

        if self.verbose > 0:
            print(f"Model loaded from {path}")

    @classmethod
    def load_pretrained(
        cls,
        path: str,
        df: pd.DataFrame,
        feature_columns: List[str],
        **kwargs
    ) -> 'PPOTrader':
        """
        Load a pretrained model.

        Args:
            path: Path to saved model
            df: Data for environment
            feature_columns: Feature columns
            **kwargs: Additional arguments for PPOTrader

        Returns:
            PPOTrader instance with loaded model
        """
        trader = cls(df, feature_columns, **kwargs)
        trader.load(path)
        return trader


def train_with_multiple_seeds(
    df: pd.DataFrame,
    feature_columns: List[str],
    seeds: List[int] = [42, 123, 456],
    total_timesteps: int = 100000,
    **kwargs
) -> Dict[str, Any]:
    """
    Train multiple models with different seeds and aggregate results.

    RL training is high-variance, so using multiple seeds provides
    more robust performance estimates.

    Args:
        df: Training data
        feature_columns: Feature columns
        seeds: List of random seeds
        total_timesteps: Training steps per seed
        **kwargs: Additional arguments for PPOTrader

    Returns:
        Aggregated metrics across seeds
    """
    all_metrics = []
    models = []

    for seed in seeds:
        print(f"\n--- Training with seed {seed} ---")

        trader = PPOTrader(
            df=df,
            feature_columns=feature_columns,
            seed=seed,
            **kwargs
        )

        trader.train(total_timesteps=total_timesteps)
        models.append(trader)

        # Evaluate on training data (for comparison)
        metrics = trader.evaluate(df)
        metrics['seed'] = seed
        all_metrics.append(metrics)

    # Aggregate results
    aggregated = {
        'mean_return': np.mean([m['total_return'] for m in all_metrics]),
        'std_return': np.std([m['total_return'] for m in all_metrics]),
        'mean_sharpe': np.mean([m['sharpe_ratio'] for m in all_metrics]),
        'std_sharpe': np.std([m['sharpe_ratio'] for m in all_metrics]),
        'mean_drawdown': np.mean([m['max_drawdown'] for m in all_metrics]),
        'n_seeds': len(seeds),
        'individual_results': all_metrics,
        'models': models,
    }

    return aggregated

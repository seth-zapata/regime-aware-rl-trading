"""
Gymnasium-compatible Trading Environment for RL agents.

This environment simulates trading with:
- Regime-conditioned state space (price features + macro + regime)
- Discrete action space (Buy, Hold, Sell)
- Risk-adjusted reward function (differential Sharpe ratio)

Design decisions:
-----------------
1. Discrete actions for stability (continuous position sizing is harder to learn)
2. State includes regime as one-hot encoding for explicit conditioning
3. Reward uses differential Sharpe for risk-adjusted optimization
4. Transaction costs and slippage modeled for realism
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


class TradingEnv(gym.Env):
    """
    Trading environment compatible with Gymnasium and Stable-Baselines3.

    State space includes:
    - Price features (returns, volatility, momentum)
    - Macro features (VIX, yield curve, etc.)
    - Regime indicator (one-hot encoded)
    - Portfolio state (position, unrealized P&L)

    Action space:
    - 0: Sell (go short or reduce long)
    - 1: Hold (maintain current position)
    - 2: Buy (go long or reduce short)

    Attributes:
        df: DataFrame with features and prices
        feature_columns: List of feature column names
        initial_balance: Starting capital
        transaction_cost: Cost per trade as fraction
        reward_scaling: Scale factor for rewards

    Example:
        >>> env = TradingEnv(df, feature_columns=['return_1d', 'volatility'])
        >>> obs, info = env.reset()
        >>> action = 2  # Buy
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {'render_modes': ['human']}

    # Action constants
    SELL = 0
    HOLD = 1
    BUY = 2

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        price_column: str = 'Close',
        regime_column: Optional[str] = None,
        n_regimes: int = 3,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_position: float = 1.0,
        reward_scaling: float = 1.0,
        window_size: int = 1,
        render_mode: Optional[str] = None
    ):
        """
        Initialize trading environment.

        Args:
            df: DataFrame with OHLCV + features, indexed by date
            feature_columns: List of column names to use as features
            price_column: Column name for prices (for P&L calculation)
            regime_column: Column name for regime labels (optional)
            n_regimes: Number of possible regimes (for one-hot encoding)
            initial_balance: Starting capital
            transaction_cost: Cost per trade as fraction of trade value
            slippage: Additional cost modeling market impact
            max_position: Maximum position size as fraction of portfolio
            reward_scaling: Scale factor for rewards (helps training stability)
            window_size: Number of historical observations in state
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__()

        # Store data
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.price_column = price_column
        self.regime_column = regime_column
        self.n_regimes = n_regimes

        # Trading parameters
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        self.render_mode = render_mode

        # Validate data
        self._validate_data()

        # Calculate observation and action space dimensions
        self.n_features = len(feature_columns)
        self.n_portfolio_features = 2  # position, unrealized_pnl_pct

        # State dimension: features + regime one-hot + portfolio state
        if regime_column:
            self.state_dim = self.n_features + n_regimes + self.n_portfolio_features
        else:
            self.state_dim = self.n_features + self.n_portfolio_features

        # Define spaces
        self.action_space = spaces.Discrete(3)  # Sell, Hold, Buy
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Episode state (initialized in reset)
        self.current_step = 0
        self.position = 0.0  # Current position (-1 to 1, where 1 = fully long, -1 = fully short)
        self.entry_price = 0.0
        self.portfolio_value = initial_balance

        # Track shares held (positive for long, negative for short)
        # This enables proper P&L calculation
        self.shares_held = 0.0
        self.cash = initial_balance  # Cash available (separate from position value)

        # For differential Sharpe reward
        self.returns_history = []
        self.prev_portfolio_value = initial_balance

        # Tracking for analysis
        self.trades = []
        self.portfolio_history = []

    def _validate_data(self):
        """Validate input data has required columns."""
        missing = [c for c in self.feature_columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        if self.price_column not in self.df.columns:
            raise ValueError(f"Missing price column: {self.price_column}")

        if self.regime_column and self.regime_column not in self.df.columns:
            raise ValueError(f"Missing regime column: {self.regime_column}")

        # Check for NaN values
        if self.df[self.feature_columns].isna().any().any():
            raise ValueError("Feature columns contain NaN values")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial state observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Reset episode state
        self.current_step = self.window_size  # Start after warmup
        self.position = 0.0
        self.entry_price = 0.0
        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance

        # Reset position tracking
        self.shares_held = 0.0
        self.cash = self.initial_balance

        # Reset tracking
        self.returns_history = []
        self.trades = []
        self.portfolio_history = [self.initial_balance]

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step.

        Args:
            action: Trading action (0=Sell, 1=Hold, 2=Buy)

        Returns:
            observation: New state after action
            reward: Reward for this step
            terminated: Whether episode ended (e.g., bankruptcy)
            truncated: Whether episode truncated (e.g., end of data)
            info: Additional information
        """
        # Get current and next prices
        current_price = self.df.iloc[self.current_step][self.price_column]

        # Execute action
        prev_position = self.position
        self._execute_action(action, current_price)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.portfolio_value <= 0  # Bankruptcy
        truncated = self.current_step >= len(self.df) - 1  # End of data

        # Calculate reward
        if not (terminated or truncated):
            next_price = self.df.iloc[self.current_step][self.price_column]
            reward = self._calculate_reward(current_price, next_price, prev_position)
        else:
            reward = 0.0

        # Get new observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int, price: float):
        """
        Execute trading action with proper share-based accounting.

        The model works as follows:
        - position: -1 (fully short) to +1 (fully long), representing target allocation
        - shares_held: actual number of shares (positive=long, negative=short)
        - cash: available cash (increases when selling, decreases when buying)
        - portfolio_value: cash + (shares_held * price)

        Args:
            action: Trading action (0=Sell, 1=Hold, 2=Buy)
            price: Current price
        """
        prev_position = self.position
        target_position = 0.0

        if action == self.SELL:
            target_position = -self.max_position
        elif action == self.HOLD:
            target_position = self.position  # Maintain current
        elif action == self.BUY:
            target_position = self.max_position

        # Calculate position change
        position_change = target_position - self.position

        if abs(position_change) > 0.01:  # Threshold to avoid tiny trades
            # Calculate current portfolio value before trade
            current_portfolio_value = self.cash + self.shares_held * price

            # Calculate target shares based on target position
            # target_position of 1.0 means 100% of portfolio in shares
            target_value_in_shares = target_position * current_portfolio_value
            target_shares = target_value_in_shares / price

            # Calculate shares to trade
            shares_to_trade = target_shares - self.shares_held
            trade_value = abs(shares_to_trade * price)

            # Calculate transaction costs
            costs = trade_value * (self.transaction_cost + self.slippage)

            # Update cash: selling adds cash, buying reduces cash
            # shares_to_trade > 0 means buying, < 0 means selling
            self.cash -= (shares_to_trade * price) + costs

            # Update shares held
            self.shares_held = target_shares

            # Record trade
            self.trades.append({
                'step': self.current_step,
                'price': price,
                'prev_position': prev_position,
                'new_position': target_position,
                'shares_traded': shares_to_trade,
                'costs': costs
            })

            # Update position
            self.position = target_position
            if target_position != 0 and prev_position == 0:
                self.entry_price = price

    def _calculate_reward(
        self,
        current_price: float,
        next_price: float,
        prev_position: float
    ) -> float:
        """
        Calculate risk-adjusted reward using differential Sharpe ratio.

        The differential Sharpe ratio rewards both returns and low volatility,
        encouraging consistent performance over high-variance strategies.

        Args:
            current_price: Price at action time
            next_price: Price after action
            prev_position: Position before action

        Returns:
            Scaled reward
        """
        # Update portfolio value based on new price
        # portfolio_value = cash + shares_held * price
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash + self.shares_held * next_price

        # Portfolio return for this step
        if self.prev_portfolio_value > 0:
            portfolio_return = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        else:
            portfolio_return = 0.0

        self.returns_history.append(portfolio_return)

        # Track portfolio value
        self.portfolio_history.append(self.portfolio_value)

        # Differential Sharpe ratio
        # This incrementally updates the Sharpe ratio estimate
        if len(self.returns_history) < 2:
            reward = portfolio_return
        else:
            returns = np.array(self.returns_history)
            std_return = returns.std() + 1e-8

            # Sharpe-like reward: penalize volatility
            reward = portfolio_return - 0.5 * std_return

        # Add drawdown penalty
        if len(self.portfolio_history) > 1:
            peak = max(self.portfolio_history)
            if peak > 0:
                drawdown = (peak - self.portfolio_value) / peak
                reward -= 0.1 * max(0, drawdown)  # Only penalize positive drawdowns

        return reward * self.reward_scaling

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation from current state.

        Returns:
            Observation array containing features, regime, and portfolio state
        """
        # Get current row
        row = self.df.iloc[self.current_step]

        # Feature values
        features = row[self.feature_columns].values.astype(np.float32)

        # Regime one-hot encoding
        if self.regime_column:
            regime = int(row[self.regime_column])
            regime_onehot = np.zeros(self.n_regimes, dtype=np.float32)
            if 0 <= regime < self.n_regimes:
                regime_onehot[regime] = 1.0
        else:
            regime_onehot = np.array([], dtype=np.float32)

        # Portfolio state
        current_price = row[self.price_column]
        if self.entry_price > 0 and self.position != 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.position
        else:
            unrealized_pnl = 0.0

        portfolio_state = np.array([
            self.position,
            unrealized_pnl
        ], dtype=np.float32)

        # Concatenate all components
        obs = np.concatenate([features, regime_onehot, portfolio_state])

        # Handle any remaining NaN/inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about current state.

        Returns:
            Dictionary with portfolio metrics and state info
        """
        current_row = self.df.iloc[self.current_step]

        info = {
            'step': self.current_step,
            'date': str(self.df.index[self.current_step]),
            'price': current_row[self.price_column],
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'shares_held': self.shares_held,
            'num_trades': len(self.trades),
        }

        if self.regime_column:
            info['regime'] = current_row[self.regime_column]

        # Calculate current metrics if we have history
        if len(self.returns_history) > 1:
            returns = np.array(self.returns_history)
            info['total_return'] = (self.portfolio_value - self.initial_balance) / self.initial_balance
            info['volatility'] = returns.std() * np.sqrt(252)
            if info['volatility'] > 0:
                info['sharpe'] = (returns.mean() * 252) / info['volatility']
            else:
                info['sharpe'] = 0.0

        return info

    def render(self):
        """Render current state (for debugging)."""
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"Step {info['step']}: Price={info['price']:.2f}, "
                  f"Position={info['position']:.2f}, "
                  f"Portfolio=${info['portfolio_value']:.2f}")

    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get portfolio value history as DataFrame.

        Returns:
            DataFrame with dates and portfolio values
        """
        # Get dates for each step
        start_idx = self.window_size
        end_idx = start_idx + len(self.portfolio_history)
        dates = self.df.index[start_idx:end_idx]

        return pd.DataFrame({
            'date': dates[:len(self.portfolio_history)],
            'portfolio_value': self.portfolio_history
        }).set_index('date')

    def get_trades_df(self) -> pd.DataFrame:
        """
        Get trades as DataFrame.

        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()

        trades_df = pd.DataFrame(self.trades)
        trades_df['date'] = [self.df.index[t['step']] for t in self.trades]
        return trades_df


def create_trading_env(
    df: pd.DataFrame,
    feature_columns: List[str],
    regime_column: Optional[str] = None,
    **kwargs
) -> TradingEnv:
    """
    Factory function to create trading environment.

    Args:
        df: DataFrame with features and prices
        feature_columns: List of feature column names
        regime_column: Optional regime column name
        **kwargs: Additional arguments passed to TradingEnv

    Returns:
        Configured TradingEnv instance
    """
    return TradingEnv(
        df=df,
        feature_columns=feature_columns,
        regime_column=regime_column,
        **kwargs
    )

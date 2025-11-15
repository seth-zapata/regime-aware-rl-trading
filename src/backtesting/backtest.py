"""
Core backtesting engine.

CRITICAL: Must avoid look-ahead bias!
Signals at time t can only use information available up to time t.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from ..strategies.baseline import BaseStrategy


class Backtest:
    """
    Backtesting engine for trading strategies.

    Simulates trading based on historical data and signals.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.0,
        slippage: float = 0.0
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital ($)
            commission: Commission per trade ($ or fraction)
            slippage: Slippage per trade (fraction of price)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        # Results storage
        self.portfolio_values = None
        self.positions = None
        self.trades = None
        self.returns = None

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        execution_price: str = 'Close'
    ) -> pd.DataFrame:
        """
        Run backtest simulation.

        Args:
            data: DataFrame with OHLCV data
            signals: Series with trading signals (1: long, 0: flat, -1: short)
            execution_price: Which price to use for execution
                           ('Close', 'Open', 'High', 'Low')

        Returns:
            DataFrame with backtest results including:
            - portfolio_value: Total portfolio value over time
            - position: Current position (shares held)
            - cash: Cash available
            - returns: Period returns

        CRITICAL: Ensure no look-ahead bias!
        - Signal at t uses data up to t
        - Execution at t uses price at t (or t+1 open)
        """
        # TODO: Implement backtest simulation
        # 1. Initialize portfolio state
        #    - cash = initial_capital
        #    - position = 0 (no shares)
        #    - portfolio_value = cash
        # 2. For each time step t:
        #    - Check signal[t]
        #    - If signal changed from previous step, execute trade:
        #      - Calculate shares to buy/sell
        #      - Update cash (subtract cost + commission + slippage)
        #      - Update position
        #    - Calculate portfolio_value = cash + (position * current_price)
        #    - Record state
        # 3. Return DataFrame with full history

        # Validate inputs
        if len(data) != len(signals):
            raise ValueError("Data and signals must have same length")

        if execution_price not in data.columns:
            raise ValueError(f"Execution price column '{execution_price}' not in data")

        # Initialize tracking
        n = len(data)
        cash = np.zeros(n)
        position = np.zeros(n)
        portfolio_value = np.zeros(n)

        cash[0] = self.initial_capital
        position[0] = 0
        portfolio_value[0] = self.initial_capital

        # Track previous signal state (start at 0 = no position)
        prev_signal_state = 0

        # Simulation loop - process each time step
        for i in range(1, n):
            curr_signal = signals.iloc[i]
            price = data[execution_price].iloc[i]

            # Copy previous state
            cash[i] = cash[i-1]
            position[i] = position[i-1]

            # Check for signal change (trade execution)
            if curr_signal != prev_signal_state:
                # Execute trade based on new signal
                if curr_signal == 1:  # Buy signal
                    # Buy as many shares as possible with available cash
                    effective_price = price * (1 + self.slippage)
                    shares_to_buy = (cash[i] - self.commission) / effective_price

                    if shares_to_buy > 0:
                        cost = shares_to_buy * effective_price + self.commission
                        cash[i] -= cost
                        position[i] += shares_to_buy

                elif curr_signal == 0 or curr_signal == -1:  # Sell/flat signal
                    # Sell all shares
                    if position[i] > 0:
                        effective_price = price * (1 - self.slippage)
                        proceeds = position[i] * effective_price - self.commission
                        cash[i] += proceeds
                        position[i] = 0

                # Update prev_signal_state after executing trade
                prev_signal_state = curr_signal

            # Update portfolio value (cash + current position value)
            portfolio_value[i] = cash[i] + position[i] * price

        # Create results DataFrame
        results = pd.DataFrame({
            'portfolio_value': portfolio_value,
            'cash': cash,
            'position': position,
            'price': data[execution_price].values
        }, index=data.index)

        # Calculate returns
        results['returns'] = results['portfolio_value'].pct_change()

        # Store results
        self.portfolio_values = results['portfolio_value']
        self.positions = results['position']
        self.returns = results['returns']

        # TODO: Track individual trades for analysis
        self._extract_trades(results, signals)

        return results

    def _extract_trades(self, results: pd.DataFrame, signals: pd.Series):
        """
        Extract individual trades from backtest results.

        Args:
            results: Backtest results DataFrame
            signals: Signal series
        """
        trades = []
        in_position = False
        entry_idx = None

        for i in range(len(signals)):
            curr_signal = signals.iloc[i]

            if curr_signal == 1 and not in_position:
                # Entering position
                in_position = True
                entry_idx = i

            elif (curr_signal == 0 or curr_signal == -1) and in_position:
                # Exiting position
                in_position = False

                if entry_idx is not None:
                    entry_date = signals.index[entry_idx]
                    exit_date = signals.index[i]
                    entry_price = results['price'].iloc[entry_idx]
                    exit_price = results['price'].iloc[i]
                    shares = results['position'].iloc[entry_idx]

                    pnl = shares * (exit_price - entry_price)
                    pct_return = (exit_price - entry_price) / entry_price

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'return_pct': pct_return,
                        'duration_days': (exit_date - entry_date).days
                    })

        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()

    def get_trade_summary(self) -> pd.DataFrame:
        """
        Get summary of all trades.

        Returns:
            DataFrame with trade details
        """
        # TODO: Return trades DataFrame if available
        if self.trades is None:
            return pd.DataFrame()
        return self.trades


def run_strategy_backtest(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    initial_capital: float = 100000.0,
    commission: float = 0.0,
    slippage: float = 0.0
) -> Dict:
    """
    Convenience function to backtest a strategy.

    Args:
        strategy: Strategy instance
        data: Historical OHLCV data
        initial_capital: Starting capital
        commission: Commission per trade
        slippage: Slippage per trade

    Returns:
        Dictionary with:
        - strategy_name: Name of the strategy
        - signals: Generated trading signals
        - results: Backtest results DataFrame
        - trades: Individual trades DataFrame
    """
    # Generate signals from strategy
    signals = strategy.generate_signals(data)

    # Run backtest
    backtest = Backtest(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage
    )
    results = backtest.run(data, signals)

    # Get trades
    trades = backtest.get_trade_summary()

    return {
        'strategy_name': strategy.name,
        'signals': signals,
        'results': results,
        'trades': trades
    }


# TODO: Additional features
# - Support for different position sizing methods
# - Support for leverage
# - Support for shorting with margin requirements
# - Multi-asset backtesting
# - Event-driven backtesting (more realistic timing)

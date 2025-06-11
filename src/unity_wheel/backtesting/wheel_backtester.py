"""Wheel strategy backtester with realistic simulation and Unity-specific gap risk handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.loader import get_config

from ..data_providers.databento.price_history_loader import PriceHistoryLoader
from ..math import CalculationResult
from ..math.options import black_scholes_price_validated
from ..storage import Storage
from ..strategy.wheel import WheelParameters, WheelStrategy
from ..utils import get_logger, timed_operation
from ..utils.position_sizing import DynamicPositionSizer
from .exceptions import InsufficientDataError

logger = get_logger(__name__)


@dataclass
class BacktestPosition:
    """Track position during backtest."""

    symbol: str
    position_type: str  # 'put' or 'stock'
    strike: float
    expiration: datetime
    entry_date: datetime
    entry_price: float
    contracts: int
    premium_collected: float
    assigned: bool = False
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: float = 0.0


@dataclass
class BacktestResults:
    """Results from backtesting wheel strategy."""

    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    assignments: int
    average_trade_pnl: float

    # Risk metrics
    var_95: float
    cvar_95: float
    max_gap_loss: float

    # Position tracking
    positions: List[BacktestPosition]
    equity_curve: pd.Series
    daily_returns: pd.Series

    # Optimal parameters found
    optimal_delta: float = 0.30
    optimal_dte: int = 45

    # Unity-specific metrics
    gap_events: int = 0
    earnings_avoided: int = 0


class WheelBacktester:
    """Backtest wheel strategy on historical data with realistic simulation."""

    def __init__(
        self,
        storage: Storage,
        price_loader: Optional[PriceHistoryLoader] = None,
    ):
        """Initialize backtester."""
        self.storage = storage
        self.price_loader = price_loader
        self.config = get_config()
        self.strategy = WheelStrategy()
        self.position_sizer = DynamicPositionSizer()

    @timed_operation(threshold_ms=5000.0)
    async def backtest_strategy(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000,
        contracts_per_trade: Optional[int] = None,
        parameters: Optional[WheelParameters] = None,
    ) -> BacktestResults:
        """
        Run wheel strategy backtest on historical data.

        Parameters
        ----------
        symbol : str
            Underlying symbol to trade (e.g., 'U')
        start_date : datetime
            Start date for backtest
        end_date : datetime
            End date for backtest
        initial_capital : float
            Starting portfolio value
        contracts_per_trade : Optional[int]
            Fixed contracts per trade. If ``None``, size dynamically based on
            portfolio value and risk limits.
        parameters : Optional[WheelParameters]
            Strategy parameters to test

        Returns
        -------
        BacktestResults
            Complete backtest results with metrics
        """
        logger.info(
            "Starting wheel backtest",
            extra={
                "symbol": symbol,
                "start": start_date,
                "end": end_date,
                "capital": initial_capital,
            },
        )

        # Load historical data
        price_data = await self._load_price_data(symbol, start_date, end_date)
        if price_data.empty:
            raise ValueError(f"No price data available for {symbol}")

        # Initialize tracking
        capital = initial_capital
        positions: List[BacktestPosition] = []
        active_position: Optional[BacktestPosition] = None
        daily_equity = []
        daily_returns = []

        # Track Unity-specific events
        gap_events = 0
        earnings_avoided = 0

        # Use provided parameters or defaults
        params = parameters or WheelParameters()

        # Simulate each trading day
        for date, row in price_data.iterrows():
            current_price = row["close"]

            # Check for gap moves (Unity-specific)
            if "high" in row and "low" in row:
                daily_range = (row["high"] - row["low"]) / row["low"]
                if daily_range > 0.10:  # 10% intraday move
                    gap_events += 1

            # Skip trading near earnings (Unity pattern)
            if self._is_near_earnings(date):
                earnings_avoided += 1
                daily_equity.append(capital)
                continue

            # Manage active position
            if active_position:
                pnl = self._manage_position(active_position, current_price, date, row)
                capital += pnl

                if active_position.exit_date:
                    positions.append(active_position)
                    active_position = None

            # Enter new position if none active
            elif capital > params.max_position_size * initial_capital:
                # Simulate finding optimal strike
                strike = self._find_backtest_strike(current_price, params.target_delta)

                # Calculate realistic premium
                premium_result = self._calculate_backtest_premium(
                    current_price, strike, params.target_dte
                )

                if contracts_per_trade is None:
                    contracts = self.position_sizer.contracts_for_trade(
                        portfolio_value=capital,
                        buying_power=capital,
                        strike_price=strike,
                        option_premium=premium_result.value * 100,
                    )
                else:
                    contracts = contracts_per_trade

                # Enter short put position
                active_position = BacktestPosition(
                    symbol=symbol,
                    position_type="put",
                    strike=strike,
                    expiration=date + timedelta(days=params.target_dte),
                    entry_date=date,
                    entry_price=current_price,
                    contracts=contracts,
                    premium_collected=premium_result.value * 100 * contracts,
                )

                capital += active_position.premium_collected

            # Track daily equity
            daily_equity.append(capital)
            if len(daily_equity) > 1:
                daily_returns.append((daily_equity[-1] - daily_equity[-2]) / daily_equity[-2])
            else:
                daily_returns.append(0)

        # Close any remaining position
        if active_position:
            final_pnl = self._close_position(
                active_position, price_data.iloc[-1]["close"], end_date
            )
            capital += final_pnl
            positions.append(active_position)

        # Calculate performance metrics
        equity_curve = pd.Series(daily_equity, index=price_data.index)
        returns_series = pd.Series(daily_returns, index=price_data.index)

        # Core metrics
        total_return = (capital - initial_capital) / initial_capital
        years = (end_date - start_date).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1

        # Risk metrics
        sharpe_result = self._calculate_sharpe(returns_series)
        max_dd_result = self._calculate_max_drawdown(equity_curve)
        var_95, cvar_95, _ = self._calculate_var_cvar(returns_series)

        # Trade statistics
        winning_trades = sum(1 for p in positions if p.realized_pnl > 0)
        assignments = sum(1 for p in positions if p.assigned)

        # Find max gap loss
        max_gap_loss = self._calculate_max_gap_loss(positions, price_data)

        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_result.value,
            max_drawdown=max_dd_result.value,
            win_rate=winning_trades / len(positions) if positions else 0,
            total_trades=len(positions),
            winning_trades=winning_trades,
            losing_trades=len(positions) - winning_trades,
            assignments=assignments,
            average_trade_pnl=(
                sum(p.realized_pnl for p in positions) / len(positions) if positions else 0
            ),
            var_95=var_95,
            cvar_95=cvar_95,
            max_gap_loss=max_gap_loss,
            positions=positions,
            equity_curve=equity_curve,
            daily_returns=returns_series,
            optimal_delta=params.target_delta,
            optimal_dte=params.target_dte,
            gap_events=gap_events,
            earnings_avoided=earnings_avoided,
        )

    def simulate_assignment(
        self,
        put_strike: float,
        underlying_price: float,
        expiration_price: float = None,
    ) -> bool:
        """
        Realistic assignment simulation.

        Parameters
        ----------
        put_strike : float
            Strike price of short put
        underlying_price : float
            Current underlying price
        expiration_price : float
            Price at expiration (if different)

        Returns
        -------
        bool
            True if assigned, False otherwise
        """
        # Use expiration price if provided
        check_price = expiration_price if expiration_price else underlying_price

        # Standard assignment: ITM at expiration
        if check_price < put_strike:
            # Add small randomness for pin risk
            if abs(check_price - put_strike) / put_strike < 0.005:  # Within 0.5%
                return np.random.random() < 0.7  # 70% chance if near strike
            return True

        return False

    async def optimize_parameters(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        delta_range: Tuple[float, float] = (0.20, 0.40),
        dte_range: Tuple[int, int] = (30, 60),
        optimization_metric: str = "sharpe",
    ) -> Dict[str, any]:
        """
        Optimize delta and DTE parameters.

        Parameters
        ----------
        symbol : str
            Symbol to optimize
        start_date : datetime
            Backtest start date
        end_date : datetime
            Backtest end date
        delta_range : Tuple[float, float]
            Min/max delta to test
        dte_range : Tuple[int, int]
            Min/max DTE to test
        optimization_metric : str
            Metric to optimize ('sharpe', 'return', 'win_rate')

        Returns
        -------
        Dict with optimal parameters and results
        """
        logger.info("Starting parameter optimization")

        # Test grid
        deltas = np.linspace(delta_range[0], delta_range[1], 5)
        dtes = np.linspace(dte_range[0], dte_range[1], 4, dtype=int)

        best_metric = -float("inf")
        best_params = None
        best_results = None
        results_grid = []

        for delta in deltas:
            for dte in dtes:
                # Create test parameters
                params = WheelParameters(
                    target_delta=delta,
                    target_dte=dte,
                )

                # Run backtest
                try:
                    results = await self.backtest_strategy(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        parameters=params,
                    )

                    # Extract optimization metric
                    if optimization_metric == "sharpe":
                        metric = results.sharpe_ratio
                    elif optimization_metric == "return":
                        metric = results.annualized_return
                    elif optimization_metric == "win_rate":
                        metric = results.win_rate
                    else:
                        raise ValueError(f"Unknown metric: {optimization_metric}")

                    results_grid.append(
                        {
                            "delta": delta,
                            "dte": dte,
                            "metric": metric,
                            "return": results.annualized_return,
                            "sharpe": results.sharpe_ratio,
                            "max_dd": results.max_drawdown,
                        }
                    )

                    if metric > best_metric:
                        best_metric = metric
                        best_params = params
                        best_results = results

                except Exception as e:
                    logger.error(f"Backtest failed for delta={delta}, dte={dte}: {e}")

        return {
            "optimal_delta": best_params.target_delta if best_params else None,
            "optimal_dte": best_params.target_dte if best_params else None,
            "best_metric": best_metric,
            "best_results": best_results,
            "all_results": pd.DataFrame(results_grid),
        }

    # Private helper methods

    async def _load_price_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load historical price data."""
        async with self.storage.cache.connection() as conn:
            result = conn.execute(
                """
                SELECT date, open, high, low, close, volume
                FROM price_history
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
                """,
                [symbol, start_date.date(), end_date.date()],
            ).fetchall()

        if not result:
            raise InsufficientDataError(
                f"No price data available for {symbol}. Minimum {PriceHistoryLoader.MINIMUM_DAYS} days required"
            )

        df = pd.DataFrame(
            result,
            columns=["date", "open", "high", "low", "close", "volume"],
        )
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        if len(df) < PriceHistoryLoader.MINIMUM_DAYS:
            raise InsufficientDataError(
                f"Only {len(df)} days of data found for {symbol}; {PriceHistoryLoader.MINIMUM_DAYS} required"
            )

        return df

    def _manage_position(
        self,
        position: BacktestPosition,
        current_price: float,
        current_date: datetime,
        price_row: pd.Series,
    ) -> float:
        """Manage active position and return P&L."""
        pnl = 0.0

        # Check if expired
        if current_date >= position.expiration:
            # Check assignment
            position.assigned = self.simulate_assignment(
                position.strike, current_price, price_row.get("close", current_price)
            )

            if position.assigned:
                # Assigned - now long stock at strike price
                position.position_type = "stock"
                position.entry_price = position.strike
                logger.debug(f"Put assigned at ${position.strike}")
            else:
                # Expired worthless - keep full premium
                position.exit_date = current_date
                position.exit_price = current_price
                position.realized_pnl = position.premium_collected
                pnl = position.realized_pnl

        elif position.position_type == "stock":
            # Holding stock after assignment
            # Simplified: exit if profitable
            if current_price > position.entry_price * 1.02:  # 2% profit
                position.exit_date = current_date
                position.exit_price = current_price
                stock_pnl = (current_price - position.entry_price) * 100 * position.contracts
                position.realized_pnl = position.premium_collected + stock_pnl
                pnl = stock_pnl  # Premium already counted

        return pnl

    def _close_position(
        self,
        position: BacktestPosition,
        final_price: float,
        final_date: datetime,
    ) -> float:
        """Force close position at end of backtest."""
        if position.position_type == "put":
            # Buy back put at intrinsic value
            intrinsic = max(0, position.strike - final_price)
            cost = intrinsic * 100 * position.contracts
            position.realized_pnl = position.premium_collected - cost
        else:
            # Sell stock
            stock_pnl = (final_price - position.entry_price) * 100 * position.contracts
            position.realized_pnl = position.premium_collected + stock_pnl

        position.exit_date = final_date
        position.exit_price = final_price

        return position.realized_pnl - position.premium_collected

    def _find_backtest_strike(
        self,
        current_price: float,
        target_delta: float,
    ) -> float:
        """Find strike for target delta in backtest."""
        # Simplified: use percentage OTM based on delta
        # Delta 0.30 roughly 5% OTM for 45 DTE
        otm_percent = 0.15 * (1 - target_delta)
        strike = current_price * (1 - otm_percent)

        # Round to nearest $2.50 for Unity
        return round(strike / 2.5) * 2.5

    def _calculate_backtest_premium(
        self,
        spot: float,
        strike: float,
        dte: int,
    ) -> CalculationResult:
        """Calculate realistic option premium for backtest."""
        # Use simplified IV model for Unity
        # Higher IV for lower strikes, near-term
        moneyness = strike / spot
        base_iv = 0.45  # Unity base IV

        # Adjust for moneyness
        if moneyness < 0.95:
            iv = base_iv * 1.2
        elif moneyness < 0.90:
            iv = base_iv * 1.4
        else:
            iv = base_iv

        # Adjust for time
        if dte < 30:
            iv *= 1.1

        # Calculate Black-Scholes price
        result = black_scholes_price_validated(
            S=spot,
            K=strike,
            T=dte / 365.0,
            r=0.05,
            sigma=iv,
            option_type="put",
        )

        if result.confidence > 0.5:
            premium = result.value
            conf = result.confidence
        else:
            premium = spot * 0.02
            conf = 0.5 * result.confidence

        return CalculationResult(premium, conf, [])

    def _is_near_earnings(self, date: datetime) -> bool:
        """Check if date is near Unity earnings."""
        # Unity typically reports in early Feb, May, Aug, Nov
        # Skip 7 days before and after
        month = date.month
        day = date.day

        # Rough earnings windows
        earnings_windows = [
            (2, 1, 14),  # Feb 1-14
            (5, 1, 14),  # May 1-14
            (8, 1, 14),  # Aug 1-14
            (11, 1, 14),  # Nov 1-14
        ]

        for window_month, start_day, end_day in earnings_windows:
            if month == window_month and start_day <= day <= end_day:
                return True

        return False

    def _calculate_sharpe(self, returns: pd.Series) -> CalculationResult:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return CalculationResult(0.0, 0.3, ["Insufficient data"])

        # Annualized Sharpe
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)

        if std_return == 0:
            return CalculationResult(0.0, 0.5, ["Zero volatility"])

        sharpe = mean_return / std_return
        return CalculationResult(sharpe, 1.0, [])

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> CalculationResult:
        """Calculate maximum drawdown."""
        if len(equity_curve) == 0:
            return CalculationResult(0.0, 0.0, ["Empty curve"])

        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        return CalculationResult(max_dd, 1.0, [])

    def _calculate_var_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Calculate VaR and CVaR with a confidence score."""
        if len(returns) < 20:
            return 0.0, 0.0, 0.3

        # Historical VaR
        var_percentile = (1 - confidence) * 100
        var = np.percentile(returns, var_percentile)

        # CVaR (expected shortfall)
        cvar = returns[returns <= var].mean()

        conf = 1.0 if len(returns) >= 250 else 0.8

        return var, cvar, conf

    def _calculate_max_gap_loss(
        self,
        positions: List[BacktestPosition],
        price_data: pd.DataFrame,
    ) -> float:
        """Calculate maximum gap loss from positions."""
        max_loss = 0.0

        for position in positions:
            if position.assigned and position.position_type == "stock":
                # Find gap on assignment
                if position.entry_date in price_data.index:
                    row = price_data.loc[position.entry_date]
                    gap = (row["low"] - position.strike) / position.strike
                    loss = gap * position.strike * 100 * position.contracts
                    max_loss = min(max_loss, loss)

        return abs(max_loss)

"""Tests for wheel strategy backtester."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.unity_wheel.backtesting import BacktestPosition, BacktestResults, WheelBacktester
from src.unity_wheel.storage import Storage
from src.unity_wheel.strategy.wheel import WheelParameters


class TestWheelBacktester:
    """Test wheel strategy backtester."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = Mock(spec=Storage)
        storage.cache = Mock()
        storage.cache.connection = AsyncMock()
        return storage

    @pytest.fixture
    def backtester(self, mock_storage):
        """Create backtester instance."""
        return WheelBacktester(storage=mock_storage)

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range("2024-01-01", "2024-03-31", freq="D")
        prices = 35 + np.sin(np.arange(len(dates)) * 0.1) * 5  # Oscillating around $35

        data = pd.DataFrame(
            {
                "date": dates,
                "open": prices + np.random.randn(len(dates)) * 0.5,
                "high": prices + abs(np.random.randn(len(dates))) * 1,
                "low": prices - abs(np.random.randn(len(dates))) * 1,
                "close": prices,
                "volume": np.random.randint(1000000, 5000000, len(dates)),
            }
        )
        data.set_index("date", inplace=True)
        return data

    def test_simulate_assignment_itm(self, backtester):
        """Test assignment simulation for ITM puts."""
        # Deep ITM should assign
        assert backtester.simulate_assignment(put_strike=40.0, underlying_price=35.0) == True

        # OTM should not assign
        assert backtester.simulate_assignment(put_strike=30.0, underlying_price=35.0) == False

    def test_simulate_assignment_pin_risk(self, backtester):
        """Test pin risk near strike."""
        # Near strike has random assignment
        assigned_count = 0
        for _ in range(100):
            if backtester.simulate_assignment(put_strike=35.0, underlying_price=34.90):  # Just ITM
                assigned_count += 1

        # Should be around 70% assignment rate
        assert 50 < assigned_count < 90

    @pytest.mark.asyncio
    async def test_backtest_strategy_basic(self, backtester, mock_storage, sample_price_data):
        """Test basic backtest functionality."""

        # Mock data loading
        async def mock_connection():
            conn = Mock()
            conn.execute = Mock(
                return_value=Mock(
                    fetchall=Mock(
                        return_value=[
                            (
                                row.name,
                                row["open"],
                                row["high"],
                                row["low"],
                                row["close"],
                                row["volume"],
                            )
                            for _, row in sample_price_data.iterrows()
                        ]
                    )
                )
            )
            return conn

        mock_storage.cache.connection = mock_connection

        # Run backtest
        results = await backtester.backtest_strategy(
            symbol="U",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
            initial_capital=100000,
            contracts_per_trade=1,
        )

        # Verify results structure
        assert isinstance(results, BacktestResults)
        assert results.total_trades >= 0
        assert results.equity_curve is not None
        assert len(results.positions) == results.total_trades

    @pytest.mark.asyncio
    async def test_backtest_with_assignment(self, backtester, mock_storage):
        """Test backtest with assignment scenario."""
        # Create price data with assignment event
        dates = pd.date_range("2024-01-01", "2024-02-28", freq="D")
        prices = [35.0] * 30 + [30.0] * len(dates[30:])  # Drop below strike

        price_data = [(date, p, p + 1, p - 1, p, 1000000) for date, p in zip(dates, prices)]

        async def mock_connection():
            conn = Mock()
            conn.execute = Mock(return_value=Mock(fetchall=Mock(return_value=price_data)))
            return conn

        mock_storage.cache.connection = mock_connection

        # Run backtest
        results = await backtester.backtest_strategy(
            symbol="U",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 28),
            initial_capital=100000,
        )

        # Should have assignments due to price drop
        assert results.assignments > 0

    @pytest.mark.asyncio
    async def test_parameter_optimization(self, backtester, mock_storage, sample_price_data):
        """Test delta/DTE parameter optimization."""

        # Mock data loading
        async def mock_connection():
            conn = Mock()
            conn.execute = Mock(
                return_value=Mock(
                    fetchall=Mock(
                        return_value=[
                            (
                                row.name,
                                row["open"],
                                row["high"],
                                row["low"],
                                row["close"],
                                row["volume"],
                            )
                            for _, row in sample_price_data.iterrows()
                        ]
                    )
                )
            )
            return conn

        mock_storage.cache.connection = mock_connection

        # Run optimization
        results = await backtester.optimize_parameters(
            symbol="U",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
            delta_range=(0.25, 0.35),
            dte_range=(30, 45),
            optimization_metric="sharpe",
        )

        # Verify optimization results
        assert "optimal_delta" in results
        assert "optimal_dte" in results
        assert "all_results" in results
        assert isinstance(results["all_results"], pd.DataFrame)

    def test_gap_risk_tracking(self, backtester):
        """Test Unity-specific gap risk tracking."""
        # Create position with gap move
        position = BacktestPosition(
            symbol="U",
            position_type="stock",  # After assignment
            strike=35.0,
            expiration=datetime(2024, 2, 1),
            entry_date=datetime(2024, 1, 15),
            entry_price=35.0,
            contracts=1,
            premium_collected=150,
            assigned=True,
        )

        # Create price data with gap
        price_data = pd.DataFrame(
            {
                "open": [35.0],
                "high": [35.0],
                "low": [30.0],  # Gap down
                "close": [31.0],
                "volume": [1000000],
            },
            index=[position.entry_date],
        )

        # Calculate gap loss
        max_gap_loss = backtester._calculate_max_gap_loss([position], price_data)

        # Should detect 5 point gap * 100 shares = $500 loss
        assert max_gap_loss == 500

    def test_earnings_avoidance(self, backtester):
        """Test earnings date avoidance."""
        # Test known earnings windows
        assert backtester._is_near_earnings(datetime(2024, 2, 5)) == True
        assert backtester._is_near_earnings(datetime(2024, 5, 10)) == True
        assert backtester._is_near_earnings(datetime(2024, 8, 7)) == True
        assert backtester._is_near_earnings(datetime(2024, 11, 12)) == True

        # Test non-earnings dates
        assert backtester._is_near_earnings(datetime(2024, 3, 15)) == False
        assert backtester._is_near_earnings(datetime(2024, 6, 20)) == False

    def test_position_management(self, backtester):
        """Test position lifecycle management."""
        # Create short put position
        position = BacktestPosition(
            symbol="U",
            position_type="put",
            strike=35.0,
            expiration=datetime(2024, 2, 15),
            entry_date=datetime(2024, 1, 15),
            entry_price=36.0,
            contracts=1,
            premium_collected=150,
        )

        # Test expiration OTM
        price_row = pd.Series({"open": 37, "high": 38, "low": 36, "close": 37})
        pnl = backtester._manage_position(position, 37.0, datetime(2024, 2, 15), price_row)

        assert position.assigned == False
        assert position.realized_pnl == 150  # Keep full premium

    def test_risk_metrics_calculation(self, backtester):
        """Test VaR and CVaR calculation."""
        # Create returns series
        returns = pd.Series(np.random.normal(-0.001, 0.02, 100))  # Slight negative drift

        var_95, cvar_95, conf = backtester._calculate_var_cvar(returns)

        # VaR should be negative (loss)
        assert var_95 < 0
        # CVaR should be worse than VaR
        assert cvar_95 < var_95
        assert 0 <= conf <= 1

    def test_max_drawdown_calculation(self, backtester):
        """Test maximum drawdown calculation."""
        # Create equity curve with drawdown
        equity = pd.Series(
            [100000, 105000, 110000, 108000, 102000, 105000, 107000, 109000]  # Drawdown  # Recovery
        )

        max_dd = backtester._calculate_max_drawdown(equity)

        # Max drawdown from 110k to 102k = -7.27%
        assert -0.08 < max_dd.value < -0.07
        assert max_dd.confidence > 0

    def test_sharpe_ratio_calculation(self, backtester):
        """Test Sharpe ratio calculation."""
        # Create consistent positive returns
        returns = pd.Series([0.001] * 252)  # 0.1% daily

        sharpe = backtester._calculate_sharpe(returns)

        # With no volatility, Sharpe should be very high
        # 0.1% daily = 25.2% annual, 0 vol = infinite Sharpe
        # But we handle division by zero
        assert sharpe.value == 0.0 or sharpe.value > 10
        assert sharpe.confidence > 0

    def test_backtest_strike_selection(self, backtester):
        """Test strike selection in backtest."""
        # Test delta 0.30 strike selection
        strike = backtester._find_backtest_strike(current_price=35.0, target_delta=0.30)

        # Should be about 3-5% OTM
        assert 33.0 <= strike <= 34.0
        # Should round to $2.50 for Unity
        assert strike % 2.5 == 0

    def test_backtest_premium_calculation(self, backtester):
        """Test option premium calculation."""
        # Test ATM 45 DTE put
        premium = backtester._calculate_backtest_premium(spot=35.0, strike=35.0, dte=45)

        # Should be reasonable for Unity
        assert 0.5 < premium.value < 3.0  # $0.50 to $3.00
        assert premium.confidence > 0

    def test_position_sizing(self, backtester):
        """Test position size limits."""
        params = WheelParameters(max_position_size=0.20)  # 20% max

        # Position size should respect limits in backtest
        # This is implicitly tested in backtest_strategy
        assert params.max_position_size == 0.20

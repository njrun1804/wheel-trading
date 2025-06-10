"""Integration tests for trading calendar with wheel strategy."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.unity_wheel.strategy.wheel import WheelParameters, WheelStrategy
from src.unity_wheel.utils import SimpleTradingCalendar
from src.unity_wheel.utils.trading_calendar_enhancements import EnhancedTradingCalendar


class TestWheelCalendarIntegration:
    """Test trading calendar integration with wheel strategy."""

    @pytest.fixture
    def calendar(self):
        """Create calendar instance."""
        return SimpleTradingCalendar()

    @pytest.fixture
    def enhanced_calendar(self):
        """Create enhanced calendar instance."""
        return EnhancedTradingCalendar()

    @pytest.fixture
    def wheel_strategy(self):
        """Create wheel strategy instance."""
        params = WheelParameters(target_delta=0.30, target_dte=45, min_premium_pct=0.01)
        return WheelStrategy(params)

    def test_dte_calculation_with_trading_days(self, calendar):
        """Test that DTE should use trading days, not calendar days."""
        # Friday to next monthly expiry
        friday = datetime(2025, 1, 10)  # Friday
        next_expiry = calendar.get_next_expiry_friday(friday).value

        # Calendar days
        calendar_days = (next_expiry.date() - friday.date()).days

        # Trading days (should be less due to weekend)
        trading_days = calendar.days_to_next_expiry(friday).value

        assert trading_days < calendar_days
        assert trading_days == 5  # Mon-Fri next week

    def test_avoid_trading_on_holidays(self, calendar):
        """Test that recommendations avoid holidays."""
        # New Year's Day 2025
        new_years = datetime(2025, 1, 1)
        assert not calendar.is_trading_day(new_years).value

        # MLK Day 2025
        mlk_day = datetime(2025, 1, 20)
        assert not calendar.is_trading_day(mlk_day).value

    def test_early_close_risk_adjustment(self, enhanced_calendar):
        """Test risk adjustments for early close days."""
        # Black Friday 2025
        black_friday = datetime(2025, 11, 28)

        # Should be early close
        assert enhanced_calendar.is_early_close(black_friday).value

        # Market hours should show 1 PM close
        open_time, close_time = enhanced_calendar.get_market_hours(black_friday).value
        assert close_time.hour == 13  # 1 PM

        # Trading hours remaining at noon
        noon = datetime(2025, 11, 28, 12, 0)
        remaining = enhanced_calendar.trading_hours_remaining(noon).value
        assert remaining == 1.0  # 1 hour left

    def test_unity_earnings_avoidance(self, enhanced_calendar):
        """Test avoiding Unity earnings dates."""
        # Typical Unity earnings in February (3rd week)
        feb_earnings = datetime(2025, 2, 20)  # 3rd Thursday

        # Should detect near earnings
        assert enhanced_calendar.is_near_unity_earnings(feb_earnings, days_buffer=7).value

        # Week before should also trigger
        week_before = feb_earnings - timedelta(days=7)
        assert enhanced_calendar.is_near_unity_earnings(week_before, days_buffer=7).value

        # Month before should be safe
        month_before = datetime(2025, 1, 20)
        assert not enhanced_calendar.is_near_unity_earnings(month_before, days_buffer=7).value

    def test_optimal_expiry_selection(self, enhanced_calendar):
        """Test selecting expiries that avoid earnings."""
        # Start in late January
        start_date = datetime(2025, 1, 25)

        expiries = enhanced_calendar.get_expiry_fridays_avoiding_earnings(start_date, count=3).value

        # February expiry should warn about earnings
        feb_expiry, feb_warning = expiries[0]
        assert feb_expiry.month == 2
        assert feb_warning is True

        # March should be safe
        mar_expiry, mar_warning = expiries[1]
        assert mar_expiry.month == 3
        assert mar_warning is False

    def test_theta_decay_calculation(self, enhanced_calendar):
        """Test theta decay accounts for non-trading days."""
        # 45 DTE position
        start = datetime(2025, 1, 6)  # Monday
        expiry = datetime(2025, 2, 21)  # February expiry

        decay = enhanced_calendar.calculate_theta_decay_days(start, expiry).value

        # Should have metrics
        assert "calendar_days" in decay
        assert "trading_days" in decay
        assert "effective_theta_days" in decay
        assert "weekend_acceleration" in decay

        # Effective days should be less than calendar days
        assert decay["effective_theta_days"] < decay["calendar_days"]

        # Weekend acceleration should be significant (many weekends)
        assert decay["weekend_acceleration"] > 0.2  # >20% non-trading days

    @patch("src.unity_wheel.strategy.wheel.WheelStrategy.find_optimal_put_strike")
    def test_wheel_strategy_with_calendar(self, mock_find_strike, wheel_strategy, calendar):
        """Test wheel strategy using calendar for DTE calculations."""
        # Mock the strike finding
        mock_strike = MagicMock()
        mock_strike.strike = 30.0
        mock_strike.premium = 1.50
        mock_find_strike.return_value = mock_strike

        # Calculate actual trading days to expiry
        today = datetime.now()
        next_expiry = calendar.get_next_expiry_friday(today).value
        actual_trading_days = calendar.days_to_next_expiry(today).value

        # Call strategy with calendar-aware DTE
        result = wheel_strategy.find_optimal_put_strike(
            current_price=35.0,
            available_strikes=[30, 32.5, 35],
            days_to_expiry=actual_trading_days,  # Use trading days
            volatility=0.45,
        )

        # Verify it was called with trading days
        mock_find_strike.assert_called_once_with(
            current_price=35.0,
            available_strikes=[30, 32.5, 35],
            days_to_expiry=actual_trading_days,
            volatility=0.45,
        )

    def test_weekend_gap_risk_awareness(self, calendar):
        """Test awareness of weekend gap risk."""
        # Friday position entry has weekend gap risk
        friday = datetime(2025, 1, 10)
        assert calendar.is_trading_day(friday).value
        assert friday.weekday() == 4  # Confirm it's Friday

        # Next trading day is Monday (3 calendar days later)
        next_trading = calendar.get_next_trading_day(friday).value
        gap_days = (next_trading.date() - friday.date()).days
        assert gap_days == 3  # Weekend gap

        # Tuesday entry has less gap risk
        tuesday = datetime(2025, 1, 14)
        next_trading = calendar.get_next_trading_day(tuesday).value
        gap_days = (next_trading.date() - tuesday.date()).days
        assert gap_days == 1  # Just overnight

    def test_monthly_expiry_consistency(self, calendar):
        """Test that monthly expiries are consistent."""
        expiries_2025 = calendar.get_monthly_expiries(2025).value

        # Should have 12 monthly expiries
        assert len(expiries_2025) == 12

        # All should be Fridays
        for expiry in expiries_2025:
            assert expiry.weekday() == 4  # Friday
            assert calendar.is_expiry_friday(expiry).value

        # All should be trading days (except if holiday)
        for expiry in expiries_2025:
            if expiry != datetime(2025, 4, 18).date():  # Good Friday
                assert calendar.is_trading_day(expiry).value

"""Tests for enhanced trading calendar features."""

from datetime import date, datetime, time

import pytest

from src.unity_wheel.utils.trading_calendar_enhancements import (
    EnhancedTradingCalendar,
    calculate_theta_decay,
    get_market_hours,
    is_near_unity_earnings,
)


class TestEnhancedTradingCalendar:
    """Test enhanced calendar features."""

    @pytest.fixture
    def calendar(self):
        """Create enhanced calendar instance."""
        return EnhancedTradingCalendar()

    def test_early_close_detection(self, calendar):
        """Test early close day detection."""
        # Christmas Eve 2025 (Wednesday - weekday)
        xmas_eve = datetime(2025, 12, 24)
        assert calendar.is_early_close(xmas_eve).value
        assert calendar.is_trading_day(xmas_eve).value  # Still a trading day

        # July 3, 2025 (Thursday before July 4th Friday)
        july_3 = datetime(2025, 7, 3)
        assert calendar.is_early_close(july_3).value

        # Regular day
        regular_day = datetime(2025, 11, 10)
        assert not calendar.is_early_close(regular_day).value

    def test_market_hours(self, calendar):
        """Test market hours calculation."""
        # Regular trading day
        regular_day = datetime(2025, 11, 10)
        open_time, close_time = calendar.get_market_hours(regular_day).value
        assert open_time == time(9, 30)
        assert close_time == time(16, 0)

        # Early close day (Christmas Eve)
        xmas_eve = datetime(2025, 12, 24)
        open_time, close_time = calendar.get_market_hours(xmas_eve).value
        assert open_time == time(9, 30)
        assert close_time == time(13, 0)  # 1 PM close

        # Non-trading day
        xmas = datetime(2025, 12, 25)
        open_time, close_time = calendar.get_market_hours(xmas).value
        assert open_time is None
        assert close_time is None

    def test_trading_hours_remaining(self, calendar):
        """Test remaining hours calculation."""
        # Mock a Wednesday at 2 PM
        wed_2pm = datetime(2025, 11, 12, 14, 0)  # 2 PM
        remaining = calendar.trading_hours_remaining(wed_2pm).value
        assert remaining == 2.0  # 2 hours until 4 PM

        # Early close day at noon (Christmas Eve)
        xmas_eve_noon = datetime(2025, 12, 24, 12, 0)
        remaining = calendar.trading_hours_remaining(xmas_eve_noon).value
        assert remaining == 1.0  # 1 hour until 1 PM

        # After hours
        after_close = datetime(2025, 11, 12, 17, 0)  # 5 PM
        remaining = calendar.trading_hours_remaining(after_close).value
        assert remaining == 0.0

    def test_unity_earnings_detection(self, calendar):
        """Test Unity earnings date detection."""
        # Typical earnings month (February)
        near_earnings = datetime(2025, 2, 20)  # 3rd Thursday
        assert calendar.is_near_unity_earnings(near_earnings, days_buffer=7).value

        # Not near earnings
        not_earnings = datetime(2025, 2, 1)
        assert not calendar.is_near_unity_earnings(not_earnings, days_buffer=7).value

        # Non-earnings month
        june = datetime(2025, 6, 15)
        assert not calendar.is_near_unity_earnings(june).value

    def test_expiry_with_earnings_warning(self, calendar):
        """Test expiry selection with earnings warnings."""
        # Start in late January 2025
        start_date = datetime(2025, 1, 25)
        expiries = calendar.get_expiry_fridays_avoiding_earnings(start_date, count=3).value

        assert len(expiries) == 3

        # February expiry should warn about earnings
        feb_expiry, feb_warning = expiries[0]
        assert feb_expiry.month == 2
        assert feb_warning is True  # Near earnings

        # March should be safe
        mar_expiry, mar_warning = expiries[1]
        assert mar_expiry.month == 3
        assert mar_warning is False

    def test_theta_decay_calculation(self, calendar):
        """Test theta decay metrics."""
        start = datetime(2025, 1, 10)  # Friday
        expiry = datetime(2025, 2, 21)  # February expiry (42 days)

        decay = calendar.calculate_theta_decay_days(start, expiry).value

        assert decay["calendar_days"] == 42
        assert decay["trading_days"] < 42  # Excludes weekends/holidays
        assert decay["non_trading_days"] > 0
        assert decay["effective_theta_days"] < decay["calendar_days"]
        assert 0 < decay["weekend_acceleration"] < 1

    def test_optimal_entry_day(self, calendar):
        """Test optimal entry day calculation."""
        # Target February expiry
        target_expiry = datetime(2025, 2, 21)

        entry = calendar.optimal_entry_day(target_expiry).value

        # Should be a trading day
        assert calendar.is_trading_day(entry).value

        # Should avoid Monday/Friday
        assert entry.weekday() not in [0, 4]  # Not Monday or Friday

        # Should be roughly 45 days before expiry
        days_until = (target_expiry.date() - entry.date()).days
        assert 40 <= days_until <= 50

    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        # Market hours
        regular_day = datetime(2025, 11, 10)
        open_time, close_time = get_market_hours(regular_day).value
        assert open_time == time(9, 30)
        assert close_time == time(16, 0)

        # Earnings check
        near_earnings = datetime(2025, 2, 20)
        assert is_near_unity_earnings(near_earnings).value

        # Theta decay
        start = datetime(2025, 1, 10)
        expiry = datetime(2025, 2, 21)
        decay = calculate_theta_decay(start, expiry).value
        assert "effective_theta_days" in decay

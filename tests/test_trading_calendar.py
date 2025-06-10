"""Tests for SimpleTradingCalendar."""

from datetime import date, datetime, timedelta

import pytest

from src.unity_wheel.utils.trading_calendar import (
    SimpleTradingCalendar,
    get_next_expiry_friday,
    is_trading_day,
)


class TestSimpleTradingCalendar:
    """Test trading calendar functionality."""

    @pytest.fixture
    def calendar(self):
        """Create calendar instance."""
        return SimpleTradingCalendar()

    def test_weekend_detection(self, calendar):
        """Test that weekends are not trading days."""
        # Saturday
        saturday = datetime(2025, 1, 11)
        assert not calendar.is_trading_day(saturday)

        # Sunday
        sunday = datetime(2025, 1, 12)
        assert not calendar.is_trading_day(sunday)

        # Monday (should be trading day if not holiday)
        monday = datetime(2025, 1, 13)
        assert calendar.is_trading_day(monday)

    def test_known_holidays_2025(self, calendar):
        """Test known 2025 holidays."""
        holidays = [
            datetime(2025, 1, 1),  # New Year's Day
            datetime(2025, 1, 20),  # MLK Day
            datetime(2025, 2, 17),  # Presidents Day
            datetime(2025, 4, 18),  # Good Friday
            datetime(2025, 5, 26),  # Memorial Day
            datetime(2025, 6, 19),  # Juneteenth
            datetime(2025, 7, 4),  # Independence Day
            datetime(2025, 9, 1),  # Labor Day
            datetime(2025, 11, 27),  # Thanksgiving
            datetime(2025, 11, 28),  # Day after Thanksgiving
            datetime(2025, 12, 25),  # Christmas
        ]

        for holiday in holidays:
            assert not calendar.is_trading_day(holiday), f"{holiday} should be a holiday"

    def test_weekend_holiday_observance(self, calendar):
        """Test holidays that fall on weekends are observed on weekdays."""
        # July 4, 2026 is a Saturday - should observe on Friday July 3
        july_3_2026 = datetime(2026, 7, 3)
        july_4_2026 = datetime(2026, 7, 4)

        assert not calendar.is_trading_day(july_3_2026)  # Observed holiday
        assert not calendar.is_trading_day(july_4_2026)  # Actual Saturday

    def test_third_friday_calculation(self, calendar):
        """Test third Friday calculations."""
        # Known third Fridays
        third_fridays = [
            (2025, 1, date(2025, 1, 17)),
            (2025, 2, date(2025, 2, 21)),
            (2025, 3, date(2025, 3, 21)),
            (2025, 4, date(2025, 4, 18)),  # Also Good Friday!
            (2025, 5, date(2025, 5, 16)),
            (2025, 6, date(2025, 6, 20)),
            (2025, 7, date(2025, 7, 18)),
            (2025, 8, date(2025, 8, 15)),
            (2025, 9, date(2025, 9, 19)),
            (2025, 10, date(2025, 10, 17)),
            (2025, 11, date(2025, 11, 21)),
            (2025, 12, date(2025, 12, 19)),
        ]

        for year, month, expected in third_fridays:
            result = calendar._get_third_friday(year, month)
            assert (
                result == expected
            ), f"Third Friday of {month}/{year} should be {expected}, got {result}"

    def test_get_next_expiry_friday(self, calendar):
        """Test getting next expiry Friday."""
        # Test from middle of month
        jan_10_2025 = datetime(2025, 1, 10)
        next_expiry = calendar.get_next_expiry_friday(jan_10_2025)
        assert next_expiry.date() == date(2025, 1, 17)

        # Test from after third Friday (should get next month)
        jan_20_2025 = datetime(2025, 1, 20)
        next_expiry = calendar.get_next_expiry_friday(jan_20_2025)
        assert next_expiry.date() == date(2025, 2, 21)

        # Test year boundary
        dec_20_2024 = datetime(2024, 12, 20)
        next_expiry = calendar.get_next_expiry_friday(dec_20_2024)
        assert next_expiry.date() == date(2025, 1, 17)

    def test_is_expiry_friday(self, calendar):
        """Test expiry Friday detection."""
        # Third Friday
        assert calendar.is_expiry_friday(datetime(2025, 1, 17))

        # Not third Friday (second Friday)
        assert not calendar.is_expiry_friday(datetime(2025, 1, 10))

        # Not Friday
        assert not calendar.is_expiry_friday(datetime(2025, 1, 16))

    def test_get_monthly_expiries(self, calendar):
        """Test getting all monthly expiries for a year."""
        expiries_2025 = calendar.get_monthly_expiries(2025)
        assert len(expiries_2025) == 12

        # Check first and last
        assert expiries_2025[0] == date(2025, 1, 17)
        assert expiries_2025[-1] == date(2025, 12, 19)

        # Test specific months
        q1_expiries = calendar.get_monthly_expiries(2025, months=[1, 2, 3])
        assert len(q1_expiries) == 3
        assert q1_expiries[0] == date(2025, 1, 17)
        assert q1_expiries[2] == date(2025, 3, 21)

    def test_trading_days_between(self, calendar):
        """Test counting trading days between dates."""
        # One week in January 2025
        start = datetime(2025, 1, 13)  # Monday
        end = datetime(2025, 1, 17)  # Friday

        trading_days = calendar.get_trading_days_between(start, end)
        assert len(trading_days) == 5  # Mon-Fri

        # Include MLK Day
        start = datetime(2025, 1, 17)  # Friday
        end = datetime(2025, 1, 21)  # Tuesday (Mon is MLK Day)

        trading_days = calendar.get_trading_days_between(start, end)
        assert len(trading_days) == 2  # Friday and Tuesday only

    def test_days_to_next_expiry(self, calendar):
        """Test calculating days to expiration."""
        # From Jan 10 to Jan 17 (third Friday)
        jan_10 = datetime(2025, 1, 10)
        days = calendar.days_to_next_expiry(jan_10)

        # Should be 5 trading days (Mon 13, Tue 14, Wed 15, Thu 16, Fri 17)
        assert days == 5

        # From Jan 17 (expiry day) should get next month
        jan_17 = datetime(2025, 1, 17)
        days = calendar.days_to_next_expiry(jan_17)

        # Count trading days from Jan 17 to Feb 21
        # Excluding MLK Day (Jan 20) and Presidents Day (Feb 17)
        assert days > 20  # More than 4 weeks of trading days

    def test_next_previous_trading_day(self, calendar):
        """Test getting next/previous trading days."""
        # From Friday to next Monday
        friday = datetime(2025, 1, 10)
        next_day = calendar.get_next_trading_day(friday)
        assert next_day.date() == date(2025, 1, 13)

        # From Monday to previous Friday
        monday = datetime(2025, 1, 13)
        prev_day = calendar.get_previous_trading_day(monday)
        assert prev_day.date() == date(2025, 1, 10)

        # Skip over holiday (MLK Day)
        friday_before_mlk = datetime(2025, 1, 17)
        next_day = calendar.get_next_trading_day(friday_before_mlk)
        assert next_day.date() == date(2025, 1, 21)  # Tuesday

    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        # Test is_trading_day
        assert not is_trading_day(datetime(2025, 1, 1))  # New Year's
        assert is_trading_day(datetime(2025, 1, 2))  # Jan 2

        # Test get_next_expiry_friday
        next_expiry = get_next_expiry_friday(datetime(2025, 1, 10))
        assert next_expiry.date() == date(2025, 1, 17)

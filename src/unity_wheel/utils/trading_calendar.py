"""Simple trading calendar for US equity markets.

Provides basic trading day detection, holiday management, and option expiration calculations.
No external dependencies required.
"""
from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta


class SimpleTradingCalendar:
    """Simple US equity market trading calendar.

    Handles:
    - Market holidays
    - Weekend detection
    - Standard option expiration dates (3rd Friday of month)
    """

    # US Market holidays for 2024-2026
    # Format: (month, day) or (month, week, weekday) for floating holidays
    FIXED_HOLIDAYS = {
        (1, 1),  # New Year's Day
        (7, 4),  # Independence Day
        (12, 25),  # Christmas
    }

    # Floating holidays: (month, occurrence, weekday)
    # weekday: 0=Monday, 4=Friday
    FLOATING_HOLIDAYS = {
        (1, 3, 0),  # MLK Day - 3rd Monday of January
        (2, 3, 0),  # Presidents Day - 3rd Monday of February
        (5, -1, 0),  # Memorial Day - Last Monday of May
        (9, 1, 0),  # Labor Day - 1st Monday of September
        (11, 4, 3),  # Thanksgiving - 4th Thursday of November
    }

    # Good Friday calculation requires Easter - simplified for 2024-2026
    GOOD_FRIDAYS = {
        date(2024, 3, 29),
        date(2025, 4, 18),
        date(2026, 4, 3),
    }

    # Juneteenth National Independence Day (since 2021)
    JUNETEENTH = (6, 19)

    def __init__(self) -> None:
        """Initialize the trading calendar."""
        self._holiday_cache: dict[int, set[date]] = {}

    def _calculate_floating_holiday(
        self, year: int, month: int, occurrence: int, weekday: int
    ) -> date:
        """Calculate date for floating holidays like MLK Day.

        Args:
            year: Year
            month: Month (1-12)
            occurrence: Which occurrence (1-5, or -1 for last)
            weekday: Day of week (0=Monday, 6=Sunday)

        Returns:
            Date of the holiday
        """
        if occurrence == -1:
            # Last occurrence of weekday in month
            last_day = calendar.monthrange(year, month)[1]
            d = date(year, month, last_day)
            while d.weekday() != weekday:
                d -= timedelta(days=1)
            return d
        else:
            # Nth occurrence
            first_day = date(year, month, 1)
            first_weekday = first_day.weekday()

            # Days until first occurrence of target weekday
            days_until = (weekday - first_weekday) % 7
            first_occurrence = first_day + timedelta(days=days_until)

            # Add weeks to get to nth occurrence
            return first_occurrence + timedelta(weeks=occurrence - 1)

    def _get_holidays_for_year(self, year: int) -> set[date]:
        """Get all market holidays for a given year.

        Returns:
            Set of holiday dates
        """
        if year in self._holiday_cache:
            return self._holiday_cache[year]

        holidays = set()

        # Fixed holidays
        for month, day in self.FIXED_HOLIDAYS:
            holiday = date(year, month, day)
            # If holiday falls on weekend, observe on nearest weekday
            if holiday.weekday() == 5:  # Saturday
                holidays.add(holiday - timedelta(days=1))  # Friday
            elif holiday.weekday() == 6:  # Sunday
                holidays.add(holiday + timedelta(days=1))  # Monday
            else:
                holidays.add(holiday)

        # Juneteenth
        juneteenth = date(year, self.JUNETEENTH[0], self.JUNETEENTH[1])
        if juneteenth.weekday() == 5:  # Saturday
            holidays.add(juneteenth - timedelta(days=1))
        elif juneteenth.weekday() == 6:  # Sunday
            holidays.add(juneteenth + timedelta(days=1))
        else:
            holidays.add(juneteenth)

        # Floating holidays
        for month, occurrence, weekday in self.FLOATING_HOLIDAYS:
            holidays.add(
                self._calculate_floating_holiday(year, month, occurrence, weekday)
            )

        # Thanksgiving Friday (day after Thanksgiving)
        thanksgiving = self._calculate_floating_holiday(year, 11, 4, 3)
        holidays.add(thanksgiving + timedelta(days=1))

        # Good Friday
        for gf in self.GOOD_FRIDAYS:
            if gf.year == year:
                holidays.add(gf)
                break

        self._holiday_cache[year] = holidays
        return holidays

    def is_trading_day(self, check_date: datetime | date) -> bool:
        """Check if a given date is a trading day.

        Args:
            check_date: Date to check

        Returns:
            True if market is open, False if weekend/holiday
        """
        # Convert to date if datetime
        if isinstance(check_date, datetime):
            check_date = check_date.date()

        # Check weekend
        if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check holidays
        holidays = self._get_holidays_for_year(check_date.year)
        return check_date not in holidays

    def get_next_trading_day(self, from_date: datetime) -> datetime:
        """Get the next trading day from a given date.

        Args:
            from_date: Starting date

        Returns:
            Next trading day
        """
        next_day = from_date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day

    def get_previous_trading_day(self, from_date: datetime) -> datetime:
        """Get the previous trading day from a given date.

        Args:
            from_date: Starting date

        Returns:
            Previous trading day
        """
        prev_day = from_date - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day

    def get_trading_days_between(
        self, start_date: datetime, end_date: datetime
    ) -> list[date]:
        """Get all trading days between two dates (inclusive).

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of trading days
        """
        trading_days = []
        current = start_date.date() if isinstance(start_date, datetime) else start_date
        end = end_date.date() if isinstance(end_date, datetime) else end_date

        while current <= end:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days

    def get_next_expiry_friday(self, from_date: datetime) -> datetime:
        """Get next standard monthly option expiration (3rd Friday).

        Standard monthly options expire on the 3rd Friday of each month.

        Args:
            from_date: Date to start searching from

        Returns:
            Next monthly option expiration date
        """
        # Start from next month if we're past the 3rd Friday of current month
        current = from_date.date() if isinstance(from_date, datetime) else from_date

        # Check current month first
        third_friday = self._get_third_friday(current.year, current.month)
        if third_friday > current:
            return datetime.combine(third_friday, datetime.min.time())

        # Move to next month
        if current.month == 12:
            year = current.year + 1
            month = 1
        else:
            year = current.year
            month = current.month + 1

        third_friday = self._get_third_friday(year, month)
        return datetime.combine(third_friday, datetime.min.time())

    def get_monthly_expiries(
        self, year: int, months: list[int] | None = None
    ) -> list[date]:
        """Get all monthly option expiration dates for a year.

        Args:
            year: Year to get expiries for
            months: Optional list of months (1-12), defaults to all months

        Returns:
            List of monthly expiration dates
        """
        if months is None:
            months = list(range(1, 13))

        expiries = []
        for month in months:
            if 1 <= month <= 12:
                expiries.append(self._get_third_friday(year, month))

        return sorted(expiries)

    def _get_third_friday(self, year: int, month: int) -> date:
        """Get the third Friday of a given month.

        Args:
            year: Year
            month: Month (1-12)

        Returns:
            Date of third Friday
        """
        # Find first Friday
        first_day = date(year, month, 1)
        first_friday = first_day
        while first_friday.weekday() != 4:  # Friday = 4
            first_friday += timedelta(days=1)

        # Third Friday is 2 weeks later
        return first_friday + timedelta(weeks=2)

    def is_expiry_friday(self, check_date: datetime) -> bool:
        """Check if a date is a standard monthly option expiration.

        Args:
            check_date: Date to check

        Returns:
            True if date is 3rd Friday of month
        """
        if isinstance(check_date, datetime):
            check_date = check_date.date()

        # Must be a Friday
        if check_date.weekday() != 4:
            return False

        # Check if it's the 3rd Friday
        third_friday = self._get_third_friday(check_date.year, check_date.month)
        return check_date == third_friday

    def days_to_next_expiry(self, from_date: datetime) -> int:
        """Calculate trading days until next monthly expiration.

        Args:
            from_date: Starting date

        Returns:
            Number of trading days until expiration
        """
        next_expiry = self.get_next_expiry_friday(from_date)
        trading_days = self.get_trading_days_between(from_date, next_expiry)
        # Subtract 1 to not count the starting date
        return len(trading_days) - 1 if trading_days else 0


# Convenience functions
_calendar = SimpleTradingCalendar()


def is_trading_day(date: datetime) -> bool:
    """Check if market is open on given date."""
    return _calendar.is_trading_day(date)


def get_next_expiry_friday(date: datetime) -> datetime:
    """Get next monthly option expiration date."""
    return _calendar.get_next_expiry_friday(date)


def days_to_expiry(from_date: datetime) -> int:
    """Get trading days to next monthly expiration."""
    return _calendar.days_to_next_expiry(from_date)

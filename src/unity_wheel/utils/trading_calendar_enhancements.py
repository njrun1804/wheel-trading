"""Enhanced trading calendar features for Unity options trading.

Additional features:
- Early market closes (half days)
- Extended holiday support
- Options-specific calculations
- Unity earnings calendar integration
"""

from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

from ..math import CalculationResult

from .trading_calendar import SimpleTradingCalendar


class EnhancedTradingCalendar(SimpleTradingCalendar):
    """Enhanced trading calendar with additional features.

    Adds support for:
    - Early market closes (1:00 PM ET)
    - Extended holiday calculations
    - Unity-specific earnings dates
    - Options expiration helpers
    """

    # Early close days (market closes at 1:00 PM ET instead of 4:00 PM ET)
    # Note: Black Friday is actually a full holiday in our calendar
    EARLY_CLOSE_DAYS = {
        # Christmas Eve (if weekday)
        "christmas_eve": lambda year: (
            date(year, 12, 24) if date(year, 12, 24).weekday() < 5 else None
        ),
        # Day before Independence Day (if weekday and July 4 is weekday)
        "july_3": lambda year: (
            date(year, 7, 3)
            if date(year, 7, 3).weekday() < 5 and date(year, 7, 4).weekday() < 5
            else None
        ),
        # New Year's Eve (if weekday)
        "new_years_eve": lambda year: (
            date(year, 12, 31) if date(year, 12, 31).weekday() < 5 else None
        ),
    }

    # Unity typical earnings months (quarterly)
    UNITY_EARNINGS_MONTHS = [2, 5, 8, 11]  # Feb, May, Aug, Nov

    # Good Friday calculations extended through 2030
    EXTENDED_GOOD_FRIDAYS = {
        date(2024, 3, 29),
        date(2025, 4, 18),
        date(2026, 4, 3),
        date(2027, 3, 26),
        date(2028, 4, 14),
        date(2029, 3, 30),
        date(2030, 4, 19),
    }

    def __init__(self):
        """Initialize enhanced calendar."""
        super().__init__()
        self._early_close_cache: Dict[int, List[date]] = {}

    def is_early_close(self, check_date: datetime) -> bool:
        """Check if market closes early on given date (1 PM ET).

        Args:
            check_date: Date to check

        Returns:
            True if early close day
        """
        if isinstance(check_date, datetime):
            check_date = check_date.date()

        year = check_date.year
        early_closes = self._get_early_closes_for_year(year)
        return check_date in early_closes

    def _get_early_closes_for_year(self, year: int) -> List[date]:
        """Get all early close dates for a year."""
        if year in self._early_close_cache:
            return self._early_close_cache[year]

        early_closes = []
        for name, calculator in self.EARLY_CLOSE_DAYS.items():
            early_close = calculator(year)
            if early_close and self.is_trading_day(early_close):
                early_closes.append(early_close)

        self._early_close_cache[year] = early_closes
        return early_closes

    def get_market_hours(self, check_date: datetime) -> Tuple[time, time]:
        """Get market open and close times for a date.

        Args:
            check_date: Date to check

        Returns:
            Tuple of (open_time, close_time) in ET
        """
        if not self.is_trading_day(check_date):
            return None, None

        open_time = time(9, 30)  # 9:30 AM ET

        if self.is_early_close(check_date):
            close_time = time(13, 0)  # 1:00 PM ET
        else:
            close_time = time(16, 0)  # 4:00 PM ET

        return open_time, close_time

    def trading_hours_remaining(self, from_time: datetime) -> CalculationResult:
        """Calculate trading hours remaining in the day.

        Args:
            from_time: Current time (assumed ET)

        Returns:
            CalculationResult with hours remaining and confidence
        """
        if not self.is_trading_day(from_time):
            return CalculationResult(0.0, 0.8, ["Non-trading day"])

        open_time, close_time = self.get_market_hours(from_time)
        current_time = from_time.time()

        # Convert times to minutes for easier calculation
        current_minutes = current_time.hour * 60 + current_time.minute
        close_minutes = close_time.hour * 60 + close_time.minute

        if current_minutes >= close_minutes:
            return CalculationResult(0.0, 0.9, ["After market close"])

        remaining_minutes = close_minutes - current_minutes
        return CalculationResult(remaining_minutes / 60.0, 1.0, [])

    def is_near_unity_earnings(self, check_date: datetime, days_buffer: int = 7) -> bool:
        """Check if date is near Unity earnings (typically 3rd week of earnings months).

        Unity reports earnings quarterly, usually in the 3rd week of:
        - February (Q4)
        - May (Q1)
        - August (Q2)
        - November (Q3)

        Args:
            check_date: Date to check
            days_buffer: Days before/after to consider "near"

        Returns:
            True if within buffer of typical earnings date
        """
        if isinstance(check_date, datetime):
            check_date = check_date.date()

        # Check if we're in an earnings month
        if check_date.month not in self.UNITY_EARNINGS_MONTHS:
            return False

        # Estimate earnings date as 3rd Thursday of the month
        # (typical for tech companies)
        earnings_estimate = self._calculate_floating_holiday(
            check_date.year, check_date.month, 3, 3  # 3rd Thursday
        )

        # Check if we're within buffer
        days_until = abs((earnings_estimate - check_date).days)
        return days_until <= days_buffer

    def get_expiry_fridays_avoiding_earnings(
        self, from_date: datetime, count: int = 3
    ) -> List[Tuple[date, bool]]:
        """Get next expiry Fridays with earnings warnings.

        Args:
            from_date: Starting date
            count: Number of expiries to return

        Returns:
            List of (expiry_date, near_earnings) tuples
        """
        expiries = []
        check_date = from_date

        while len(expiries) < count:
            expiry = self.get_next_expiry_friday(check_date)
            near_earnings = self.is_near_unity_earnings(expiry)
            expiries.append((expiry.date(), near_earnings))
            check_date = expiry + timedelta(days=1)

        return expiries

    def calculate_theta_decay_days(
        self, start_date: datetime, expiry_date: datetime
    ) -> Dict[str, float]:
        """Calculate detailed theta decay metrics.

        Options decay faster over weekends and holidays since
        time passes but markets are closed.

        Args:
            start_date: Position entry date
            expiry_date: Option expiration date

        Returns:
            Dict with decay metrics
        """
        # Total calendar days
        total_days = (expiry_date.date() - start_date.date()).days

        # Trading days
        trading_days = len(self.get_trading_days_between(start_date, expiry_date)) - 1

        # Non-trading days accelerate theta decay
        non_trading_days = total_days - trading_days

        # Early close days have reduced decay (only 5.5 hours vs 6.5)
        early_closes = 0
        current = start_date
        while current <= expiry_date:
            if self.is_early_close(current):
                early_closes += 1
            current += timedelta(days=1)

        # Effective decay days (weekends count as 0.5 days for theta)
        effective_days = trading_days + (non_trading_days * 0.5) - (early_closes * 0.15)

        return {
            "calendar_days": total_days,
            "trading_days": trading_days,
            "non_trading_days": non_trading_days,
            "early_close_days": early_closes,
            "effective_theta_days": effective_days,
            "daily_decay_rate": 1.0 / effective_days if effective_days > 0 else 0,
            "weekend_acceleration": non_trading_days / total_days if total_days > 0 else 0,
        }

    def optimal_entry_day(self, target_expiry: datetime) -> datetime:
        """Find optimal entry day for a target expiration.

        Considers:
        - Avoiding Mondays (weekend gap risk)
        - Avoiding Fridays (assignment risk)
        - Preferring mid-week entries

        Args:
            target_expiry: Target expiration date

        Returns:
            Suggested entry date
        """
        # Calculate target DTE
        target_dte = 45

        # Work backwards from expiry
        entry_date = target_expiry - timedelta(days=target_dte)

        # Adjust to trading day
        while not self.is_trading_day(entry_date):
            entry_date -= timedelta(days=1)

        # Prefer Tuesday-Thursday
        if entry_date.weekday() == 0:  # Monday
            entry_date += timedelta(days=1)
        elif entry_date.weekday() == 4:  # Friday
            entry_date -= timedelta(days=1)

        return entry_date


# Convenience functions
def get_market_hours(date: datetime) -> Tuple[time, time]:
    """Get market open/close for a date."""
    calendar = EnhancedTradingCalendar()
    return calendar.get_market_hours(date)


def is_near_unity_earnings(date: datetime, buffer: int = 7) -> bool:
    """Check if near Unity earnings date."""
    calendar = EnhancedTradingCalendar()
    return calendar.is_near_unity_earnings(date, buffer)


def calculate_theta_decay(start: datetime, expiry: datetime) -> Dict[str, float]:
    """Calculate theta decay metrics."""
    calendar = EnhancedTradingCalendar()
    return calendar.calculate_theta_decay_days(start, expiry)

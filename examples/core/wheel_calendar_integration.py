"""Example of integrating SimpleTradingCalendar with wheel strategy for Unity."""

from datetime import datetime, timedelta

from unity_wheel.utils import SimpleTradingCalendar


def find_optimal_expiration(target_dte: int = 45) -> datetime:
    """Find the optimal expiration date for Unity options.

    Args:
        target_dte: Target days to expiration (default 45)

    Returns:
        Optimal expiration date (monthly expiry)
    """
    calendar = SimpleTradingCalendar()
    today = datetime.now()

    # Get next few monthly expiries
    expiries = []
    check_date = today
    for _ in range(4):  # Check next 4 months
        expiry = calendar.get_next_expiry_friday(check_date)
        trading_days = len(calendar.get_trading_days_between(today, expiry)) - 1
        expiries.append((expiry, trading_days))
        check_date = expiry + timedelta(days=1)

    # Find closest to target DTE
    best_expiry = min(expiries, key=lambda x: abs(x[1] - target_dte))

    return best_expiry[0]


def validate_expiration_selection(expiry_date: datetime) -> bool:
    """Validate that selected expiration is appropriate for Unity.

    Unity earnings typically happen quarterly, so we want to avoid
    expiries that are too close to potential earnings dates.

    Args:
        expiry_date: Proposed expiration date

    Returns:
        True if expiration is valid
    """
    calendar = SimpleTradingCalendar()

    # Check it's a valid expiry Friday
    if not calendar.is_expiry_friday(expiry_date):
        print(f"Warning: {expiry_date.strftime('%Y-%m-%d')} is not a standard expiry Friday")
        return False

    # Check it's a trading day
    if not calendar.is_trading_day(expiry_date):
        print(f"Warning: {expiry_date.strftime('%Y-%m-%d')} is not a trading day")
        return False

    # Unity-specific: Avoid January expiries due to year-end volatility
    if expiry_date.month == 1:
        print("Warning: January expiries have higher vol due to Unity's year-end patterns")

    return True


def calculate_trading_days_metrics(start: datetime, expiry: datetime) -> dict:
    """Calculate detailed metrics for a given expiration.

    Args:
        start: Start date (today)
        expiry: Expiration date

    Returns:
        Dictionary with trading metrics
    """
    calendar = SimpleTradingCalendar()

    # Get all trading days
    trading_days = calendar.get_trading_days_between(start, expiry)

    # Calculate metrics
    total_days = (expiry.date() - start.date()).days
    trading_days_count = len(trading_days) - 1  # Don't count start day

    # Check for holidays in period
    holidays = []
    current = start
    while current <= expiry:
        if current.weekday() < 5 and not calendar.is_trading_day(current):
            holidays.append(current)
        current += timedelta(days=1)

    return {
        "expiry_date": expiry,
        "total_calendar_days": total_days,
        "trading_days": trading_days_count,
        "weekends": (total_days - trading_days_count - len(holidays)),
        "holidays": len(holidays),
        "holiday_dates": [h.strftime("%Y-%m-%d") for h in holidays],
        "daily_theta_multiplier": trading_days_count / total_days,
    }


def main():
    """Demonstrate calendar integration with wheel strategy."""
    print("Unity Wheel Strategy - Trading Calendar Integration")
    print("=" * 50)

    # Find optimal expiration
    target_dte = 45
    optimal_expiry = find_optimal_expiration(target_dte)
    print(f"\nTarget DTE: {target_dte} days")
    print(f"Optimal expiry: {optimal_expiry.strftime('%Y-%m-%d %A')}")

    # Validate the selection
    is_valid = validate_expiration_selection(optimal_expiry)
    print(f"Expiration valid: {is_valid}")

    # Calculate detailed metrics
    today = datetime.now()
    metrics = calculate_trading_days_metrics(today, optimal_expiry)

    print(f"\nExpiration Metrics:")
    print(f"  Calendar days: {metrics['total_calendar_days']}")
    print(f"  Trading days: {metrics['trading_days']}")
    print(f"  Weekends: {metrics['weekends']}")
    print(f"  Holidays: {metrics['holidays']}")
    if metrics["holiday_dates"]:
        print(f"  Holiday dates: {', '.join(metrics['holiday_dates'])}")
    print(f"  Theta multiplier: {metrics['daily_theta_multiplier']:.2f}")

    # Show next 3 monthly expiries with metrics
    print(f"\nNext Monthly Expiries (from {today.strftime('%Y-%m-%d')}):")
    calendar = SimpleTradingCalendar()
    check_date = today
    for i in range(3):
        expiry = calendar.get_next_expiry_friday(check_date)
        days = calendar.days_to_next_expiry(check_date)
        print(f"  {expiry.strftime('%Y-%m-%d')}: {days} trading days")
        check_date = expiry + timedelta(days=1)


if __name__ == "__main__":
    main()

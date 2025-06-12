"""Example usage of SimpleTradingCalendar for Unity options trading."""

from datetime import datetime

from unity_wheel.utils import (
    SimpleTradingCalendar,
    days_to_expiry,
    get_next_expiry_friday,
    is_trading_day,
)


def main():
    """Demonstrate trading calendar usage."""
    calendar = SimpleTradingCalendar()

    # Current date
    today = datetime.now()
    print(f"Today: {today.strftime('%Y-%m-%d %A')}")
    print(f"Is trading day: {is_trading_day(today)}")
    print()

    # Next expiry
    next_expiry = get_next_expiry_friday(today)
    print(f"Next monthly expiry: {next_expiry.strftime('%Y-%m-%d %A')}")
    print(f"Trading days to expiry: {days_to_expiry(today)}")
    print()

    # Check upcoming dates
    print("Next 5 trading days:")
    current = today
    for i in range(5):
        current = calendar.get_next_trading_day(current)
        print(f"  {current.strftime('%Y-%m-%d %A')}")
    print()

    # Monthly expiries for 2025
    print("2025 Monthly option expirations:")
    expiries = calendar.get_monthly_expiries(2025)
    for expiry in expiries:
        print(f"  {expiry.strftime('%Y-%m-%d %A')}")
    print()

    # Check specific holidays
    test_dates = [
        datetime(2025, 1, 1),  # New Year's
        datetime(2025, 1, 20),  # MLK Day
        datetime(2025, 7, 4),  # July 4th
        datetime(2025, 12, 25),  # Christmas
    ]

    print("Holiday checks:")
    for date in test_dates:
        trading = "Yes" if calendar.is_trading_day(date) else "No (Holiday)"
        print(f"  {date.strftime('%Y-%m-%d %A')}: Trading? {trading}")
    print()

    # Option expiration planning for Unity
    print("Unity option expiration planning:")
    print("For 45 DTE target, look for expiries around:")

    target_date = today
    for _ in range(3):  # Next 3 months
        expiry = calendar.get_next_expiry_friday(target_date)
        days = calendar.days_to_next_expiry(target_date)
        print(
            f"  {expiry.strftime('%Y-%m-%d')}: {days} trading days from {target_date.strftime('%Y-%m-%d')}"
        )
        target_date = expiry


if __name__ == "__main__":
    main()

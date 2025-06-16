"""Helper script to analyze trading calendars and option expirations for Unity.

Useful for:
- Finding optimal entry/exit dates
- Avoiding holidays and earnings
- Planning DTE targets
"""

from datetime import datetime, timedelta

from src.unity_wheel.utils.trading_calendar_enhancements import EnhancedTradingCalendar


def analyze_upcoming_expiries(months_ahead: int = 3) -> None:
    """Analyze upcoming option expiries with Unity-specific considerations."""
    calendar = EnhancedTradingCalendar()
    today = datetime.now()

    print("Trading Calendar Analysis for Unity (U)")
    print(f"Analysis Date: {today.strftime('%Y-%m-%d %A')}")
    print(f"Market Status: {'OPEN' if calendar.is_trading_day(today) else 'CLOSED'}")

    if calendar.is_early_close(today):
        print("⚠️  Early Close Today (1 PM ET)")

    print("\n" + "=" * 60 + "\n")

    # Get upcoming expiries
    expiries = calendar.get_expiry_fridays_avoiding_earnings(today, count=months_ahead)

    print("Upcoming Monthly Expiries:")
    print("-" * 60)
    print(f"{'Expiry Date':<15} {'DTE':<10} {'Earnings Risk':<15} {'Notes':<20}")
    print("-" * 60)

    for expiry_date, near_earnings in expiries:
        trading_days = len(calendar.get_trading_days_between(today, expiry_date)) - 1
        calendar_days = (expiry_date - today.date()).days

        # Check for other risks
        notes = []
        if near_earnings:
            notes.append("⚠️ Near earnings")
        if not calendar.is_trading_day(expiry_date):
            notes.append("❌ Holiday expiry!")
        if trading_days > 45:
            notes.append("Long DTE")
        elif trading_days < 30:
            notes.append("Short DTE")

        earnings_indicator = "YES ⚠️" if near_earnings else "No ✓"

        print(
            f"{expiry_date.strftime('%Y-%m-%d')} {trading_days:>3}T/{calendar_days:>3}C   {earnings_indicator:<15} {', '.join(notes)}"
        )

    print("\nLegend: T=Trading days, C=Calendar days")

    # Show optimal 45 DTE targets
    print("\n" + "=" * 60 + "\n")
    print("Optimal Entry Dates for 45 DTE Target:")
    print("-" * 60)

    for expiry_date, near_earnings in expiries[:3]:  # Next 3 months
        optimal_entry = calendar.optimal_entry_day(
            datetime.combine(expiry_date, datetime.min.time())
        )
        trading_days = (
            len(calendar.get_trading_days_between(optimal_entry, expiry_date)) - 1
        )

        print(
            f"For {expiry_date.strftime('%b %d')} expiry: Enter around {optimal_entry.strftime('%b %d (%a)')} ({trading_days} trading days)"
        )

        if near_earnings:
            print("  ⚠️  Warning: This expiry is near Unity earnings")


def check_holidays_and_early_closes(days_ahead: int = 30) -> None:
    """Check for upcoming holidays and early closes."""
    calendar = EnhancedTradingCalendar()
    today = datetime.now()
    end_date = today + timedelta(days=days_ahead)

    print("\nMarket Holidays & Early Closes")
    print(f"Next {days_ahead} days from {today.strftime('%Y-%m-%d')}")
    print("=" * 60)

    holidays = []
    early_closes = []

    current = today
    while current <= end_date:
        if current.weekday() < 5:  # Weekday
            if not calendar.is_trading_day(current):
                holidays.append(current)
            elif calendar.is_early_close(current):
                early_closes.append(current)
        current += timedelta(days=1)

    if holidays:
        print("\nMarket Holidays (CLOSED):")
        for holiday in holidays:
            print(f"  - {holiday.strftime('%Y-%m-%d %A')}")
    else:
        print("\nNo market holidays in period")

    if early_closes:
        print("\nEarly Closes (1 PM ET):")
        for early in early_closes:
            print(f"  - {early.strftime('%Y-%m-%d %A')}")
    else:
        print("\nNo early closes in period")


def calculate_theta_decay_comparison(dte_target: int = 45) -> None:
    """Compare theta decay for different entry dates."""
    calendar = EnhancedTradingCalendar()
    today = datetime.now()

    print("\nTheta Decay Analysis")
    print(f"Comparing {dte_target} DTE positions starting on different days")
    print("=" * 60)

    # Test different weekdays
    test_days = []
    current = today
    for i in range(7):
        if calendar.is_trading_day(current):
            test_days.append(current)
        current += timedelta(days=1)

    print(
        f"{'Entry Day':<15} {'Expiry':<12} {'Cal Days':<10} {'Trade Days':<12} {'Eff Theta':<12} {'Weekend %':<10}"
    )
    print("-" * 60)

    for entry_date in test_days:
        # Find expiry approximately DTE days out
        target_expiry = entry_date + timedelta(days=dte_target)
        actual_expiry = calendar.get_next_expiry_friday(target_expiry)

        # Calculate decay metrics
        decay = calendar.calculate_theta_decay_days(entry_date, actual_expiry)

        weekday_name = entry_date.strftime("%a")
        print(
            f"{entry_date.strftime('%Y-%m-%d')} {weekday_name:<4} "
            f"{actual_expiry.strftime('%Y-%m-%d')} "
            f"{decay['calendar_days']:>9} "
            f"{decay['trading_days']:>11} "
            f"{decay['effective_theta_days']:>11.1f} "
            f"{decay['weekend_acceleration']*100:>9.1f}%"
        )

    print("\nKey: Lower effective theta days = faster decay")


def main():
    """Run all calendar analyses."""
    print("\n🗓️  UNITY TRADING CALENDAR ANALYSIS\n")

    # Upcoming expiries
    analyze_upcoming_expiries(months_ahead=4)

    # Holidays and early closes
    check_holidays_and_early_closes(days_ahead=60)

    # Theta decay comparison
    calculate_theta_decay_comparison(dte_target=45)

    print("\n" + "=" * 60)
    print("💡 Tips:")
    print("- Avoid expiries near Unity earnings (Feb/May/Aug/Nov)")
    print("- Enter positions Tuesday-Thursday for better decay profile")
    print("- Watch for early closes when managing Friday expiries")
    print("- Account for holidays in DTE calculations")


if __name__ == "__main__":
    main()

"""Calculate exact data requirements for Unity wheel strategy validation."""

from datetime import datetime, timedelta

import pandas as pd

from unity_wheel.config.unified_config import get_config
config = get_config()



def calculate_data_requirements():
    """Calculate minimum data requirements for wheel strategy validation."""

    print("UNITY WHEEL STRATEGY - DATA REQUIREMENTS ANALYSIS")
    print("=" * 60)

    # Strategy parameters
    target_dte = config.trading.target_dte  # Days to expiry
    trades_per_year = 365 / target_dte  # ~8 trades/year
    min_trades_for_significance = 30  # Statistical significance

    print("\n1. MINIMUM HISTORICAL STOCK DATA:")
    print("-" * 40)

    # Calculate minimum time period
    min_years = min_trades_for_significance / trades_per_year
    min_trading_days = int(min_years * 252)  # 252 trading days/year

    print(f"   Minimum period: {min_years:.1f} years ({min_trading_days} trading days)")
    print(f"   Why: Need {min_trades_for_significance} trades for statistical significance")
    print(f"   At {target_dte} DTE, that's ~{trades_per_year:.1f} trades/year")

    print("\n   Required fields:")
    print("   - Date")
    print("   - Open (for gap risk analysis)")
    print("   - High (for intraday volatility)")
    print("   - Low (for assignment risk)")
    print("   - Close (for returns calculation)")
    print("   - Volume (optional but helpful)")

    print("\n2. RECOMMENDED HISTORICAL DATA:")
    print("-" * 40)

    # Different market regimes
    print("   Period: 2-3 years minimum")
    print("   Why: Capture different market conditions")
    print("   - Bull market periods")
    print("   - Bear market periods")
    print("   - High volatility (VIX > 30)")
    print("   - Low volatility (VIX < 15)")
    print("   - Earnings seasons (4x per year)")

    print("\n3. OPTIONS DATA (FOR REALISTIC BACKTESTING):")
    print("-" * 40)

    print("   Minimum: 6 months of end-of-day chains")
    print("   Recommended: 1+ years")
    print("   Required fields:")
    print("   - Strike prices")
    print("   - Expiration dates")
    print("   - Bid/Ask prices")
    print("   - Implied volatility")
    print("   - Volume/Open Interest")

    print("\n4. UNITY-SPECIFIC CONSIDERATIONS:")
    print("-" * 40)

    # Unity patterns
    print("   Earnings dates: Critical (Unity moves ±15-25%)")
    print("   Gap risk events: Track days with >10% moves")
    print("   Volatility regime: Unity IV typically 40-80%")
    print("   Strike intervals: Usually $2.50 increments")

    print("\n5. DATA VOLUME ESTIMATES:")
    print("-" * 40)

    # Calculate data size
    days_1yr = 252
    days_2yr = 504

    # Stock data
    stock_row_size = 6 * 8  # 6 fields * 8 bytes
    stock_1yr = days_1yr * stock_row_size / 1024
    stock_2yr = days_2yr * stock_row_size / 1024

    print(f"   Stock data (1 year): ~{stock_1yr:.1f} KB")
    print(f"   Stock data (2 years): ~{stock_2yr:.1f} KB")

    # Options data (more complex)
    strikes_per_expiry = 20  # Typical
    expiries_per_day = 8  # Weekly + monthly
    option_row_size = 10 * 8  # More fields

    options_1yr = days_1yr * strikes_per_expiry * expiries_per_day * option_row_size / 1024 / 1024

    print(f"   Options data (1 year): ~{options_1yr:.1f} MB")

    print("\n6. BACKTESTING WITHOUT FULL OPTIONS DATA:")
    print("-" * 40)

    print("   Use simplified premium model:")
    print("   - Black-Scholes with historical volatility")
    print("   - Add bid-ask spread estimate (0.05-0.10)")
    print("   - Adjust for volatility smile")
    print("   - Account for Unity's high volatility")

    print("\n7. FREE DATA SOURCES:")
    print("-" * 40)

    print("   Stock prices:")
    print("   - Yahoo Finance (1-2 years free)")
    print("   - Alpha Vantage (limited calls)")
    print("   - IEX Cloud (limited free tier)")

    print("   Options data:")
    print("   - Limited free sources")
    print("   - Consider paper trading to collect")
    print("   - Or use synthetic premiums")

    print("\n8. VALIDATION APPROACH WITH LIMITED DATA:")
    print("-" * 40)

    print("   Phase 1: Stock-only backtest (1 year)")
    print("   - Use historical volatility for premiums")
    print("   - Assume standard bid-ask spreads")
    print("   - Test core assignment logic")

    print("   Phase 2: Enhanced backtest (6 months)")
    print("   - Add actual option chains if available")
    print("   - Validate premium assumptions")
    print("   - Refine strike selection")

    print("   Phase 3: Paper trading validation")
    print("   - Run strategy with live data")
    print("   - Compare to backtest results")
    print("   - Adjust models as needed")

    return {
        "min_stock_days": min_trading_days,
        "min_years": min_years,
        "recommended_years": 2,
        "trades_needed": min_trades_for_significance,
    }


def estimate_backtest_quality(years_of_data: float):
    """Estimate backtest quality based on available data."""

    print(f"\nBACKTEST QUALITY WITH {years_of_data} YEARS OF DATA:")
    print("-" * 50)

    trades_per_year = 8  # 45 DTE average
    total_trades = int(years_of_data * trades_per_year)

    # Quality scoring
    if years_of_data < 1:
        quality = "Poor"
        confidence = "Low"
        recommendation = "Not recommended - insufficient data"
    elif years_of_data < 2:
        quality = "Fair"
        confidence = "Medium"
        recommendation = "Acceptable for initial testing"
    elif years_of_data < 3:
        quality = "Good"
        confidence = "High"
        recommendation = "Suitable for strategy validation"
    else:
        quality = "Excellent"
        confidence = "Very High"
        recommendation = "Ideal for parameter optimization"

    print(f"Expected trades: {total_trades}")
    print(f"Statistical significance: {'Yes' if total_trades >= 30 else 'No'}")
    print(f"Market regimes covered: {'Partial' if years_of_data < 2 else 'Multiple'}")
    print(f"Earnings cycles: {int(years_of_data * 4)}")
    print(f"Quality rating: {quality}")
    print(f"Confidence level: {confidence}")
    print(f"Recommendation: {recommendation}")

    return {
        "total_trades": total_trades,
        "quality": quality,
        "confidence": confidence,
    }


if __name__ == "__main__":
    # Calculate requirements
    reqs = calculate_data_requirements()

    print("\n" + "=" * 60)
    print("SUMMARY - MINIMUM VIABLE DATASET:")
    print("=" * 60)
    print(f"Stock data: {reqs['min_years']:.1f} years ({reqs['min_stock_days']} days)")
    print("Options data: 6 months (or synthetic)")
    print(f"Expected trades: {reqs['trades_needed']} minimum")
    print("Data size: <100 MB total")

    # Test different scenarios
    print("\n" + "=" * 60)
    print("QUALITY ASSESSMENT:")
    print("=" * 60)

    for years in [0.5, 1.0, 1.5, 2.0, 3.0]:
        estimate_backtest_quality(years)

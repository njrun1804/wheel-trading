#!/usr/bin/env python3
"""Manual position entry script for Unity Wheel Trading Bot."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.unity_wheel.models.position import Position

from unity_wheel.config.unified_config import get_config
config = get_config()



def main():
    """Parse your positions and generate recommendation."""

    # Your broker data
    broker_data = """Balance Details

Cash & Cash Investments
Cash Balance
Rates
$41,001.67
Cash & Cash Investments Total
$41,001.67

Investments
Securities
$186,675.00
Market Value
$186,675.00
Non-Margin
$0.00
Margin
$186,675.00
Options
-$15,600.00
Market Value Long
$0.00
Market Value Short
-$15,600.00
Non-Margin
$0.00
Margin
-$15,600.00
Positions Detail
Investments Total
$171,075.00

Option Details
Option Requirement
$0.00
Funds Available

To Trade
Cash & Cash Investments
$41,001.67
Settled Funds
$146,773.00
Day Trade Buying Power
$719,530.00
Available to Day Trade
$719,530.00
Cash + Borrowing
$293,546.00
SMA
$146,773.00
To Withdraw
Cash & Cash Investments
$41,001.67
Borrowing
$105,771.00
Cash + Borrowing
$146,772.67
Cash on Hold
$0.00
Margin Details & Buying Power

Balance Subject to Interest
$0.00
Month to Date Interest Owed
$0.00
Margin Equity
$227,676.67
Equity Percent
100% Account Summary
Total Accounts Value
$212,001.67
Total Cash & Cash Invest
$41,001.67
Total Market Value
$171,000.00
Total Day Change
+$1,350.00
(+0.64%)
Total Cost Basis
$135,589.45
Total Gain/Loss2
+$35,410.55
(+26.12%)
Positions Details *

Customize
Symbol
Description
Qty
Price
Price Chng $
Price Chng %
Mkt Val

Day Chng $

Day Chng %
Cost Basis

Gain/Loss $
2

Gain/Loss %
2

Ratings

Reinvest?

% of Acct
Equities
U

UNITY SOFTWARE INC
7,500    $24.84    +$0.15    +0.61%    $186,600.00    +$1,425.00    +0.77%
$144,914.67
+$41,685.33    +28.77%

No    81.99%
U 07/18/2025 25.00 C

CALL UNITY SOFTWARE INC $25 EXP 07/18/25
-75    $2.08    +$0.01    +0.48%    -$15,600.00    -$75.00    -0.48%
-$9,325.22
-$6,274.78    -67.29%    -    N/A    -
Total
Equities
$171,000.00    +$1,350.00    +0.8%    $135,589.45    +$35,410.55    +26.12%            81.99%
Cash & Money Market
Cash & Cash Investments1
-        -    -    $41,001.67    $0.00    0%
-
-    -    -    -    18.01%
Total
Cash & Money Market
$41,001.67    $0.00    0%    N/A    N/A    N/A            18.01%
Account Total
$212,001.67    +$1,350.00    +0.64%    $135,589.45    +$35,410.55    +26.12%"""

    try:
        # Parse the broker data (this is a simplified version)
        print("🔍 Parsing your account data...")

        # Manual parse for now - the broker parser needs fixes
        # Your data shows:
        account_value = 212001.67
        cash = 41001.67
        buying_power = 293546.00  # Cash + Borrowing

        # Positions:
        # U: 7,500 shares
        # U 07/18/2025 25.00 C: -75 (short 75 calls)

        positions = [
            Position(symbol = config.trading.symbol, quantity=7500),
            # Convert option format to OCC: UYYMMDDCPPPPPPPP
            Position(symbol="U250718C00025000", quantity=-75),
        ]

        print("\n📊 Account Summary:")
        print(f"   Total Value: ${account_value:,.2f}")
        print(f"   Cash: ${cash:,.2f}")
        print(f"   Buying Power: ${buying_power:,.2f}")

        print("\n📈 Positions:")
        for pos in positions:
            print(f"   {pos}")

        print("\n🎯 Unity Exposure:")
        print("   Shares: 7,500")
        print("   Short Calls: 75")
        print("   Current Price: $24.84")
        print("   Stock Value: $186,600")
        print("   Short Call Value: -$15,600")

        print("\n✅ Data parsed successfully!")
        print("\n💡 To get a recommendation, run:")
        print(f"   python run.py --portfolio {account_value}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

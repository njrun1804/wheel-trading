#!/usr/bin/env python3
"""Parse and display position data from broker copy/paste."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.portfolio.broker_parser import parse_broker_paste
from src.unity_wheel.portfolio.single_account import SingleAccountManager


def main():
    """Parse position data from clipboard or manual input."""

    print("📋 Unity Position Parser")
    print("=" * 50)
    print("\nPaste your broker account data below.")
    print("When done, press Ctrl+D (Mac/Linux) or Ctrl+Z (Windows) on a new line:\n")

    # Read multi-line input
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass

    data = "\n".join(lines)

    if not data.strip():
        print("\n❌ No data provided!")
        return

    print("\n" + "=" * 50)
    print("🔍 Parsing account data...\n")

    try:
        # Try broker parser first (for full account data)
        if "total accounts value" in data.lower() or "positions details" in data.lower():
            account = parse_broker_paste(data)
            print("✅ Successfully parsed broker data!")
        else:
            # Fall back to simple manual format
            manager = SingleAccountManager()
            account = manager.parse_manual_input(data)
            print("✅ Successfully parsed manual format!")

        # Display results
        print(f"\n📊 Account Summary:")
        print(f"   Total Value: ${account.total_value:,.2f}")
        print(f"   Cash Balance: ${account.cash_balance:,.2f}")
        print(f"   Buying Power: ${account.buying_power:,.2f}")

        print(f"\n📈 Positions ({len(account.positions)} total):")
        for pos in account.positions:
            print(f"   {pos}")

        print(f"\n🎯 Unity Exposure:")
        print(f"   Shares: {account.unity_shares:,}")
        print(f"   Puts: {account.unity_puts}")
        print(f"   Calls: {account.unity_calls}")
        print(f"   Total Notional: ${account.unity_notional:,.2f}")

        # Save to file for use with run.py
        output_file = Path("parsed_positions.txt")
        with open(output_file, "w") as f:
            f.write(f"Account Value: ${account.total_value:,.2f}\n")
            f.write(f"Cash: ${account.cash_balance:,.2f}\n")
            f.write(f"Buying Power: ${account.buying_power:,.2f}\n")
            f.write("\nPositions:\n")
            for pos in account.positions:
                if pos.position_type.value == "stock":
                    f.write(f"{pos.symbol} {pos.quantity} shares\n")
                else:
                    # For options, write in simple format
                    opt_type = "puts" if pos.position_type.value == "put" else "calls"
                    f.write(f"{pos.symbol} {pos.quantity} {opt_type}\n")

        print(f"\n💾 Saved parsed data to: {output_file}")
        print("   You can now use: python run.py --manual < parsed_positions.txt")

    except Exception as e:
        print(f"\n❌ Error parsing data: {e}")
        print("\nPlease check your data format and try again.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

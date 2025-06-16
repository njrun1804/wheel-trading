"""Example of simplified single Schwab account usage with hard failures."""

import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.api import SimpleWheelAdvisor
from src.unity_wheel.execution import UnityFillModel
from src.unity_wheel.portfolio import SingleAccountManager


def main():
    """Demonstrate single account usage with hard failures on missing data."""
    print("\n=== Single Schwab Account Demo ===\n")

    # Sample Schwab account data - ANY missing field will kill the program
    account_data = {
        "securitiesAccount": {
            "accountNumber": "12345678",
            "type": "MARGIN",
            "currentBalances": {
                "liquidationValue": 150000,
                "cashBalance": 50000,
                "buyingPower": 100000,
                "marginBuyingPower": 100000,
            },
            "positions": [
                {
                    "instrument": {"symbol": "U", "assetType": "EQUITY"},
                    "quantity": 1000,
                    "averagePrice": 35.00,
                    "marketValue": 35000,
                },
                {
                    "instrument": {"symbol": "U240119P00035000", "assetType": "OPTION"},
                    "quantity": -5,  # Short 5 puts
                    "averagePrice": 2.00,
                    "marketValue": -1000,
                },
            ],
        }
    }

    # Create account manager
    manager = SingleAccountManager()

    # Parse account - dies if ANY required field is missing
    account = manager.parse_account(account_data)

    print(f"Account ID: {account.account_id}")
    print(f"Total Value: ${account.total_value:,.2f}")
    print(f"Buying Power: ${account.buying_power:,.2f}")
    print("\nUnity Exposure:")
    print(f"  Shares: {account.unity_shares:,}")
    print(f"  Short Puts: {account.unity_puts}")
    print(f"  Total Notional: ${account.unity_notional:,.2f}")

    # Test validation - uncomment to see hard failures

    # This would die with "Insufficient buying power"
    # manager.validate_buying_power(200000, account)

    # This would die with "Position limit exceeded"
    # manager.validate_position_limits(50000, account)

    print("\n=== Fill Model Demo (Dies on Invalid Data) ===\n")

    fill_model = UnityFillModel()

    # Valid fill estimation
    estimate, confidence = fill_model.estimate_fill_price(
        bid=2.00, ask=2.10, size=5, is_opening=True, urgency=0.5
    )

    print(f"Fill estimate: ${estimate.fill_price:.2f}")
    print(f"Total cost: ${estimate.total_cost:.2f}")
    print(f"Confidence: {confidence:.0%}")

    # These would kill the program:
    # fill_model.estimate_fill_price(bid=0, ask=2.10, size=5, is_opening=True)  # Invalid bid
    # fill_model.estimate_fill_price(bid=2.10, ask=2.00, size=5, is_opening=True)  # bid > ask
    # fill_model.estimate_fill_price(bid=2.00, ask=2.10, size=-5, is_opening=True)  # Invalid size

    print("\n=== Simple Advisor Demo ===\n")

    # Create advisor
    advisor = SimpleWheelAdvisor()

    # Create market snapshot - missing ANY field will die
    market_snapshot = {
        "timestamp": datetime.now(),
        "ticker": "U",
        "current_price": 35.50,
        "buying_power": 100000,  # Will be overridden by actual account data
        "margin_used": 10000,
        "positions": [],
        "implied_volatility": 0.45,
        "risk_free_rate": 0.05,
        "option_chain": {
            "U_35P_45DTE": {
                "strike": 35.0,
                "expiration": "2024-02-15",
                "bid": 2.00,
                "ask": 2.10,
                "mid": 2.05,
                "volume": 150,
                "open_interest": 500,
                "delta": -0.30,
                "gamma": 0.02,
                "theta": -0.05,
                "vega": 0.15,
                "implied_volatility": 0.45,
            }
        },
    }

    # Get recommendation - dies if any required data missing
    recommendation = advisor.advise_with_fills(market_snapshot, account_data)

    print(f"Action: {recommendation['action']}")
    print(f"Rationale: {recommendation['rationale']}")
    print(f"Confidence: {recommendation['confidence']:.0%}")

    if "fill_estimate" in recommendation.get("details", {}):
        fill_est = recommendation["details"]["fill_estimate"]
        print("\nFill Estimate:")
        print(f"  Price: ${fill_est['estimated_fill']:.2f}")
        print(f"  Total Cost: ${fill_est['total_cost']:.2f}")

    if "account_summary" in recommendation.get("details", {}):
        acc_sum = recommendation["details"]["account_summary"]
        print("\nAccount Summary:")
        print(f"  Buying Power: ${acc_sum['buying_power']:,.2f}")
        print(f"  Unity Exposure: ${acc_sum['unity_notional']:,.2f}")

    print("\n✅ All validations passed - no missing data!")


def demonstrate_failures():
    """Demonstrate various failure modes - uncomment to test."""

    # Missing account data
    # manager = SingleAccountManager()
    # manager.parse_account(bad_account)  # Dies: Missing 'currentBalances'

    # Missing balance field
    # manager.parse_account(bad_account2)  # Dies: Missing 'cashBalance'

    # Invalid fill parameters
    # fill_model = UnityFillModel()
    # fill_model.estimate_fill_price(bid=-1, ask=2, size=5, is_opening=True)  # Dies: Invalid bid

    # Missing option data
    # advisor = SimpleWheelAdvisor()
    # advisor.advise_with_fills(market_snapshot_bad, account_data)  # Dies: No option data


if __name__ == "__main__":
    main()
    # demonstrate_failures()  # Uncomment to see failure modes

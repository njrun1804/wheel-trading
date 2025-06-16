#!/usr/bin/env python3
"""Example usage of the advise_position API."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.unity_wheel.api.advisor import WheelAdvisor
from src.unity_wheel.api.types import MarketSnapshot


def main() -> None:
    """Demonstrate API usage with Unity options."""

    # Example market snapshot for Unity
    market_snapshot: MarketSnapshot = {
        "timestamp": datetime.now(),
        "ticker": "U",
        "current_price": 35.50,
        "buying_power": 100000.0,
        "margin_used": 0.0,
        "positions": [],  # No existing positions
        "option_chain": {
            "30.0": {
                "strike": 30.0,
                "expiration": "2025-02-21",
                "bid": 0.45,
                "ask": 0.55,
                "mid": 0.50,
                "volume": 245,
                "open_interest": 1523,
                "delta": -0.15,
                "gamma": 0.02,
                "theta": -0.03,
                "vega": 0.08,
                "implied_volatility": 0.62,
            },
            "32.5": {
                "strike": 32.5,
                "expiration": "2025-02-21",
                "bid": 0.85,
                "ask": 0.95,
                "mid": 0.90,
                "volume": 512,
                "open_interest": 2834,
                "delta": -0.25,
                "gamma": 0.04,
                "theta": -0.04,
                "vega": 0.12,
                "implied_volatility": 0.64,
            },
            "35.0": {
                "strike": 35.0,
                "expiration": "2025-02-21",
                "bid": 1.45,
                "ask": 1.55,
                "mid": 1.50,
                "volume": 823,
                "open_interest": 4521,
                "delta": -0.40,
                "gamma": 0.05,
                "theta": -0.05,
                "vega": 0.15,
                "implied_volatility": 0.65,
            },
            "37.5": {
                "strike": 37.5,
                "expiration": "2025-02-21",
                "bid": 2.35,
                "ask": 2.45,
                "mid": 2.40,
                "volume": 412,
                "open_interest": 2103,
                "delta": -0.58,
                "gamma": 0.04,
                "theta": -0.06,
                "vega": 0.13,
                "implied_volatility": 0.67,
            },
            "40.0": {
                "strike": 40.0,
                "expiration": "2025-02-21",
                "bid": 3.80,
                "ask": 3.95,
                "mid": 3.875,
                "volume": 189,
                "open_interest": 987,
                "delta": -0.75,
                "gamma": 0.03,
                "theta": -0.07,
                "vega": 0.10,
                "implied_volatility": 0.70,
            },
        },
        "implied_volatility": 0.65,  # Overall IV for Unity
        "risk_free_rate": 0.05,
    }

    # Get recommendation
    print("🎯 Unity Wheel Trading Advisor")
    print("=" * 50)
    print(f"Current Price: ${market_snapshot['current_price']}")
    print(f"Buying Power: ${market_snapshot['buying_power']:,.0f}")
    print()

    # Create advisor and get recommendation
    advisor = WheelAdvisor()
    recommendation = advisor.advise_position(market_snapshot)

    print(f"Action: {recommendation['action']}")
    print(f"Rationale: {recommendation['rationale']}")
    print(f"Confidence: {recommendation['confidence']:.1%}")
    print()
    print("Risk Metrics:")
    print(f"  Max Loss: ${recommendation['risk']['max_loss']:,.0f}")
    print(f"  P(Assignment): {recommendation['risk']['probability_assignment']:.1%}")
    print(f"  Expected Return: {recommendation['risk']['expected_return']:.1%}")
    print(f"  Edge Ratio: {recommendation['risk']['edge_ratio']:.3f}")

    # Also show as JSON
    print("\nJSON Output:")
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()

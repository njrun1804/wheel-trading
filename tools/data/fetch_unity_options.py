#!/usr/bin/env python3
"""Fetch Unity options chain data from Schwab."""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.unity_wheel.schwab.client import SchwabClient
from src.unity_wheel.storage.duckdb_cache import DuckDBCache
from src.unity_wheel.utils.logging import get_logger

logger = get_logger(__name__)


async def fetch_unity_options(days_to_expiry: int = 60, store_in_cache: bool = True):
    """
    Fetch current Unity options chain.

    Args:
        days_to_expiry: Maximum days to expiration to fetch
        store_in_cache: Whether to store results in DuckDB cache

    Returns:
        Options chain data
    """
    try:
        # Initialize clients
        logger.info("Initializing Schwab client...")
        client_id = os.getenv("SCHWAB_CLIENT_ID")
        client_secret = os.getenv("SCHWAB_CLIENT_SECRET")

        if not client_id or not client_secret:
            logger.error(
                "Missing Schwab credentials. Set SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET"
            )
            return None

        async with SchwabClient(client_id, client_secret) as client:
            # Fetch Unity price first
            logger.info("Fetching Unity current price...")
            quote = await client.get_quote("U")
            current_price = quote.get("lastPrice", 0)
            logger.info(f"Unity current price: ${current_price:.2f}")

            # Fetch options chain
            logger.info(f"Fetching Unity options chain (up to {days_to_expiry} DTE)...")

            # Calculate date range
            from_date = datetime.now()
            to_date = from_date + timedelta(days=days_to_expiry)

            chain_response = await client.get_option_chain(
                symbol="U",
                from_date=from_date.strftime("%Y-%m-%d"),
                to_date=to_date.strftime("%Y-%m-%d"),
                strike_count=20,  # 20 strikes above and below
                include_quotes=True,
            )

            if not chain_response or chain_response.get("status") == "FAILED":
                logger.error("Failed to fetch options chain")
                return None

            # Process the chain
            puts = []
            calls = []

            put_exp_map = chain_response.get("putExpDateMap", {})
            call_exp_map = chain_response.get("callExpDateMap", {})

            # Process puts (what we care about for wheel strategy)
            for exp_date, strikes in put_exp_map.items():
                for strike, contracts in strikes.items():
                    for contract in contracts:
                        puts.append(
                            {
                                "symbol": contract["symbol"],
                                "underlying": "U",
                                "strike": float(strike),
                                "expiration": exp_date.split(":")[0],
                                "type": "put",
                                "bid": contract.get("bid", 0),
                                "ask": contract.get("ask", 0),
                                "last": contract.get("last", 0),
                                "volume": contract.get("totalVolume", 0),
                                "open_interest": contract.get("openInterest", 0),
                                "implied_volatility": contract.get("volatility", 0)
                                / 100,  # Convert to decimal
                                "delta": contract.get("delta", None),
                                "gamma": contract.get("gamma", None),
                                "theta": contract.get("theta", None),
                                "vega": contract.get("vega", None),
                                "rho": contract.get("rho", None),
                                "underlying_price": current_price,
                                "timestamp": datetime.now(),
                            }
                        )

            # Process calls (for covered calls if assigned)
            for exp_date, strikes in call_exp_map.items():
                for strike, contracts in strikes.items():
                    for contract in contracts:
                        calls.append(
                            {
                                "symbol": contract["symbol"],
                                "underlying": "U",
                                "strike": float(strike),
                                "expiration": exp_date.split(":")[0],
                                "type": "call",
                                "bid": contract.get("bid", 0),
                                "ask": contract.get("ask", 0),
                                "last": contract.get("last", 0),
                                "volume": contract.get("totalVolume", 0),
                                "open_interest": contract.get("openInterest", 0),
                                "implied_volatility": contract.get("volatility", 0) / 100,
                                "delta": contract.get("delta", None),
                                "gamma": contract.get("gamma", None),
                                "theta": contract.get("theta", None),
                                "vega": contract.get("vega", None),
                                "rho": contract.get("rho", None),
                                "underlying_price": current_price,
                                "timestamp": datetime.now(),
                            }
                        )

            logger.info(f"Fetched {len(puts)} put contracts and {len(calls)} call contracts")

            # Store in cache if requested
            if store_in_cache and (puts or calls):
                logger.info("Storing options data in cache...")
                cache = DuckDBCache()

                # Store as option chain
                chain_data = {
                    "symbol": "U",
                    "underlying_price": current_price,
                    "timestamp": datetime.now(),
                    "puts": puts,
                    "calls": calls,
                }

                # Use async context to store data
                await cache.initialize()
                await cache.store_option_chain("U", chain_data)
                logger.info(f"Stored {len(puts) + len(calls)} option contracts in cache")

            # Display summary
            if puts:
                print(f"\n📊 Unity Options Summary:")
                print(f"   Current price: ${current_price:.2f}")
                print(f"   Put contracts: {len(puts)}")
                print(f"   Call contracts: {len(calls)}")

                # Find ATM put for wheel strategy
                atm_strike = min(puts, key=lambda x: abs(x["strike"] - current_price * 0.95))
                print(f"\n   Suggested wheel put (5% OTM):")
                print(f"   Strike: ${atm_strike['strike']}")
                print(f"   Bid: ${atm_strike['bid']:.2f}")
                print(
                    f"   Delta: {atm_strike['delta']:.3f}"
                    if atm_strike["delta"]
                    else "   Delta: N/A"
                )
                print(f"   IV: {atm_strike['implied_volatility']*100:.1f}%")

            return {"puts": puts, "calls": calls, "underlying_price": current_price}

    except Exception as e:
        logger.error(f"Error fetching Unity options: {e}")
        import traceback

        traceback.print_exc()
        return None


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Unity options chain")
    parser.add_argument("--days", type=int, default=60, help="Days to expiry to fetch")
    parser.add_argument("--no-cache", action="store_true", help="Skip storing in cache")

    args = parser.parse_args()

    result = await fetch_unity_options(days_to_expiry=args.days, store_in_cache=not args.no_cache)

    if result:
        print(f"\n✅ Successfully fetched Unity options data")
    else:
        print(f"\n❌ Failed to fetch Unity options data")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

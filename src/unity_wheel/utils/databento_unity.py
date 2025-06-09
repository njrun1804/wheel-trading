"""
Databento Unity integration - correct implementation for Unity options.
Based on the official documentation showing Unity DOES have options via OPRA.PILLAR.
"""

from __future__ import annotations
from functools import lru_cache
from datetime import datetime, timedelta, timezone
import os
import databento as db
from databento import Schema, SType
import pandas as pd
import duckdb
from typing import Optional, List, Dict, Any

from ..utils.logging import get_logger
from ...config.loader import get_config

logger = get_logger(__name__)

# Initialize API with environment variable
API = db.Historical(key=os.getenv("DATABENTO_API_KEY"))
UTC = timezone.utc


@lru_cache
def chain(start: str, end: str) -> pd.DataFrame:
    """Return option definitions for Unity between two ISO dates."""
    logger.info(f"Fetching Unity option chain from {start} to {end}")

    df = API.timeseries.get_range(
        dataset="OPRA.PILLAR",
        schema=Schema.DEFINITION,
        stype_in=SType.PARENT,
        symbols=["U.OPT"],
        start=start,
        end=end,
    ).to_df()

    logger.info(f"Found {len(df)} Unity option definitions")
    return df


def quotes(
    raw_symbols: list[str], start: datetime, end: datetime, schema: str = "mbp-1"
) -> pd.DataFrame:
    """Get quotes for specific option symbols."""
    if not raw_symbols:
        return pd.DataFrame()

    logger.info(f"Fetching quotes for {len(raw_symbols)} Unity options")

    return API.timeseries.get_range(
        dataset="OPRA.PILLAR",
        schema=schema,
        stype_in=SType.RAW_SYMBOL,
        symbols=raw_symbols,
        start=start,
        end=end,
    ).to_df()


def spot(days_back: int = 2) -> pd.DataFrame:
    """Get Unity equity spot prices."""
    now = datetime.now(UTC)
    logger.info(f"Fetching Unity spot prices for last {days_back} days")

    return API.timeseries.get_range(
        dataset="EQUUS.MINI",
        schema="mbp-1",
        stype_in=SType.RAW_SYMBOL,
        symbols=[get_config().unity.ticker],
        start=now - timedelta(days=days_back),
        end=now,
    ).to_df()


def get_wheel_candidates(
    target_delta: float = 0.30, dte_range: tuple[int, int] = (30, 60), moneyness_range: float = 0.15
) -> pd.DataFrame:
    """
    Get Unity put options suitable for wheel strategy.

    Args:
        target_delta: Target delta (e.g., 0.30 for 30-delta puts)
        dte_range: (min_dte, max_dte) range
        moneyness_range: Max distance from ATM (e.g., 0.15 = ±15%)

    Returns:
        DataFrame with filtered options and quotes
    """
    now = datetime.now(UTC)

    # Get spot price first
    spot_df = spot(days_back=1)
    if spot_df.empty:
        logger.error("No Unity spot price available")
        return pd.DataFrame()

    # Use most recent spot price
    spot_px = float(spot_df["ask_px"].iloc[-1]) / 1e9  # Convert from fixed point
    logger.info(f"Unity spot price: ${spot_px:.2f}")

    # Calculate date range for options we care about
    min_exp = now + timedelta(days=dte_range[0])
    max_exp = now + timedelta(days=dte_range[1])

    # Get option definitions
    # Use T-1 for historical data
    end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=7)).strftime("%Y-%m-%d")

    defs = chain(start_date, end_date)

    if defs.empty:
        logger.error("No Unity option definitions found")
        return pd.DataFrame()

    # Convert timestamps
    defs["expiration"] = pd.to_datetime(defs["expiration"])

    # Filter to puts only and our criteria
    puts = defs[
        (defs["instrument_class"] == "P")
        & (defs["expiration"] >= min_exp)  # Puts only
        & (defs["expiration"] <= max_exp)
    ].copy()

    # Calculate moneyness and filter
    puts["moneyness"] = (puts["strike"] / spot_px) - 1
    puts = puts[puts["moneyness"].abs() <= moneyness_range]

    # Calculate DTE
    puts["dte"] = (puts["expiration"] - now).dt.days

    logger.info(f"Found {len(puts)} puts matching criteria")

    if puts.empty:
        return pd.DataFrame()

    # Get quotes for these options
    raw_syms = puts["raw_symbol"].unique().tolist()

    # Get recent quotes (T-1)
    quote_end = now - timedelta(days=1)
    quote_start = quote_end - timedelta(hours=8)  # Last trading hours

    quotes_df = quotes(raw_syms, quote_start, quote_end)

    if quotes_df.empty:
        logger.warning("No quotes available for Unity options")
        return puts  # Return definitions without quotes

    # Merge with latest quotes
    # Get most recent quote for each symbol
    latest_quotes = quotes_df.sort_values("ts_event").groupby("raw_symbol").last().reset_index()

    # Merge definitions with quotes
    result = puts.merge(
        latest_quotes[["raw_symbol", "bid_px", "ask_px", "bid_sz", "ask_sz"]],
        on="raw_symbol",
        how="left",
    )

    # Convert prices from fixed point
    result["bid"] = result["bid_px"] / 1e9
    result["ask"] = result["ask_px"] / 1e9
    result["mid"] = (result["bid"] + result["ask"]) / 2

    # Calculate simple Greeks approximations
    # This is simplified - in production use proper Black-Scholes
    result["approx_delta"] = -0.5 + (result["moneyness"] * 2)  # Simple linear approximation
    result["approx_delta"] = result["approx_delta"].clip(-1, 0)  # Puts are negative

    # Sort by closest to target delta
    result["delta_diff"] = (result["approx_delta"] - (-target_delta)).abs()
    result = result.sort_values("delta_diff")

    # Select key columns
    columns = [
        "raw_symbol",
        "strike",
        "expiration",
        "dte",
        "moneyness",
        "bid",
        "ask",
        "mid",
        "bid_sz",
        "ask_sz",
        "approx_delta",
    ]

    return result[columns]


def store_options_in_duckdb(options_df: pd.DataFrame) -> int:
    """Store Unity options in DuckDB cache."""
    if options_df.empty:
        return 0

    db_path = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")
    conn = duckdb.connect(db_path)

    try:
        # Get current Unity price
        current_price = float(
            conn.execute(
                f"SELECT close FROM price_history WHERE symbol = '{get_config().unity.ticker}' ORDER BY date DESC LIMIT 1"
            ).fetchone()[0]
        )

        # Clear old Unity options
        conn.execute(f"DELETE FROM options_data WHERE underlying = '{get_config().unity.ticker}'")

        # Insert new options
        for _, row in options_df.iterrows():
            conn.execute(
                """
                INSERT INTO options_data
                (symbol, underlying, strike, expiration, type, bid, ask, last,
                 volume, open_interest, implied_volatility, delta, gamma, theta,
                 vega, rho, underlying_price, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    row["raw_symbol"],
                    get_config().unity.ticker,
                    float(row["strike"]),
                    row["expiration"].date(),
                    "put",
                    float(row["bid"]),
                    float(row["ask"]),
                    float(row["mid"]),
                    int(row.get("volume", 100)),
                    int(row.get("open_interest", 1000)),
                    0.7,  # Placeholder IV
                    float(row["approx_delta"]),
                    0.01,  # Placeholder Greeks
                    -0.05,
                    0.1,
                    0.01,
                    current_price,
                    datetime.now(),
                ],
            )

        conn.commit()
        logger.info(f"Stored {len(options_df)} Unity options in DuckDB")
        return len(options_df)

    finally:
        conn.close()


def get_equity_bars(days: int = 250) -> pd.DataFrame:
    """Get Unity daily bars for risk calculations."""
    now = datetime.now(UTC)

    # Account for T-1 data availability
    end = now - timedelta(days=1)
    start = end - timedelta(days=int(days * 1.5))  # Extra buffer for weekends

    logger.info(f"Fetching Unity daily bars from {start.date()} to {end.date()}")

    # Get daily bars from EQUUS dataset
    bars = API.timeseries.get_range(
        dataset="EQUUS.MINI",
        schema="ohlcv-1d",
        stype_in=SType.RAW_SYMBOL,
        symbols=[get_config().unity.ticker],
        start=start,
        end=end,
    ).to_df()

    if not bars.empty:
        # Convert prices from fixed point
        for col in ["open", "high", "low", "close"]:
            if col in bars.columns:
                bars[col] = bars[col] / 1e9

        # Calculate returns
        bars["returns"] = bars["close"].pct_change()

        logger.info(f"Retrieved {len(bars)} days of Unity price data")

    return bars


def cost_estimate(start: str, end: str) -> Dict[str, Any]:
    """Estimate cost before pulling data."""
    est = API.metadata.get_cost(
        dataset="OPRA.PILLAR",
        schema="mbp-1",
        symbols=["U.OPT"],
        start=start,
        end=end,
    )

    return {"bytes": est.bytes, "cost_usd": est.cost, "readable_bytes": f"{est.bytes / 1e6:.1f} MB"}

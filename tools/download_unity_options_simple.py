#!/usr/bin/env python3
"""
Simple Unity options daily download - uses ohlcv-1d schema.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient

from unity_wheel.config.unified_config import get_config
config = get_config()


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Download Unity options daily data."""
    client = DatabentoClient()
    db_path = Path(config.storage.database_path).expanduser()
    conn = duckdb.connect(str(db_path))

    # Test with a specific date range
    start_date = datetime(2025, 6, 3).date()  # Recent date
    end_date = datetime(2025, 6, 7).date()  # Few days

    logger.info(f"Testing Unity options download from {start_date} to {end_date}")

    # Convert to UTC timestamps
    start = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=pytz.UTC)
    end = datetime.combine(end_date + timedelta(days=1), datetime.min.time()).replace(
        tzinfo=pytz.UTC
    )

    try:
        # Try ohlcv-1d schema for daily bars
        logger.info("Trying ohlcv-1d schema...")
        data = client.client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="ohlcv-1d",  # Daily OHLCV bars
            symbols=["U.OPT"],  # Unity options
            stype_in="parent",
            start=start,
            end=end,
            limit=10000,
        )

        # Convert to dataframe
        df = data.to_df()
        logger.info(f"Got {len(df)} records")

        if not df.empty:
            logger.info("Sample data:")
            logger.info(df.head())

            # Store in database
            for _, row in df.iterrows():
                # Get symbol
                symbol = row.get("symbol", row.get("raw_symbol", ""))

                # Extract date
                if "ts_event" in row:
                    trade_date = row["ts_event"].date()
                else:
                    trade_date = start_date

                # Parse option details from symbol
                expiration = None
                strike = None
                option_type = None

                if len(symbol) >= 15 and symbol.startswith("U"):
                    try:
                        # Unity option format: "U     250613C00032000"
                        # Find where the date starts (after spaces)
                        symbol_clean = symbol.strip()

                        # Extract expiration (YYMMDD) - starts after 'U' and spaces
                        for i in range(1, len(symbol)):
                            if symbol[i].isdigit():
                                exp_str = symbol[i : i + 6]
                                if len(exp_str) == 6 and exp_str.isdigit():
                                    expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                                break

                        # Extract type (C or P)
                        type_pos = None
                        if "C" in symbol:
                            option_type = "C"
                            type_pos = symbol.index("C")
                        elif "P" in symbol:
                            option_type = "P"
                            type_pos = symbol.index("P")

                        # Extract strike (8 digits after C/P)
                        if type_pos is not None and type_pos + 8 < len(symbol):
                            strike_str = symbol[type_pos + 1 : type_pos + 9]
                            if strike_str.isdigit():
                                strike = float(strike_str) / 1000.0
                    except Exception as e:
                        logger.debug(f"Failed to parse {symbol}: {e}")

                # Convert prices (Databento uses fixed point)
                def convert_price(val):
                    if val is None:
                        return None
                    if isinstance(val, (int, float)) and val > 1000:
                        return val / 10000.0
                    return float(val)

                # Only insert if we have valid data
                if expiration and strike and option_type:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO unity_options_daily
                        (date, symbol, expiration, strike, option_type,
                         last, volume, open_interest)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            trade_date,
                            symbol,
                            expiration,
                            strike,
                            option_type,
                            convert_price(row.get("close")),
                            row.get("volume", 0),
                            0,  # open_interest not in OHLCV
                        ),
                    )
                else:
                    logger.debug(f"Skipping {symbol} - incomplete parsing")

            conn.commit()
            logger.info(f"✅ Successfully stored {len(df)} Unity option records")

            # Show what we stored
            result = conn.execute(
                """
                SELECT COUNT(*) as count,
                       COUNT(DISTINCT symbol) as symbols,
                       MIN(date) as first_date,
                       MAX(date) as last_date
                FROM unity_options_daily
                WHERE date >= ?
            """,
                (start_date,),
            ).fetchone()

            logger.info(f"Database now contains:")
            logger.info(f"  Records: {result[0]}")
            logger.info(f"  Unique options: {result[1]}")
            logger.info(f"  Date range: {result[2]} to {result[3]}")

        else:
            logger.warning("No data returned from Databento")

    except Exception as e:
        logger.error(f"Error: {e}")

        # Try a different approach - get tick data and aggregate
        try:
            logger.info("\nTrying mbp-1 schema (market by price)...")
            data = client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="mbp-1",  # Market by price (best bid/ask)
                symbols=["U.OPT"],
                stype_in="parent",
                start=start,
                end=end,
                limit=1000,  # Just get a sample
            )

            df = data.to_df()
            logger.info(f"Got {len(df)} MBP records")

            if not df.empty:
                logger.info("Sample MBP data:")
                logger.info(df.head())
        except Exception as e2:
            logger.error(f"MBP also failed: {e2}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

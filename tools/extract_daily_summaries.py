#!/usr/bin/env python3
"""
Extract daily summaries from existing REAL data.
NO SYNTHETIC DATA - only summarizes what's actually in the database.
"""

from pathlib import Path

import duckdb

from unity_wheel.config.unified_config import get_config

config = get_config()


# Database path
db_path = Path(config.storage.database_path).expanduser()


def extract_daily_summaries():
    """Extract daily summaries from tick data."""
    conn = duckdb.connect(str(db_path))

    print("=" * 60)
    print("EXTRACTING DAILY SUMMARIES FROM REAL DATA")
    print("=" * 60)

    # Create daily summary table if not exists
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS unity_daily_summary_real (
            date DATE NOT NULL,
            symbol VARCHAR NOT NULL,
            data_type VARCHAR NOT NULL,
            open DECIMAL(10,2),
            high DECIMAL(10,2),
            low DECIMAL(10,2),
            close DECIMAL(10,2),
            volume BIGINT,
            record_count INTEGER,
            PRIMARY KEY (date, symbol, data_type)
        )
    """
    )

    # 1. Stock daily data - already have it
    print("\n1. STOCK DAILY DATA:")
    stock_count = conn.execute(
        """
        INSERT OR REPLACE INTO unity_daily_summary_real
        SELECT
            date,
            symbol,
            'STOCK' as data_type,
            open,
            high,
            low,
            close,
            volume,
            1 as record_count
        FROM price_history
        WHERE symbol = config.trading.symbol
    """
    ).fetchone()

    stock_days = conn.execute(
        """
        SELECT COUNT(*) FROM unity_daily_summary_real
        WHERE symbol = config.trading.symbol AND data_type = 'STOCK'
    """
    ).fetchone()[0]

    print(f"   ✓ {stock_days} days of stock data")

    # 2. Options daily summaries from tick data
    print("\n2. OPTIONS DAILY SUMMARIES:")

    # Get unique dates with options data
    options_dates = conn.execute(
        """
        SELECT DISTINCT trade_date
        FROM unity_options_ticks
        ORDER BY trade_date
    """
    ).fetchall()

    if options_dates:
        print(f"   Processing {len(options_dates)} days of options data...")

        for date_row in options_dates:
            date = date_row[0]

            # Get end-of-day snapshot for each option
            eod_data = conn.execute(
                """
                WITH eod_quotes AS (
                    SELECT
                        raw_symbol,
                        trade_date,
                        LAST_VALUE(bid_px) OVER (PARTITION BY raw_symbol ORDER BY ts_event) as eod_bid,
                        LAST_VALUE(ask_px) OVER (PARTITION BY raw_symbol ORDER BY ts_event) as eod_ask,
                        COUNT(*) OVER (PARTITION BY raw_symbol) as ticks
                    FROM unity_options_ticks
                    WHERE trade_date = ?
                    AND raw_symbol IS NOT NULL
                    AND raw_symbol != ''
                )
                SELECT
                    raw_symbol,
                    trade_date,
                    eod_bid,
                    eod_ask,
                    (eod_bid + eod_ask) / 2 as mid,
                    ticks
                FROM eod_quotes
                GROUP BY raw_symbol, trade_date, eod_bid, eod_ask, ticks
            """,
                (date,),
            ).fetchall()

            for row in eod_data:
                if row[0]:  # Has symbol
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO unity_daily_summary_real
                        (date, symbol, data_type, open, high, low, close, volume, record_count)
                        VALUES (?, ?, 'OPTION', ?, ?, ?, ?, 0, ?)
                    """,
                        (
                            row[1],  # date
                            row[0],  # symbol
                            row[4],  # mid as open
                            row[4],  # mid as high
                            row[4],  # mid as low
                            row[4],  # mid as close
                            row[5],  # tick count
                        ),
                    )

            print(f"   ✓ Processed {date}: {len(eod_data)} options")
    else:
        print("   No options tick data to summarize")

    # Show summary
    print("\n" + "=" * 60)
    print("DAILY SUMMARY STATISTICS:")
    print("=" * 60)

    summary = conn.execute(
        """
        SELECT
            data_type,
            COUNT(DISTINCT date) as days,
            COUNT(DISTINCT symbol) as symbols,
            MIN(date) as start_date,
            MAX(date) as end_date
        FROM unity_daily_summary_real
        GROUP BY data_type
    """
    ).fetchall()

    for row in summary:
        print(f"\n{row[0]}:")
        print(f"  Days: {row[1]}")
        print(f"  Symbols: {row[2]}")
        print(f"  Date range: {row[3]} to {row[4]}")

    # Sample daily data
    print("\n" + "=" * 60)
    print("SAMPLE DAILY DATA (latest available):")
    print("=" * 60)

    # Latest stock price
    latest_stock = conn.execute(
        """
        SELECT date, open, high, low, close, volume
        FROM unity_daily_summary_real
        WHERE data_type = 'STOCK'
        ORDER BY date DESC
        LIMIT 1
    """
    ).fetchone()

    if latest_stock:
        print(f"\nSTOCK (U) on {latest_stock[0]}:")
        print(f"  Open: ${latest_stock[1]:.2f}, High: ${latest_stock[2]:.2f}")
        print(f"  Low: ${latest_stock[3]:.2f}, Close: ${latest_stock[4]:.2f}")
        print(f"  Volume: {latest_stock[5]:,}")

    conn.commit()
    conn.close()

    print("\n✓ Daily summaries extracted from REAL data only")
    print("✓ NO SYNTHETIC DATA used or created")


if __name__ == "__main__":
    extract_daily_summaries()

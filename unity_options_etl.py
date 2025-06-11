#!/usr/bin/env python3
"""
Unity Options ETL - Turn-key ingestion for OPRA options data.
Based on Databento's Historical batch job output with ohlcv-1d schema.
"""

import glob
import io
import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import zstandard as zstd
from pyarrow import Table as pa_Table
from pyarrow import parquet as pq

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
RAW_GLOB = "data/unity-options/raw/*.csv.zst"
SYMBOLOGY_PATH = "data/unity-options/symbology/symbology.json"
OUTPUT_PATH = "data/unity-options/processed/unity_ohlcv_3y.parquet"

# Fixed-precision scalar used by Databento CSVs
_PRICE_Q = 1e-9  # 1 "pip" = 0.000000001
_TS_COL = "ts_event"  # ohlcv-1d has ts_event only
_NUM_COLS = ["open", "high", "low", "close", "volume"]
_DTYPES = {c: "int64" for c in _NUM_COLS} | {"instrument_id": "int32"}


def _decompress(path: str) -> io.BytesIO:
    """Decompress a .zst file into memory."""
    with open(path, "rb") as f:
        return io.BytesIO(zstd.decompress(f.read()))


def _read_single(path: str, symbol_map: dict) -> pd.DataFrame:
    """Read and process a single compressed CSV file."""
    logger.info(f"Processing {Path(path).name}")

    # Decompress into memory
    buf = _decompress(path)

    # Read CSV with explicit dtypes
    df = pd.read_csv(buf, dtype=_DTYPES)

    if df.empty:
        logger.warning(f"Empty file: {Path(path).name}")
        return pd.DataFrame()

    # nanosecond epoch → UTC Date
    df[_TS_COL] = pd.to_datetime(df[_TS_COL], unit="ns", utc=True).dt.date

    # Scale prices; leave volume as int
    for p in ["open", "high", "low", "close"]:
        df[p] = df[p] * _PRICE_Q

    # Map instrument_id → option symbol (OPRA)
    df["symbol"] = df["instrument_id"].map(symbol_map)

    # Filter out unmapped symbols
    mapped = df["symbol"].notna()
    if not mapped.any():
        logger.warning(f"No mapped symbols in {Path(path).name}")
        return pd.DataFrame()

    df = df[mapped]

    # Return tidy order
    return df[["symbol", _TS_COL, *_NUM_COLS]]


def load_symbology(path: str) -> dict:
    """Load symbology mapping from JSON."""
    logger.info(f"Loading symbology from {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Build instrument_id → symbol mapping
    mapper = {}
    for symbol, mappings in data.get("result", {}).items():
        for m in mappings:
            instrument_id = int(m["s"])
            mapper[instrument_id] = symbol

    logger.info(f"Loaded {len(mapper)} instrument mappings")
    return mapper


def refresh_unity_parquet(
    force: bool = False,
    raw_dir: str = "/Users/mikeedwards/Downloads/OPRA-20250611-NPFCXSAHG6",
    output_dir: str = "data/unity-options",
) -> pd.DataFrame:
    """
    Main ETL function - idempotent and can be called repeatedly.

    Args:
        force: Force re-processing even if output exists
        raw_dir: Directory containing downloaded OPRA files
        output_dir: Directory for processed output

    Returns:
        DataFrame of all Unity options data
    """
    # Set up paths
    output_path = Path(output_dir) / "processed" / "unity_ohlcv_3y.parquet"

    # Check if already processed
    if output_path.exists() and not force:
        logger.info(f"Loading existing parquet from {output_path}")
        return pd.read_parquet(output_path)

    # Create output directories
    os.makedirs(output_path.parent, exist_ok=True)

    # Load symbology - try JSON first, fall back to CSV
    symbology_json = Path(raw_dir) / "symbology.json"
    symbology_csv = Path(raw_dir) / "symbology.csv"

    if symbology_json.exists():
        symbol_map = load_symbology(str(symbology_json))
    elif symbology_csv.exists():
        # Fall back to CSV format
        logger.info("Using CSV symbology")
        sym_df = pd.read_csv(symbology_csv)
        symbol_map = dict(zip(sym_df["instrument_id"], sym_df["raw_symbol"]))
        logger.info(f"Loaded {len(symbol_map)} mappings from CSV")
    else:
        raise FileNotFoundError(f"No symbology file found in {raw_dir}")

    # Find all OHLCV files
    pattern = str(Path(raw_dir) / "*.ohlcv-1d.csv.zst")
    files = sorted(glob.glob(pattern))
    logger.info(f"Found {len(files)} OHLCV files to process")

    if not files:
        raise FileNotFoundError(f"No OHLCV files found matching {pattern}")

    # Process files in streaming fashion
    frames = []
    total_rows = 0

    for i, filepath in enumerate(files):
        df = _read_single(filepath, symbol_map)
        if not df.empty:
            frames.append(df)
            total_rows += len(df)

        # Log progress every 10 files
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(files)} files, {total_rows:,} rows so far")

    if not frames:
        logger.warning("No data found in any files!")
        return pd.DataFrame()

    # Concatenate all frames
    logger.info(f"Concatenating {len(frames)} dataframes with {total_rows:,} total rows")
    daily = pd.concat(frames, ignore_index=True)

    # Filter to Unity options only
    unity_mask = daily["symbol"].str.startswith("U", na=False)
    unity_daily = daily[unity_mask]
    logger.info(f"Filtered to {len(unity_daily):,} Unity option records")

    # Save to Parquet
    logger.info(f"Writing to {output_path}")
    pq.write_table(pa_Table.from_pandas(unity_daily), str(output_path))

    # Print summary statistics
    print("\n📊 ETL Summary:")
    print(f"   Total files processed: {len(files)}")
    print(f"   Total records: {len(daily):,}")
    print(f"   Unity records: {len(unity_daily):,}")
    print(f"   Date range: {unity_daily[_TS_COL].min()} to {unity_daily[_TS_COL].max()}")
    print(f"   Unique symbols: {unity_daily['symbol'].nunique():,}")
    print(f"   Output file: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return unity_daily


def query_examples(df: pd.DataFrame):
    """Show some example queries on the processed data."""
    print("\n📈 Sample Data:")

    # Parse option details from symbol
    df["expiration"] = (
        df["symbol"]
        .str.slice(1, 7)
        .apply(lambda x: pd.to_datetime(x, format="%y%m%d", errors="coerce"))
    )
    df["option_type"] = df["symbol"].str.slice(12, 13)
    df["strike"] = df["symbol"].str.slice(13, 21).astype(float) / 1000

    # Recent puts
    recent_puts = df[
        (df["option_type"] == "P") & (df["strike"].between(30, 40)) & (df[_TS_COL] >= "2025-01-01")
    ].nlargest(5, _TS_COL)

    print("\nRecent Unity Puts (30-40 strike):")
    for _, row in recent_puts.iterrows():
        print(f"   {row['symbol']}: ${row['close']:.2f} on {row[_TS_COL]} (vol: {row['volume']:,})")

    # Most liquid strikes
    liquidity = (
        df[(df["option_type"] == "P") & (df[_TS_COL] >= "2025-01-01")]
        .groupby("strike")["volume"]
        .agg(["mean", "count"])
        .round()
    )

    top_strikes = liquidity.nlargest(10, "mean")
    print("\nMost Liquid Put Strikes:")
    for strike, (avg_vol, days) in top_strikes.iterrows():
        print(f"   ${strike:.2f}: {avg_vol:,.0f} avg volume ({days} days)")


def integrate_with_duckdb(parquet_path: str, duckdb_path: str = "data/cache/wheel_cache.duckdb"):
    """Optional: Load parquet into existing DuckDB instance."""
    try:
        import duckdb

        logger.info(f"Loading parquet into DuckDB at {duckdb_path}")
        os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)

        conn = duckdb.connect(duckdb_path)

        # Create table from parquet
        conn.execute(
            f"""
            CREATE OR REPLACE TABLE unity_options_ohlcv AS
            SELECT * FROM read_parquet('{parquet_path}')
        """
        )

        # Create indexes
        conn.execute("CREATE INDEX idx_unity_symbol_date ON unity_options_ohlcv(symbol, ts_event)")

        # Show row count
        count = conn.execute("SELECT COUNT(*) FROM unity_options_ohlcv").fetchone()[0]
        logger.info(f"Loaded {count:,} rows into DuckDB")

        conn.close()

    except ImportError:
        logger.warning("DuckDB not available - skipping database integration")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unity Options ETL")
    parser.add_argument(
        "--raw-dir",
        default="/Users/mikeedwards/Downloads/OPRA-20250611-NPFCXSAHG6",
        help="Directory containing raw OPRA files",
    )
    parser.add_argument(
        "--output-dir", default="data/unity-options", help="Output directory for processed data"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-processing even if output exists"
    )
    parser.add_argument(
        "--integrate-duckdb", action="store_true", help="Load into DuckDB after processing"
    )

    args = parser.parse_args()

    # Run ETL
    df = refresh_unity_parquet(force=args.force, raw_dir=args.raw_dir, output_dir=args.output_dir)

    if not df.empty:
        # Show examples
        query_examples(df)

        # Optionally integrate with DuckDB
        if args.integrate_duckdb:
            parquet_path = Path(args.output_dir) / "processed" / "unity_ohlcv_3y.parquet"
            integrate_with_duckdb(str(parquet_path))
    else:
        print("❌ No data processed!")

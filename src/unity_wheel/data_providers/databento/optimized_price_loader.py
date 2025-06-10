"""
Optimized historical price loader for M4 Pro MacBook.
Handles Databento API limitations and maximizes throughput.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from databento_dbn import Schema

from src.config.loader import get_config
from src.unity_wheel.storage import Storage
from src.unity_wheel.utils import get_logger

from .auth_client import DatabentoClient

logger = get_logger(__name__)


class OptimizedPriceHistoryLoader:
    """Optimized loader for M4 Pro with proper error handling."""

    def __init__(self, client: DatabentoClient, storage: Storage):
        # Load configuration
        config = get_config()

        # M4 Pro optimizations
        self.MAX_WORKERS = config.databento.loader.max_workers
        self.CHUNK_SIZE = config.databento.loader.chunk_size

        # Databento API limits
        self.MAX_REQUESTS_PER_SECOND = config.databento.loader.max_requests_per_second
        self.RETRY_DELAYS = config.databento.loader.retry_delays

        # Data requirements from our analysis
        self.REQUIRED_DAYS = config.databento.loader.required_days
        self.MINIMUM_DAYS = config.databento.loader.minimum_days

        self.client = client
        self.storage = storage

        # Rate limiting
        self._request_semaphore = asyncio.Semaphore(10)  # Concurrent requests
        self._last_request_time = 0
        self._request_interval = 1.0 / self.MAX_REQUESTS_PER_SECOND

    async def ensure_table_exists(self):
        """Create price history table if it doesn't exist."""
        async with self.storage.cache.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS price_history (
                    symbol VARCHAR NOT NULL,
                    date DATE NOT NULL,
                    open DECIMAL(10,2),
                    high DECIMAL(10,2),
                    low DECIMAL(10,2),
                    close DECIMAL(10,2),
                    volume BIGINT,
                    returns DECIMAL(8,6),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date)
                )
            """
            )

            # Create index for faster queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_price_history_symbol_date
                ON price_history(symbol, date DESC)
            """
            )

    async def load_price_history(self, symbol: str, days: int = None) -> bool:
        """Load historical prices with optimized batching."""
        days = days or self.REQUIRED_DAYS

        logger.info(
            "loading_price_history_optimized",
            symbol=symbol,
            days=days,
            workers=self.MAX_WORKERS,
            chunk_size=self.CHUNK_SIZE,
        )

        # Ensure table exists first
        await self.ensure_table_exists()

        try:
            # Check existing data
            existing_days = await self._get_existing_days(symbol)
            logger.info(f"Existing data: {existing_days} days")

            if existing_days >= days:
                logger.info("Sufficient data already exists")
                return True

            # Calculate date ranges for chunked fetching
            end_date = datetime.now() - timedelta(days=1)
            chunks = self._calculate_date_chunks(days, end_date)

            logger.info(f"Fetching {len(chunks)} chunks of data")

            # Fetch chunks concurrently with rate limiting
            all_bars = []
            errors = 0

            for i, (chunk_start, chunk_end) in enumerate(chunks):
                try:
                    await self._rate_limit()

                    logger.info(
                        f"Fetching chunk {i+1}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()}"
                    )

                    bars = await self._fetch_chunk_with_retry(symbol, chunk_start, chunk_end)

                    if bars:
                        all_bars.extend(bars)
                        logger.info(f"  Retrieved {len(bars)} bars")

                        # Store incrementally to avoid memory issues
                        if len(all_bars) > 100:
                            await self._store_bars_batch(symbol, all_bars)
                            all_bars = []
                    else:
                        logger.warning(f"  No data for chunk")

                except Exception as e:
                    errors += 1
                    logger.error(f"Error fetching chunk {i+1}: {e}")

                    if errors > 3:
                        logger.error("Too many errors, aborting")
                        break

            # Store any remaining bars
            if all_bars:
                await self._store_bars_batch(symbol, all_bars)

            # Verify final count
            final_days = await self._get_existing_days(symbol)
            logger.info(f"Final data count: {final_days} days")

            return final_days >= self.MINIMUM_DAYS

        except Exception as e:
            logger.error(f"Critical error loading price history: {e}")
            return False

    def _calculate_date_chunks(self, total_days: int, end_date: datetime) -> List[tuple]:
        """Calculate optimal date chunks for fetching."""
        chunks = []

        # Work backwards from end_date
        current_end = end_date
        remaining_days = total_days

        while remaining_days > 0:
            chunk_days = min(remaining_days, self.CHUNK_SIZE)
            chunk_start = current_end - timedelta(days=chunk_days)

            # Adjust for weekends - extend range
            chunk_start = chunk_start - timedelta(days=int(chunk_days * 0.4))  # Add 40% buffer

            chunks.append((chunk_start, current_end))

            current_end = chunk_start - timedelta(days=1)
            remaining_days -= chunk_days

        return list(reversed(chunks))  # Chronological order

    async def _fetch_chunk_with_retry(
        self, symbol: str, start: datetime, end: datetime, attempt: int = 0
    ) -> List[Dict]:
        """Fetch a chunk with retry logic."""
        try:
            async with self._request_semaphore:
                dataset = self._get_dataset_for_symbol(symbol)

                # Log request details
                logger.debug(f"API Request: {dataset} {symbol} {start.date()} to {end.date()}")

                response = self.client.client.timeseries.get_range(
                    dataset=dataset, schema=Schema.OHLCV_1D, start=start, end=end, symbols=[symbol]
                )

                bars = []
                for bar in response:
                    bars.append(
                        {
                            "date": pd.to_datetime(bar.ts_event, unit="ns").date(),
                            "open": float(bar.open) / 1e9,
                            "high": float(bar.high) / 1e9,
                            "low": float(bar.low) / 1e9,
                            "close": float(bar.close) / 1e9,
                            "volume": bar.volume if hasattr(bar, "volume") else 0,
                        }
                    )

                return bars

        except Exception as e:
            if attempt < len(self.RETRY_DELAYS):
                delay = self.RETRY_DELAYS[attempt]
                logger.warning(f"Retry {attempt + 1} after {delay}s: {e}")
                await asyncio.sleep(delay)
                return await self._fetch_chunk_with_retry(symbol, start, end, attempt + 1)
            else:
                raise

    async def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._request_interval:
            await asyncio.sleep(self._request_interval - time_since_last)

        self._last_request_time = asyncio.get_event_loop().time()

    async def _store_bars_batch(self, symbol: str, bars: List[Dict]):
        """Store bars efficiently in batch."""
        if not bars:
            return

        # Sort by date
        bars = sorted(bars, key=lambda x: x["date"])

        # Calculate returns
        for i in range(1, len(bars)):
            prev_close = bars[i - 1]["close"]
            curr_close = bars[i]["close"]
            bars[i]["returns"] = (curr_close - prev_close) / prev_close if prev_close > 0 else 0

        bars[0]["returns"] = 0

        # Prepare batch data
        records = []
        for bar in bars:
            records.append(
                (
                    symbol,
                    bar["date"],
                    bar["open"],
                    bar["high"],
                    bar["low"],
                    bar["close"],
                    bar["volume"],
                    bar.get("returns", 0),
                )
            )

        # Batch insert with conflict resolution
        async with self.storage.cache.connection() as conn:
            # Use DuckDB's efficient bulk insert
            conn.execute("BEGIN TRANSACTION")

            try:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO price_history
                    (symbol, date, open, high, low, close, volume, returns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
                conn.execute("COMMIT")
                logger.info(f"Stored {len(records)} price records")

            except Exception as e:
                conn.execute("ROLLBACK")
                raise

    async def _get_existing_days(self, symbol: str) -> int:
        """Get count of existing days for symbol."""
        try:
            async with self.storage.cache.connection() as conn:
                result = conn.execute(
                    "SELECT COUNT(DISTINCT date) FROM price_history WHERE symbol = ?", [symbol]
                ).fetchone()

                return result[0] if result else 0
        except (sqlite3.Error, AttributeError) as e:
            logger.warning(f"Failed to get existing days count: {e}")
            return 0

    def _get_dataset_for_symbol(self, symbol: str) -> str:
        """Determine correct Databento dataset."""
        config = get_config()
        # Unity trades on NYSE American
        if symbol in [config.unity.ticker, "UNIT"]:
            return "XNAS.BASIC"  # Try NASDAQ first
        elif symbol in ["SPY", "QQQ", "IWM"]:
            return "ARCX.BASIC"  # NYSE Arca for ETFs
        else:
            return "XNAS.BASIC"  # Default

    async def verify_data_quality(self, symbol: str) -> Dict[str, any]:
        """Verify data quality after loading."""
        async with self.storage.cache.connection() as conn:
            # Check for gaps
            gaps_query = """
                WITH date_series AS (
                    SELECT
                        date,
                        LAG(date) OVER (ORDER BY date) as prev_date
                    FROM price_history
                    WHERE symbol = ?
                )
                SELECT COUNT(*) as gaps
                FROM date_series
                WHERE julianday(date) - julianday(prev_date) > 5  -- More than 5 days gap
            """

            gaps = conn.execute(gaps_query, [symbol]).fetchone()[0]

            # Get statistics
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_days,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    AVG(returns) * 252 as annual_return,
                    STDDEV(returns) * SQRT(252) as annual_vol,
                    MIN(returns) as worst_day,
                    MAX(returns) as best_day
                FROM price_history
                WHERE symbol = ?
            """,
                [symbol],
            ).fetchone()

            return {
                "symbol": symbol,
                "total_days": stats[0],
                "date_range": f"{stats[1]} to {stats[2]}",
                "gaps": gaps,
                "annual_return": f"{stats[3]:.1%}" if stats[3] else "N/A",
                "annual_volatility": f"{stats[4]:.1%}" if stats[4] else "N/A",
                "worst_day": f"{stats[5]:.2%}" if stats[5] else "N/A",
                "best_day": f"{stats[6]:.2%}" if stats[6] else "N/A",
                "data_quality": "GOOD" if gaps < 5 else "GAPS DETECTED",
            }

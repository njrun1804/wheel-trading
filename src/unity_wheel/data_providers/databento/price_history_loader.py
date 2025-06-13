"""
from __future__ import annotations

Load historical price data for risk calculations.
Only needs 250 days of daily bars - no options history required.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from databento_dbn import Schema

from ..config.loader import get_config
from ....storage import Storage
from ....utils import get_logger, with_recovery

from .client import DatabentoClient

logger = get_logger(__name__)


class PriceHistoryLoader:
    """Loads minimal historical data needed for wheel strategy."""

    # Based on risk analytics requirements
    REQUIRED_DAYS = 250  # For reliable VaR/CVaR calculations
    MINIMUM_DAYS = 20  # Absolute minimum for basic risk metrics

    def __init__(self, client: DatabentoClient, storage: Storage):
        self.client = client
        self.storage = storage

    async def load_price_history(self, symbol: str, days: int = None) -> bool:
        """Load historical daily prices for risk calculations.

        Args:
            symbol: Stock symbol (e.g., 'U')
            days: Number of days to load (default: REQUIRED_DAYS)

        Returns:
            Success status
        """
        days = days or self.REQUIRED_DAYS

        logger.info("loading_price_history", symbol=symbol, days=days)

        try:
            # Calculate date range (add buffer for weekends/holidays)
            end_date = datetime.now() - timedelta(days=1)  # Yesterday
            start_date = end_date - timedelta(days=int(days * 1.5))  # Extra for non-trading days

            # Fetch daily bars from Databento
            bars = await self._fetch_daily_bars(symbol, start_date, end_date)

            if len(bars) < self.MINIMUM_DAYS:
                logger.error(
                    "insufficient_price_data",
                    symbol=symbol,
                    received=len(bars),
                    minimum=self.MINIMUM_DAYS,
                )
                return False

            # Store in DuckDB for risk calculations
            await self._store_price_history(symbol, bars)

            logger.info(
                "price_history_loaded",
                symbol=symbol,
                bars_loaded=len(bars),
                date_range=f"{bars[0]['date']} to {bars[-1]['date']}",
            )

            return True

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("price_history_error", symbol=symbol, error=str(e))
            return False

    @with_recovery(max_attempts=3)
    async def _fetch_daily_bars(self, symbol: str, start: datetime, end: datetime) -> List[Dict]:
        """Fetch daily OHLCV bars from Databento."""

        bars = []

        # Use the appropriate dataset for the symbol
        dataset = self._get_dataset_for_symbol(symbol)

        response = self.client.client.timeseries.get_range(
            dataset=dataset,
            schema=Schema.OHLCV_1D,  # Daily bars
            start=start,
            end=end,
            symbols=[symbol],
        )

        for bar in response:
            bars.append(
                {
                    "date": pd.to_datetime(bar.ts_event, unit="ns").date(),
                    "open": float(bar.open) / 1e9,  # Databento prices in nano
                    "high": float(bar.high) / 1e9,
                    "low": float(bar.low) / 1e9,
                    "close": float(bar.close) / 1e9,
                    "volume": bar.volume,
                }
            )

        return sorted(bars, key=lambda x: x["date"])

    async def _store_price_history(self, symbol: str, bars: List[Dict]):
        """Store price history in DuckDB."""

        # Create table if not exists
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
                    returns DECIMAL(8,6),  -- Daily return for risk calcs
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date)
                )
            """
            )

        # Calculate returns
        for i in range(1, len(bars)):
            prev_close = bars[i - 1]["close"]
            curr_close = bars[i]["close"]
            bars[i]["returns"] = (curr_close - prev_close) / prev_close if prev_close > 0 else 0

        bars[0]["returns"] = 0  # First day has no return

        # Batch insert
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

            await conn.executemany(
                """
                INSERT OR REPLACE INTO price_history
                (symbol, date, open, high, low, close, volume, returns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )

    async def update_daily_price(self, symbol: str) -> bool:
        """Update with latest daily price (for daily cron job)."""

        try:
            # Get yesterday's close
            yesterday = datetime.now().date() - timedelta(days=1)

            bars = await self._fetch_daily_bars(
                symbol, datetime.combine(yesterday, datetime.min.time()), datetime.now()
            )

            if bars:
                await self._store_price_history(symbol, bars[-1:])  # Just the latest

                # Trim old data to keep storage bounded
                cutoff = datetime.now().date() - timedelta(days=self.REQUIRED_DAYS + 30)
                async with self.storage.cache.connection() as conn:
                    conn.execute(
                        "DELETE FROM price_history WHERE symbol = ? AND date < ?", [symbol, cutoff]
                    )

                return True

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("daily_update_error", symbol=symbol, error=str(e))

        return False

    async def get_returns_for_risk(
        self, symbol: str, days: Optional[int] = None
    ) -> Optional[pd.Series]:
        """Get returns series for risk calculations."""

        days = days or self.REQUIRED_DAYS
        cutoff = datetime.now().date() - timedelta(days=days)

        async with self.storage.cache.connection() as conn:
            result = conn.execute(
                """
                SELECT date, returns
                FROM price_history
                WHERE symbol = ? AND date >= ?
                ORDER BY date
                """,
                [symbol, cutoff],
            ).fetchall()

        if not result:
            return None

        dates, returns = zip(*result)
        return pd.Series(returns, index=pd.to_datetime(dates), name=symbol)

    def _get_dataset_for_symbol(self, symbol: str) -> str:
        """Determine appropriate Databento dataset for symbol."""
        config = get_config()
        unity_ticker = config.unity.ticker

        # This is simplified - in production would have proper mapping
        # Unity trades on NYSE American
        if symbol == unity_ticker or symbol == "UNIT":
            return "XASE.BASIC"  # NYSE American
        elif symbol in ["SPY", "QQQ", "IWM"]:
            return "ARCX.BASIC"  # NYSE Arca for ETFs
        else:
            return "XNAS.BASIC"  # Default to NASDAQ

    async def check_data_availability(self, symbol: str) -> Dict[str, any]:
        """Check how much historical data is available."""

        async with self.storage.cache.connection() as conn:
            result = conn.execute(
                """
                SELECT
                    COUNT(*) as days,
                    MIN(date) as earliest,
                    MAX(date) as latest,
                    AVG(returns) as avg_return,
                    STDDEV(returns) as volatility
                FROM price_history
                WHERE symbol = ?
                """,
                [symbol],
            ).fetchone()

        days, earliest, latest, avg_return, volatility = result

        return {
            "symbol": symbol,
            "days_available": days or 0,
            "date_range": f"{earliest} to {latest}" if earliest else "No data",
            "sufficient_for_risk": days >= self.MINIMUM_DAYS,
            "optimal_data": days >= self.REQUIRED_DAYS,
            "annualized_return": (avg_return or 0) * 252,
            "annualized_volatility": (volatility or 0) * (252**0.5),
        }
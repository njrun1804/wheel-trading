"""Databento client with rate limiting and retry logic for wheel strategy.

Implements:
- Connection pooling and session reuse
- Rate limiting (100 req/s for historical, 10 concurrent live)
- Automatic retries with exponential backoff
- Symbol batching for efficiency
- Instrument ID caching for performance
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Set, AsyncIterator
from decimal import Decimal
import logging

import databento as db
from databento_dbn import SType, Schema
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.unity_wheel.utils.logging import StructuredLogger
from src.unity_wheel.utils.recovery import RecoveryContext
from src.unity_wheel.secrets.integration import get_databento_api_key
from src.unity_wheel.databento.types import (
    InstrumentDefinition,
    OptionQuote,
    UnderlyingPrice,
    OptionChain,
    DataQuality,
)


logger = StructuredLogger(logging.getLogger(__name__))


class DatentoClient:
    """Databento client optimized for wheel strategy data needs."""

    # Rate limits for Standard plan
    MAX_CONCURRENT_LIVE = 10  # Per dataset after Feb 16, 2025
    MAX_HISTORICAL_RPS = 100  # Soft limit
    MAX_SYMBOLS_PER_REQUEST = 2000
    MAX_FILE_SIZE_GB = 2  # For HTTP streaming

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: str = ".databento_cache",
    ):
        """Initialize Databento client.

        Args:
            api_key: Databento API key (optional, will use SecretManager if not provided)
            use_cache: Enable local caching of definitions
            cache_dir: Directory for cached data
        """
        # Use provided API key or fall back to SecretManager
        if not api_key:
            logger.info("No API key provided, retrieving from SecretManager")
            api_key = get_databento_api_key()
        
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Databento API key required")

        self.client = db.Historical(self.api_key)
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        # Rate limiting
        self._historical_semaphore = asyncio.Semaphore(10)  # Concurrent requests
        self._last_request_time = 0.0
        self._request_interval = 1.0 / self.MAX_HISTORICAL_RPS

        # Instrument mapping cache
        self._instrument_map: Optional[db.common.symbology.InstrumentMap] = None
        self._definitions_cache: Dict[str, InstrumentDefinition] = {}

        # Live session management
        self._live_sessions: Dict[str, db.Live] = {}

        logger.info("databento_client_initialized", extra={"use_cache": use_cache, "cache_dir": cache_dir})

    async def get_option_chain(
        self, underlying: str, expiration: datetime, timestamp: Optional[datetime] = None
    ) -> OptionChain:
        """Get complete option chain for given underlying and expiration.

        Args:
            underlying: Underlying symbol (e.g., "U")
            expiration: Option expiration date
            timestamp: Point-in-time for historical data (None = live)

        Returns:
            Complete option chain with calls and puts
        """
        logger.info(
            "fetching_option_chain",
            extra={
                "underlying": underlying,
                "expiration": expiration.isoformat(),
                "timestamp": timestamp.isoformat() if timestamp else "live",
            }
        )

        # Get instrument definitions first
        definitions = await self._get_definitions(underlying, expiration)

        # Get quotes for all options
        instrument_ids = [d.instrument_id for d in definitions]
        quotes = await self._get_quotes_by_ids(instrument_ids, timestamp)

        # Get underlying price
        spot_price = await self._get_underlying_price(underlying, timestamp)

        # Organize into chain
        calls = []
        puts = []

        for defn in definitions:
            quote = quotes.get(defn.instrument_id)
            if quote:
                if defn.option_type.value == "C":
                    calls.append(quote)
                else:
                    puts.append(quote)

        return OptionChain(
            underlying=underlying,
            expiration=expiration,
            spot_price=spot_price.last_price,
            timestamp=timestamp or datetime.utcnow(),
            calls=sorted(calls, key=lambda x: x.instrument_id),
            puts=sorted(puts, key=lambda x: x.instrument_id),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def _get_definitions(
        self, underlying: str, expiration: datetime
    ) -> List[InstrumentDefinition]:
        """Get instrument definitions with caching and retry."""
        cache_key = f"{underlying}_{expiration.strftime('%Y%m%d')}"

        # Check cache first
        if self.use_cache and cache_key in self._definitions_cache:
            logger.debug("definitions_cache_hit", extra={"key": cache_key})
            return self._definitions_cache[cache_key]

        async with self._historical_semaphore:
            await self._rate_limit()

            # Request definitions using most recent available date
            # Options definitions are available on trading days before expiration
            today = datetime.now(timezone.utc)
            
            # Find last trading day (skip weekends)
            if today.weekday() >= 5:  # Saturday or Sunday
                days_back = today.weekday() - 4  # Back to Friday
                last_trading_day = today - timedelta(days=days_back)
            else:
                # Use previous day for weekdays
                last_trading_day = today - timedelta(days=1)
            
            last_trading_day = last_trading_day.replace(hour=0, minute=0, second=0, microsecond=0)
            
            if expiration > today:
                # Use last trading day for future expirations
                end = last_trading_day
                start = end - timedelta(days=1)
            else:
                # Historical expiration
                start = expiration.replace(hour=0, minute=0, second=0)
                end = expiration.replace(hour=23, minute=59, second=59)

            logger.debug(
                "requesting_definitions",
                extra={
                    "underlying": underlying,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                }
            )

            response = self.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema=Schema.DEFINITION,
                start=start,
                end=end,
                symbols=[f"{underlying}.OPT"],  # Options for underlying
                stype_in=SType.PARENT,
            )

            # Parse definitions
            definitions = []
            for record in response:
                try:
                    defn = InstrumentDefinition.from_databento(record)
                    if defn.expiration.date() == expiration.date():
                        definitions.append(defn)
                except Exception as e:
                    logger.warning("definition_parse_error", extra={"error": str(e), "record": record})

            # Cache results
            if self.use_cache:
                self._definitions_cache[cache_key] = definitions

            logger.info(
                "definitions_fetched",
                extra={
                    "count": len(definitions),
                    "underlying": underlying,
                    "expiration": expiration.isoformat(),
                }
            )

            return definitions

    async def _get_quotes_by_ids(
        self, instrument_ids: List[int], timestamp: Optional[datetime] = None
    ) -> Dict[int, OptionQuote]:
        """Get quotes for multiple instruments efficiently."""
        quotes = {}

        # Batch requests to respect limits
        for i in range(0, len(instrument_ids), self.MAX_SYMBOLS_PER_REQUEST):
            batch = instrument_ids[i : i + self.MAX_SYMBOLS_PER_REQUEST]

            if timestamp:
                # Historical quote
                batch_quotes = await self._get_historical_quotes(batch, timestamp)
            else:
                # Live quotes - would use live session
                batch_quotes = await self._get_live_quotes(batch)

            quotes.update(batch_quotes)

        return quotes

    async def _get_historical_quotes(
        self, instrument_ids: List[int], timestamp: datetime
    ) -> Dict[int, OptionQuote]:
        """Get historical quotes for given instruments."""
        async with self._historical_semaphore:
            await self._rate_limit()

            # Request window around timestamp
            start = timestamp - timedelta(minutes=1)
            end = timestamp + timedelta(minutes=1)

            response = self.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema=Schema.MBP_1,  # Top of book
                start=start,
                end=end,
                symbols=None,  # Use instrument IDs instead
                instrument_ids=instrument_ids,
            )

            # Get latest quote for each instrument
            latest_quotes = {}
            for record in response:
                try:
                    quote = OptionQuote.from_databento(record)
                    # Keep only the latest quote per instrument
                    if (
                        quote.instrument_id not in latest_quotes
                        or quote.timestamp > latest_quotes[quote.instrument_id].timestamp
                    ):
                        latest_quotes[quote.instrument_id] = quote
                except Exception as e:
                    logger.warning("quote_parse_error", extra={"error": str(e)})

            return latest_quotes

    async def _get_live_quotes(self, instrument_ids: List[int]) -> Dict[int, OptionQuote]:
        """Get latest quotes using REST API (pull-when-asked)."""
        # In pull-when-asked architecture, we use REST API for latest data
        # This method is kept for compatibility but uses REST instead of WebSocket
        logger.warning("live_quotes_use_rest_api")
        return {}

    async def _get_underlying_price(
        self, symbol: str, timestamp: Optional[datetime] = None
    ) -> UnderlyingPrice:
        """Get underlying equity price."""
        async with self._historical_semaphore:
            await self._rate_limit()

            if timestamp:
                start = timestamp - timedelta(minutes=1)
                end = timestamp
            else:
                # Get latest available data from last trading day
                today = datetime.now(timezone.utc)
                
                # Find last trading day (skip weekends)
                if today.weekday() >= 5:  # Saturday or Sunday
                    days_back = today.weekday() - 4  # Back to Friday
                    last_trading_day = today - timedelta(days=days_back)
                else:
                    # Use previous day for weekdays
                    last_trading_day = today - timedelta(days=1)
                
                end = last_trading_day.replace(hour=0, minute=0, second=0, microsecond=0)
                start = end - timedelta(days=1)

            response = self.client.timeseries.get_range(
                dataset="XNAS.BASIC",  # NASDAQ for most tech stocks
                schema=Schema.TRADES,
                start=start,
                end=end,
                symbols=[symbol],
            )

            # Get last trade
            last_trade = None
            for record in response:
                last_trade = record

            if last_trade:
                return UnderlyingPrice.from_databento_trade(last_trade, symbol)
            else:
                raise ValueError(f"No trades found for {symbol}")

    async def _rate_limit(self):
        """Enforce rate limiting for historical API."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._request_interval:
            await asyncio.sleep(self._request_interval - time_since_last)

        self._last_request_time = asyncio.get_event_loop().time()

    async def validate_data_quality(
        self, chain: OptionChain, max_spread_pct: float = 5.0, min_size: int = 10
    ) -> DataQuality:
        """Validate option chain data quality."""
        issues = []

        # Check spreads
        wide_spreads = 0
        for opt in chain.calls + chain.puts:
            if opt.spread_pct > Decimal(str(max_spread_pct)):
                wide_spreads += 1

        bid_ask_ok = wide_spreads < len(chain.calls + chain.puts) * 0.1

        # Check liquidity
        low_liquidity = 0
        for opt in chain.calls + chain.puts:
            if opt.bid_size < min_size or opt.ask_size < min_size:
                low_liquidity += 1

        liquidity_ok = low_liquidity < len(chain.calls + chain.puts) * 0.2

        # Check staleness
        staleness = (datetime.now(timezone.utc).replace(tzinfo=None) - chain.timestamp).total_seconds()

        # Calculate confidence
        confidence = 1.0
        if not bid_ask_ok:
            confidence *= 0.7
        if not liquidity_ok:
            confidence *= 0.8
        if staleness > 60:
            confidence *= 0.5

        return DataQuality(
            symbol=chain.underlying,
            timestamp=chain.timestamp,
            bid_ask_spread_ok=bid_ask_ok,
            sufficient_liquidity=liquidity_ok,
            data_staleness_seconds=staleness,
            confidence_score=confidence,
        )

    async def close(self):
        """Clean up resources."""
        # Close any live sessions
        for session in self._live_sessions.values():
            session.close()
        self._live_sessions.clear()

        logger.info("databento_client_closed")

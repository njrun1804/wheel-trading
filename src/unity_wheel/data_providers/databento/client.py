"""Databento client with rate limiting and retry logic for wheel strategy.

Implements:
- Connection pooling and session reuse
- Rate limiting (100 req/s for historical, 10 concurrent live)
- Automatic retries with exponential backoff
- Symbol batching for efficiency
- Instrument ID caching for performance
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import databento as db
from databento_dbn import Schema, SType
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config.loader import get_config
from src.unity_wheel.secrets.integration import get_databento_api_key
from src.unity_wheel.utils.data_validator import die
from src.unity_wheel.utils.logging import StructuredLogger

from ..audit_logger import get_audit_logger
from .types import DataQuality, InstrumentDefinition, OptionChain, OptionQuote, UnderlyingPrice

logger = StructuredLogger(logging.getLogger(__name__))


class DatabentoClient:
    """Databento client optimized for wheel strategy data needs."""

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
        # Load rate limits from config
        config = get_config()
        self.MAX_CONCURRENT_LIVE = config.databento.rate_limits.max_concurrent_live
        self.MAX_HISTORICAL_RPS = config.databento.rate_limits.max_historical_rps
        self.MAX_SYMBOLS_PER_REQUEST = config.databento.rate_limits.max_symbols_per_request
        self.MAX_FILE_SIZE_GB = config.databento.rate_limits.max_file_size_gb

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
        self._historical_semaphore = asyncio.Semaphore(
            self.MAX_CONCURRENT_LIVE
        )  # Concurrent requests
        self._last_request_time = 0.0
        self._request_interval = 1.0 / self.MAX_HISTORICAL_RPS

        # Instrument mapping cache
        self._instrument_map: Optional[db.common.symbology.InstrumentMap] = None

        # Initialize audit logger for data tracking
        self.audit_logger = get_audit_logger()
        self._definitions_cache: Dict[str, InstrumentDefinition] = {}

        # Cache successful symbol formats for Unity
        self._symbol_format_cache: Dict[str, Tuple[List[str], SType]] = {}

        # Live session management
        self._live_sessions: Dict[str, db.Live] = {}

        logger.info(
            "databento_client_initialized", extra={"use_cache": use_cache, "cache_dir": cache_dir}
        )

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
            },
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

    async def _get_definitions(
        self, underlying: str, expiration: datetime
    ) -> List[InstrumentDefinition]:
        """Get instrument definitions with caching and retry."""
        config = get_config()
        retry_config = config.data.retry

        @retry(
            stop=stop_after_attempt(retry_config.max_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=retry_config.delays[0] if retry_config.delays else 2,
                max=retry_config.delays[-1] if retry_config.delays else 10,
            ),
            retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        )
        async def _get_with_retry():
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
                },
            )

            # Check cache for successful symbol format
            if underlying in self._symbol_format_cache:
                symbols, stype_in = self._symbol_format_cache[underlying]
                logger.debug(
                    "Using cached symbol format",
                    extra={"underlying": underlying, "symbols": symbols},
                )
                symbol_formats = [(symbols, stype_in)]
            else:
                unity_ticker = get_config().unity.ticker
                # Handle Unity-specific symbol format
                if underlying == unity_ticker:
                    # Unity requires special handling - try multiple formats
                    symbol_formats = [
                        ([f"{unity_ticker}.OPT"], SType.PARENT),  # Standard format first
                        ([f"{unity_ticker}     *"], SType.PARENT),
                        ([unity_ticker], SType.RAW_SYMBOL),
                    ]
                else:
                    symbol_formats = [
                        ([f"{underlying}.OPT"], SType.PARENT),  # Standard format
                    ]

            # Try each format until one works
            last_error = None
            response = None
            for symbols, stype_in in symbol_formats:
                try:
                    logger.debug(
                        "Trying symbol format",
                        extra={"underlying": underlying, "symbols": symbols, "stype": stype_in},
                    )

                    response = self.client.timeseries.get_range(
                        dataset="OPRA.PILLAR",
                        schema=Schema.DEFINITION,
                        start=start,
                        end=end,
                        symbols=symbols,
                        stype_in=stype_in,
                    )

                    # Cache successful format
                    if underlying not in self._symbol_format_cache:
                        self._symbol_format_cache[underlying] = (symbols, stype_in)
                        logger.info(
                            "Cached successful symbol format",
                            extra={"underlying": underlying, "symbols": symbols},
                        )
                    break

                except Exception as e:
                    last_error = e
                    if "subscription" in str(e).lower():
                        # Subscription error - don't retry
                        logger.error(
                            "Databento subscription error",
                            extra={"underlying": underlying, "error": str(e)},
                        )
                        raise ValueError(
                            f"CRITICAL: Databento subscription does not include options data for {underlying}. "
                            f"Cannot proceed without real market data. Please check your subscription."
                        )
                    logger.debug(
                        "Symbol format failed", extra={"symbols": symbols, "error": str(e)}
                    )
                    continue

            if response is None:
                # All formats failed
                if last_error:
                    raise ValueError(
                        f"CRITICAL: Could not retrieve real options data for {underlying}. "
                        f"Cannot proceed without market data. Last error: {last_error}. "
                        f"Try setting DATABENTO_SKIP_VALIDATION=true to skip data validation."
                    )

            # Parse definitions
            definitions = []
            for record in response:
                try:
                    defn = InstrumentDefinition.from_databento(record)
                    if defn.expiration.date() == expiration.date():
                        definitions.append(defn)
                except Exception as e:
                    logger.warning(
                        "definition_parse_error", extra={"error": str(e), "record": record}
                    )

            # Cache results
            if self.use_cache:
                self._definitions_cache[cache_key] = definitions

            logger.info(
                "definitions_fetched",
                extra={
                    "count": len(definitions),
                    "underlying": underlying,
                    "expiration": expiration.isoformat(),
                },
            )

            return definitions

        # Call the retry-wrapped function
        return await _get_with_retry()

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

            # Use TRADES schema instead of deprecated MBP_1 for OPRA
            response = self.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema=Schema.TRADES,  # Use trades instead of deprecated mbp-1
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

                # Use a wider date range to ensure we get data
                end = last_trading_day.replace(hour=23, minute=59, second=59, microsecond=0)
                start = last_trading_day.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) - timedelta(days=5)

            # Unity trades on NASDAQ - use XNAS.BASIC
            response = self.client.timeseries.get_range(
                dataset="XNAS.BASIC",
                schema=Schema.TRADES,
                start=start,
                end=end,
                symbols=[symbol],
            )

            # Get last trade or quote
            last_record = None
            for record in response:
                last_record = record

            if last_record:
                # Check if we got MBP-1 data (from EQUS.MINI fallback)
                if hasattr(last_record, "bid_px") and hasattr(last_record, "ask_px"):
                    # Validate required attributes before accessing
                    if not hasattr(last_record, "ts_event"):
                        die(f"Missing ts_event in databento MBP-1 record for {symbol}")
                    if not hasattr(last_record, "bid_px"):
                        die(f"Missing bid_px in databento MBP-1 record for {symbol}")
                    if not hasattr(last_record, "ask_px"):
                        die(f"Missing ask_px in databento MBP-1 record for {symbol}")

                    # Convert MBP-1 record to UnderlyingPrice
                    from decimal import Decimal

                    bid = Decimal(str(last_record.bid_px / 1e9))  # Convert from fixed-point
                    ask = Decimal(str(last_record.ask_px / 1e9))
                    mid = (bid + ask) / 2

                    price = UnderlyingPrice(
                        symbol=symbol,
                        last_price=mid,
                        bid_price=bid,
                        ask_price=ask,
                        timestamp=datetime.fromtimestamp(
                            last_record.ts_event / 1e9, tz=timezone.utc
                        ),
                    )

                    # Audit log the quote data fetch
                    self.audit_logger.log_data_fetch(
                        source="databento",
                        symbol=symbol,
                        data_type="underlying_quote",
                        data={
                            "bid": float(bid),
                            "ask": float(ask),
                            "mid": float(mid),
                            "timestamp": price.timestamp.isoformat(),
                        },
                        request_params={
                            "dataset": "XNAS.BASIC",
                            "schema": "TRADES",
                            "start": start.isoformat(),
                            "end": end.isoformat(),
                        },
                    )

                    return price
                else:
                    # Standard trade record - validate required attributes
                    if not hasattr(last_record, "ts_event"):
                        die(f"Missing ts_event in databento trade record for {symbol}")
                    if not hasattr(last_record, "price"):
                        die(f"Missing price in databento trade record for {symbol}")

                    price = UnderlyingPrice.from_databento_trade(last_record, symbol)

                    # Audit log the data fetch
                    self.audit_logger.log_data_fetch(
                        source="databento",
                        symbol=symbol,
                        data_type="underlying_price",
                        data={
                            "last_price": float(price.last_price),
                            "timestamp": price.timestamp.isoformat(),
                        },
                        request_params={
                            "dataset": "XNAS.BASIC",
                            "schema": "TRADES",
                            "start": start.isoformat(),
                            "end": end.isoformat(),
                        },
                    )

                    return price
            else:
                die(f"No trades found for {symbol} in databento data")

    async def _rate_limit(self):
        """Enforce rate limiting for historical API."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._request_interval:
            await asyncio.sleep(self._request_interval - time_since_last)

        self._last_request_time = asyncio.get_event_loop().time()

    async def validate_data_quality(
        self, chain: OptionChain, max_spread_pct: float = None, min_size: int = None
    ) -> DataQuality:
        """Validate option chain data quality."""
        config = get_config()

        # Use config values as defaults
        if max_spread_pct is None:
            max_spread_pct = config.data.quality.max_spread_pct
        if min_size is None:
            min_size = config.data.quality.min_quote_size

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
        staleness = (
            datetime.now(timezone.utc).replace(tzinfo=None) - chain.timestamp
        ).total_seconds()

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

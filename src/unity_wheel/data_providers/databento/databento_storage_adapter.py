"""
from __future__ import annotations

Databento storage adapter implementing the documented storage plan.
Integrates with unified storage layer for options data.
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from ..config.loader import get_config
from ....storage import Storage
from ....utils import get_logger

from .types import InstrumentDefinition, OptionChain, OptionQuote
from .validation import DataValidator

logger = get_logger(__name__)


class DatabentoStorageAdapter:
    """Storage adapter for Databento options data with moneyness filtering."""

    def __init__(self, storage: Storage):
        """Initialize with unified storage layer."""
        self.storage = storage
        self.validator = DataValidator()

        # Load configuration
        config = get_config()

        # Storage optimization constants from config
        self.MONEYNESS_RANGE = config.databento.filters.moneyness_range
        self.MAX_EXPIRATIONS = config.databento.filters.max_expirations
        self.INTRADAY_TTL_MINUTES = (
            config.data.cache_ttl.intraday / 60
        )  # Convert seconds to minutes
        self.GREEKS_TTL_MINUTES = config.data.cache_ttl.greeks / 60  # Convert seconds to minutes

        # Track storage metrics
        self._metrics = {
            "options_stored": 0,
            "options_filtered": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def initialize(self):
        """Create Databento-specific tables in DuckDB."""
        # Enhanced option chains table per storage plan
        await self.storage.cache.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS databento_option_chains (
                symbol VARCHAR NOT NULL,
                expiration DATE NOT NULL,
                strike DECIMAL(10,2) NOT NULL,
                option_type VARCHAR(4) NOT NULL,
                bid DECIMAL(10,4),
                ask DECIMAL(10,4),
                mid DECIMAL(10,4),
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility DECIMAL(6,4),
                delta DECIMAL(5,4),
                gamma DECIMAL(5,4),
                theta DECIMAL(5,4),
                vega DECIMAL(5,4),
                rho DECIMAL(5,4),
                timestamp TIMESTAMP NOT NULL,
                spot_price DECIMAL(10,2) NOT NULL,
                moneyness DECIMAL(5,4),  -- For efficient filtering
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, expiration, strike, option_type, timestamp)
            )
        """
        )

        # Create indexes for performance
        await self.storage.cache.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_databento_symbol_expiry
            ON databento_option_chains(symbol, expiration, timestamp)
        """
        )

        await self.storage.cache.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_databento_moneyness
            ON databento_option_chains(symbol, moneyness, expiration)
        """
        )

        await self.storage.cache.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_databento_delta_range
            ON databento_option_chains(symbol, delta, expiration)
        """
        )

        # Wheel candidates table for pre-filtered recommendations
        await self.storage.cache.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wheel_candidates (
                symbol VARCHAR NOT NULL,
                target_delta DECIMAL(5,4) NOT NULL,
                expiration DATE NOT NULL,
                strike DECIMAL(10,2) NOT NULL,
                option_type VARCHAR(4) NOT NULL,
                bid DECIMAL(10,4),
                ask DECIMAL(10,4),
                mid DECIMAL(10,4),
                implied_volatility DECIMAL(6,4),
                delta DECIMAL(5,4),
                expected_return DECIMAL(6,4),
                annualized_return DECIMAL(6,4),
                probability_profit DECIMAL(5,4),
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, target_delta, timestamp)
            )
        """
        )

        logger.info("databento_storage_initialized")

    async def store_option_chain(
        self, chain: OptionChain, definitions: List[InstrumentDefinition], enriched: bool = False
    ) -> bool:
        """Store option chain with moneyness filtering.

        Args:
            chain: Raw option chain from Databento
            definitions: Instrument definitions
            enriched: Whether Greeks have been calculated

        Returns:
            Success status
        """
        try:
            # Apply moneyness filter to reduce storage by 80%
            filtered_chain = self._filter_by_moneyness(chain)

            # Create definition lookup
            def_map = {d.instrument_id: d for d in definitions}

            # Prepare batch insert data
            records = []
            spot = float(chain.spot_price)

            # Process all options (calls and puts)
            for opt in filtered_chain.calls + filtered_chain.puts:
                if opt.instrument_id not in def_map:
                    continue

                defn = def_map[opt.instrument_id]
                strike = float(defn.strike_price)
                moneyness = strike / spot

                record = {
                    "symbol": chain.underlying,
                    "expiration": defn.expiration.date(),
                    "strike": strike,
                    "option_type": defn.option_type.value,
                    "bid": float(opt.bid_price),
                    "ask": float(opt.ask_price),
                    "mid": float(opt.mid_price),
                    "volume": opt.volume or 0,
                    "open_interest": 0,  # Would need additional data
                    "implied_volatility": None,  # Calculated separately
                    "delta": None,
                    "gamma": None,
                    "theta": None,
                    "vega": None,
                    "rho": None,
                    "timestamp": chain.timestamp,
                    "spot_price": spot,
                    "moneyness": moneyness,
                }

                # Add Greeks if available
                if enriched and hasattr(opt, "greeks"):
                    record.update(
                        {
                            "delta": opt.greeks.get("delta"),
                            "gamma": opt.greeks.get("gamma"),
                            "theta": opt.greeks.get("theta"),
                            "vega": opt.greeks.get("vega"),
                            "rho": opt.greeks.get("rho"),
                            "implied_volatility": opt.greeks.get("iv"),
                        }
                    )

                records.append(record)

            # Batch insert
            if records:
                await self._batch_insert_options(records)
                self._metrics["options_stored"] += len(records)

                logger.info(
                    "options_stored",
                    symbol=chain.underlying,
                    count=len(records),
                    filtered_pct=len(records) / (len(chain.calls) + len(chain.puts)),
                )

            # Clean up old data
            await self._enforce_retention_policy()

            return True

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("storage_error", error=str(e))
            return False

    def _filter_by_moneyness(self, chain: OptionChain) -> OptionChain:
        """Filter options by moneyness to reduce storage by 80%.

        Only keeps options within MONEYNESS_RANGE (20%) of spot price.
        """
        spot = float(chain.spot_price)
        min_strike = spot * (1 - self.MONEYNESS_RANGE)
        max_strike = spot * (1 + self.MONEYNESS_RANGE)

        # Count for metrics
        original_count = len(chain.calls) + len(chain.puts)

        # Filter calls
        filtered_calls = []
        for call in chain.calls:
            # Need to get strike from definition - for now approximate
            if hasattr(call, "strike_price"):
                strike = float(call.strike_price)
                if min_strike <= strike <= max_strike:
                    filtered_calls.append(call)

        # Filter puts
        filtered_puts = []
        for put in chain.puts:
            if hasattr(put, "strike_price"):
                strike = float(put.strike_price)
                if min_strike <= strike <= max_strike:
                    filtered_puts.append(put)

        # Update metrics
        filtered_count = len(filtered_calls) + len(filtered_puts)
        self._metrics["options_filtered"] += original_count - filtered_count

        # Return filtered chain
        chain.calls = filtered_calls
        chain.puts = filtered_puts

        logger.debug(
            "moneyness_filter_applied",
            original=original_count,
            filtered=filtered_count,
            reduction_pct=(1 - filtered_count / original_count) if original_count > 0 else 0,
        )

        return chain

    async def get_or_fetch_option_chain(
        self,
        symbol: str,
        expiration: datetime,
        fetch_func: Optional[Any] = None,
        max_age_minutes: int = None,
    ) -> Optional[Dict[str, Any]]:
        """Get option chain using get_or_fetch pattern.

        Args:
            symbol: Underlying symbol
            expiration: Option expiration
            fetch_func: Function to fetch fresh data if cache miss
            max_age_minutes: Override default TTL

        Returns:
            Option chain data or None
        """
        max_age = max_age_minutes or self.INTRADAY_TTL_MINUTES

        # 1. Check cache first
        cached = await self._get_cached_chain(symbol, expiration, max_age)
        if cached:
            self._metrics["cache_hits"] += 1
            logger.debug("cache_hit", symbol=symbol, expiration=expiration)
            return cached

        self._metrics["cache_misses"] += 1

        # 2. Cache miss - fetch fresh data if function provided
        if fetch_func:
            try:
                fresh_data = await fetch_func(symbol, expiration)

                # 3. Validate and store
                if fresh_data:
                    # Store in our format
                    await self.store_option_chain(
                        fresh_data["chain"],
                        fresh_data["definitions"],
                        fresh_data.get("enriched", False),
                    )

                    # Return in expected format
                    return self._format_chain_response(fresh_data)

            except (ValueError, KeyError, AttributeError) as e:
                logger.error("fetch_error", error=str(e))

        return None

    async def _get_cached_chain(
        self, symbol: str, expiration: datetime, max_age_minutes: int
    ) -> Optional[Dict[str, Any]]:
        """Get cached option chain from DuckDB."""
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)

        # Query with moneyness range for efficiency
        query = """
            SELECT strike, option_type, bid, ask, mid, volume,
                   implied_volatility, delta, gamma, theta, vega,
                   timestamp, spot_price
            FROM databento_option_chains
            WHERE symbol = ?
            AND expiration = ?
            AND timestamp > ?
            AND moneyness BETWEEN ? AND ?
            ORDER BY strike, option_type
        """

        result = await self.storage.cache.conn.execute(
            query,
            [symbol, expiration.date(), cutoff, 1 - self.MONEYNESS_RANGE, 1 + self.MONEYNESS_RANGE],
        ).fetchall()

        if not result:
            return None

        # Format as expected by application
        calls = []
        puts = []
        spot_price = None
        timestamp = None

        for row in result:
            strike, opt_type, bid, ask, mid, volume, iv, delta, gamma, theta, vega, ts, spot = row

            option_data = {
                "strike": float(strike),
                "bid": float(bid) if bid else 0,
                "ask": float(ask) if ask else 0,
                "mid": float(mid) if mid else 0,
                "volume": volume or 0,
                "iv": float(iv) if iv else None,
                "delta": float(delta) if delta else None,
                "gamma": float(gamma) if gamma else None,
                "theta": float(theta) if theta else None,
                "vega": float(vega) if vega else None,
            }

            if opt_type == "CALL":
                calls.append(option_data)
            else:
                puts.append(option_data)

            spot_price = float(spot)
            timestamp = ts

        return {
            "symbol": symbol,
            "expiration": expiration.isoformat(),
            "spot_price": spot_price,
            "timestamp": timestamp.isoformat() if timestamp else None,
            "calls": calls,
            "puts": puts,
            "cached": True,
        }

    async def store_wheel_candidates(
        self, symbol: str, target_delta: float, candidates: List[Dict[str, Any]]
    ):
        """Store pre-filtered wheel candidates."""
        records = []

        for cand in candidates:
            records.append(
                {
                    "symbol": symbol,
                    "target_delta": target_delta,
                    "expiration": cand["expiration"],
                    "strike": cand["strike"],
                    "option_type": "PUT",  # Wheel uses puts
                    "bid": cand["bid"],
                    "ask": cand["ask"],
                    "mid": cand["mid"],
                    "implied_volatility": cand.get("iv"),
                    "delta": cand["delta"],
                    "expected_return": cand["expected_return"],
                    "annualized_return": cand["annualized_return"],
                    "probability_profit": cand.get("prob_profit", 0),
                    "timestamp": datetime.utcnow(),
                }
            )

        if records:
            # Batch insert
            placeholders = ",".join(["(?,?,?,?,?,?,?,?,?,?,?,?,?,?)"] * len(records))
            values = []
            for r in records:
                values.extend(
                    [
                        r["symbol"],
                        r["target_delta"],
                        r["expiration"],
                        r["strike"],
                        r["option_type"],
                        r["bid"],
                        r["ask"],
                        r["mid"],
                        r["implied_volatility"],
                        r["delta"],
                        r["expected_return"],
                        r["annualized_return"],
                        r["probability_profit"],
                        r["timestamp"],
                    ]
                )

            await self.storage.cache.conn.execute(
                f"INSERT OR REPLACE INTO wheel_candidates VALUES {placeholders}", values
            )

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        # Get table sizes
        stats = await self.storage.cache.conn.execute(
            """
            SELECT
                COUNT(*) as total_options,
                COUNT(DISTINCT symbol) as symbols,
                COUNT(DISTINCT expiration) as expirations,
                MIN(timestamp) as oldest_data,
                MAX(timestamp) as newest_data
            FROM databento_option_chains
        """
        ).fetchone()

        total, symbols, expirations, oldest, newest = stats

        # Get database size
        db_size = await self.storage.cache.conn.execute(
            """
            SELECT page_count * page_size / 1024.0 / 1024.0 as size_mb
            FROM pragma_database_page_count(), pragma_page_size()
        """
        ).fetchone()

        return {
            "total_options": total,
            "unique_symbols": symbols,
            "unique_expirations": expirations,
            "oldest_data": oldest,
            "newest_data": newest,
            "db_size_mb": db_size[0] if db_size else 0,
            "metrics": self._metrics,
            "cache_hit_rate": (
                self._metrics["cache_hits"]
                / (self._metrics["cache_hits"] + self._metrics["cache_misses"])
                if self._metrics["cache_misses"] > 0
                else 0
            ),
        }

    async def _batch_insert_options(self, records: List[Dict]):
        """Efficiently insert multiple option records."""
        if not records:
            return

        # Build insert query
        columns = list(records[0].keys())
        placeholders = ",".join(["?" for _ in columns])

        query = f"""
            INSERT OR REPLACE INTO databento_option_chains
            ({','.join(columns)})
            VALUES ({placeholders})
        """

        # Execute in batches
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            # Prepare values
            values = []
            for record in batch:
                values.append([record[col] for col in columns])

            # Execute batch
            await self.storage.cache.conn.executemany(query, values)

    async def _enforce_retention_policy(self):
        """Clean up old data according to retention policy."""
        # Remove data older than 30 days
        cutoff = datetime.utcnow() - timedelta(days=30)

        deleted = await self.storage.cache.conn.execute(
            """
            DELETE FROM databento_option_chains
            WHERE created_at < ?
        """,
            [cutoff],
        )

        if deleted.rowcount > 0:
            logger.info("old_data_cleaned", rows_deleted=deleted.rowcount)

        # Also clean expired options
        today = datetime.utcnow().date()
        await self.storage.cache.conn.execute(
            """
            DELETE FROM databento_option_chains
            WHERE expiration < ?
        """,
            [today],
        )

    def _format_chain_response(self, data: Dict) -> Dict[str, Any]:
        """Format chain data for application use."""
        chain = data["chain"]
        definitions = data["definitions"]

        # Create definition lookup
        def_map = {d.instrument_id: d for d in definitions}

        # Format options
        calls = []
        puts = []

        for opt in chain.calls:
            if opt.instrument_id in def_map:
                defn = def_map[opt.instrument_id]
                calls.append(
                    {
                        "strike": float(defn.strike_price),
                        "bid": float(opt.bid_price),
                        "ask": float(opt.ask_price),
                        "mid": float(opt.mid_price),
                        "volume": opt.volume or 0,
                    }
                )

        for opt in chain.puts:
            if opt.instrument_id in def_map:
                defn = def_map[opt.instrument_id]
                puts.append(
                    {
                        "strike": float(defn.strike_price),
                        "bid": float(opt.bid_price),
                        "ask": float(opt.ask_price),
                        "mid": float(opt.mid_price),
                        "volume": opt.volume or 0,
                    }
                )

        return {
            "symbol": chain.underlying,
            "expiration": chain.expiration.isoformat() if hasattr(chain, "expiration") else None,
            "spot_price": float(chain.spot_price),
            "timestamp": chain.timestamp.isoformat(),
            "calls": sorted(calls, key=lambda x: x["strike"]),
            "puts": sorted(puts, key=lambda x: x["strike"]),
            "cached": False,
        }
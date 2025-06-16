"""
ML-Enhanced Data Collector for Databento
Captures comprehensive market snapshots for future strategy development
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, time

import duckdb
import pandas as pd

from src.config.loader import get_config
from unity_wheel.utils import get_logger

from .client import DatabentoClient

logger = get_logger(__name__)


@dataclass
class MarketSnapshot:
    """Complete market snapshot for ML training"""

    timestamp: datetime
    spot_price: float
    volume: int

    # Market microstructure
    bid_ask_spread: float
    order_imbalance: float | None = None

    # Options surface
    atm_iv: float
    iv_skew: float  # 25-delta put IV - 25-delta call IV
    term_structure_slope: float  # 60-day IV - 30-day IV

    # Market regime indicators
    intraday_range: float  # (high - low) / open
    volume_ratio: float  # current volume / 20-day avg

    # Greeks aggregates
    total_gamma: float  # Market maker gamma exposure
    total_vanna: float  # Cross-Greeks


class MLEnhancedDataCollector:
    """
    Collects comprehensive market data for ML strategy development
    Flexible design to support future strategies
    """

    def __init__(self, db_path: str = "data/wheel_trading_optimized.duckdb"):
        self.db_path = db_path
        self.client = DatabentoClient()
        self.config = get_config()

        # Snapshot frequencies based on market conditions
        self.snapshot_rules = {
            "regular_hours": {
                "start": time(9, 30),
                "end": time(16, 0),
                "frequency_minutes": 30,  # Every 30 minutes
            },
            "near_expiry": {"frequency_minutes": 5},  # For positions < 2 DTE
            "high_volatility": {"frequency_minutes": 15},  # When VIX > 30
            "around_events": {"frequency_minutes": 5},  # ±2 hours of earnings/Fed
        }

        # Data collection parameters
        self.collection_params = {
            "moneyness_range": 0.50,  # ±50% for full surface
            "min_strikes": 20,  # Minimum strikes per expiration
            "expirations": 10,  # Collect more expirations
            "include_weeklies": True,  # For term structure
            "collect_trades": True,  # For microstructure
            "book_depth": 5,  # Levels of order book
        }

    async def collect_snapshot(self, symbol: str = "U") -> dict:
        """Collect comprehensive market snapshot"""
        logger.info(f"📸 Collecting ML snapshot for {symbol}")

        snapshot_data = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "collection_type": self._determine_collection_type(),
        }

        try:
            # 1. Get underlying data with microstructure
            underlying_data = await self._collect_underlying_data(symbol)
            snapshot_data.update(underlying_data)

            # 2. Get full option surface
            options_data = await self._collect_option_surface(symbol)
            snapshot_data.update(options_data)

            # 3. Calculate market indicators
            market_indicators = self._calculate_market_indicators(
                underlying_data, options_data
            )
            snapshot_data.update(market_indicators)

            # 4. Store snapshot
            await self._store_snapshot(snapshot_data)

            logger.info("✅ Snapshot collected successfully")
            return snapshot_data

        except Exception as e:
            logger.error(f"❌ Snapshot collection failed: {e}")
            raise

    async def _collect_underlying_data(self, symbol: str) -> dict:
        """Collect underlying price and microstructure data"""
        # In production, this would query Databento
        # For now, return structured data

        return {
            "spot_price": 10.5,  # Would come from Databento
            "bid": 10.49,
            "ask": 10.51,
            "bid_size": 1000,
            "ask_size": 1200,
            "last_trade_size": 100,
            "volume": 2500000,
            "vwap": 10.48,
            "high": 10.75,
            "low": 10.25,
            "open": 10.40,
            # Microstructure metrics
            "bid_ask_spread": 0.02,
            "effective_spread": 0.018,  # Trade-weighted
            "order_imbalance": -0.05,  # (bid_size - ask_size) / (bid_size + ask_size)
            "quote_count": 15000,  # Number of quote updates
            "trade_count": 8000,
        }

    async def _collect_option_surface(self, symbol: str) -> dict:
        """Collect comprehensive option surface data"""

        # Get option chains from Databento
        chains = await self.client.get_option_chains(symbol)

        if not chains:
            logger.warning("No option chains available")
            return {}

        # Process into surface metrics
        surface_data = {"option_chains": [], "surface_metrics": {}}

        # Collect all strikes within extended range
        all_options = []
        for chain in chains:
            for option in chain.options:
                moneyness = option.strike / chain.spot_price

                # Collect wider range for ML
                if 0.5 <= moneyness <= 1.5:
                    option_data = {
                        "strike": option.strike,
                        "expiration": option.expiration,
                        "option_type": option.option_type,
                        "bid": option.bid,
                        "ask": option.ask,
                        "volume": option.volume,
                        "open_interest": option.open_interest,
                        "implied_volatility": option.implied_volatility,
                        "delta": option.delta,
                        "gamma": option.gamma,
                        "vega": option.vega,
                        "theta": option.theta,
                        "moneyness": moneyness,
                        "days_to_expiry": (
                            option.expiration - datetime.now().date()
                        ).days,
                        # Additional ML features
                        "bid_ask_spread_pct": (option.ask - option.bid) / option.ask
                        if option.ask > 0
                        else None,
                        "volume_oi_ratio": option.volume / option.open_interest
                        if option.open_interest > 0
                        else 0,
                    }
                    all_options.append(option_data)

        surface_data["option_chains"] = all_options

        # Calculate surface-level metrics
        if all_options:
            surface_data["surface_metrics"] = self._calculate_surface_metrics(
                all_options
            )

        return surface_data

    def _calculate_surface_metrics(self, options: list[dict]) -> dict:
        """Calculate IV surface metrics for ML features"""

        df = pd.DataFrame(options)

        metrics = {}

        # ATM IV (closest to 100% moneyness)
        atm_options = df[df["moneyness"].between(0.98, 1.02)]
        if not atm_options.empty:
            metrics["atm_iv"] = atm_options["implied_volatility"].mean()

        # IV Skew (25-delta put IV - 25-delta call IV)
        put_25d = df[(df["option_type"] == "PUT") & df["delta"].between(-0.30, -0.20)]
        call_25d = df[(df["option_type"] == "CALL") & df["delta"].between(0.20, 0.30)]

        if not put_25d.empty and not call_25d.empty:
            metrics["iv_skew"] = (
                put_25d["implied_volatility"].mean()
                - call_25d["implied_volatility"].mean()
            )

        # Term structure slope
        if "days_to_expiry" in df.columns:
            short_term = df[df["days_to_expiry"].between(20, 40)]
            long_term = df[df["days_to_expiry"].between(50, 70)]

            if not short_term.empty and not long_term.empty:
                metrics["term_structure_slope"] = (
                    long_term["implied_volatility"].mean()
                    - short_term["implied_volatility"].mean()
                )

        # Put/Call ratio
        puts = df[df["option_type"] == "PUT"]
        calls = df[df["option_type"] == "CALL"]

        if not puts.empty and not calls.empty:
            metrics["put_call_volume_ratio"] = (
                puts["volume"].sum() / calls["volume"].sum()
            )
            metrics["put_call_oi_ratio"] = (
                puts["open_interest"].sum() / calls["open_interest"].sum()
            )

        # Gamma exposure (market maker positioning)
        # Simplified: Sum of gamma * open_interest * contract_multiplier
        metrics["total_gamma"] = (df["gamma"] * df["open_interest"] * 100).sum()

        # Average bid-ask spread by moneyness bucket
        for bucket, (low, high) in [
            ("otm", (0.7, 0.9)),
            ("atm", (0.9, 1.1)),
            ("itm", (1.1, 1.3)),
        ]:
            bucket_options = df[df["moneyness"].between(low, high)]
            if not bucket_options.empty:
                metrics[f"avg_spread_{bucket}"] = bucket_options[
                    "bid_ask_spread_pct"
                ].mean()

        return metrics

    def _calculate_market_indicators(self, underlying: dict, options: dict) -> dict:
        """Calculate market regime indicators"""

        indicators = {}

        # Price-based indicators
        if all(k in underlying for k in ["high", "low", "open"]):
            indicators["intraday_range"] = (
                underlying["high"] - underlying["low"]
            ) / underlying["open"]
            indicators["intraday_trend"] = (
                underlying["spot_price"] - underlying["open"]
            ) / underlying["open"]

        # Microstructure health
        if "quote_count" in underlying and "trade_count" in underlying:
            indicators["quote_to_trade_ratio"] = (
                underlying["quote_count"] / underlying["trade_count"]
            )

        # Options-based indicators
        if "surface_metrics" in options:
            metrics = options["surface_metrics"]

            # Market stress indicators
            if "atm_iv" in metrics:
                indicators["iv_level"] = metrics["atm_iv"]

                # IV regime (would compare to historical percentiles)
                if metrics["atm_iv"] < 0.4:
                    indicators["iv_regime"] = "low"
                elif metrics["atm_iv"] < 0.6:
                    indicators["iv_regime"] = "normal"
                else:
                    indicators["iv_regime"] = "high"

            # Skew regime
            if "iv_skew" in metrics:
                indicators["skew_level"] = metrics["iv_skew"]
                indicators["skew_extreme"] = abs(metrics["iv_skew"]) > 0.1

        return {"market_indicators": indicators}

    async def _store_snapshot(self, snapshot: dict):
        """Store comprehensive snapshot in database"""

        conn = duckdb.connect(self.db_path)

        try:
            # Store in new ML snapshots table
            self._ensure_ml_tables_exist(conn)

            # 1. Store market snapshot
            conn.execute(
                """
                INSERT INTO ml_data.market_snapshots (
                    symbol, timestamp, collection_type,
                    spot_price, bid, ask, volume, vwap,
                    bid_ask_spread, order_imbalance,
                    iv_regime, skew_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    snapshot.get("symbol"),
                    snapshot.get("timestamp"),
                    snapshot.get("collection_type"),
                    snapshot.get("spot_price"),
                    snapshot.get("bid"),
                    snapshot.get("ask"),
                    snapshot.get("volume"),
                    snapshot.get("vwap"),
                    snapshot.get("bid_ask_spread"),
                    snapshot.get("order_imbalance"),
                    snapshot.get("market_indicators", {}).get("iv_regime"),
                    snapshot.get("market_indicators", {}).get("skew_level"),
                ],
            )

            # 2. Store options surface (in batches)
            if "option_chains" in snapshot:
                for option in snapshot["option_chains"]:
                    conn.execute(
                        """
                        INSERT INTO ml_data.option_snapshots (
                            symbol, timestamp, strike, expiration, option_type,
                            bid, ask, volume, open_interest,
                            implied_volatility, delta, gamma, vega, theta,
                            moneyness, days_to_expiry,
                            bid_ask_spread_pct, volume_oi_ratio
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        [
                            snapshot["symbol"],
                            snapshot["timestamp"],
                            option["strike"],
                            option["expiration"],
                            option["option_type"],
                            option["bid"],
                            option["ask"],
                            option["volume"],
                            option["open_interest"],
                            option["implied_volatility"],
                            option["delta"],
                            option["gamma"],
                            option["vega"],
                            option["theta"],
                            option["moneyness"],
                            option["days_to_expiry"],
                            option["bid_ask_spread_pct"],
                            option["volume_oi_ratio"],
                        ],
                    )

            # 3. Store surface metrics
            if "surface_metrics" in snapshot:
                metrics = snapshot["surface_metrics"]
                conn.execute(
                    """
                    INSERT INTO ml_data.surface_metrics (
                        symbol, timestamp,
                        atm_iv, iv_skew, term_structure_slope,
                        put_call_volume_ratio, put_call_oi_ratio,
                        total_gamma
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        snapshot["symbol"],
                        snapshot["timestamp"],
                        metrics.get("atm_iv"),
                        metrics.get("iv_skew"),
                        metrics.get("term_structure_slope"),
                        metrics.get("put_call_volume_ratio"),
                        metrics.get("put_call_oi_ratio"),
                        metrics.get("total_gamma"),
                    ],
                )

            conn.commit()

        finally:
            conn.close()

    def _ensure_ml_tables_exist(self, conn):
        """Create ML data tables if they don't exist"""

        # Create ML data schema
        conn.execute("CREATE SCHEMA IF NOT EXISTS ml_data")

        # Market snapshots table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_data.market_snapshots (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                collection_type VARCHAR,
                spot_price DECIMAL(10,2),
                bid DECIMAL(10,2),
                ask DECIMAL(10,2),
                volume BIGINT,
                vwap DECIMAL(10,2),
                bid_ask_spread DECIMAL(10,4),
                order_imbalance DECIMAL(6,4),
                iv_regime VARCHAR,
                skew_level DECIMAL(6,4),
                PRIMARY KEY (symbol, timestamp)
            )
        """
        )

        # Options snapshots table (stores all options, not just filtered)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_data.option_snapshots (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                strike DECIMAL(10,2) NOT NULL,
                expiration DATE NOT NULL,
                option_type VARCHAR(4) NOT NULL,
                bid DECIMAL(10,4),
                ask DECIMAL(10,4),
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility DECIMAL(8,6),
                delta DECIMAL(6,4),
                gamma DECIMAL(8,6),
                vega DECIMAL(10,4),
                theta DECIMAL(10,4),
                moneyness DECIMAL(6,4),
                days_to_expiry INTEGER,
                bid_ask_spread_pct DECIMAL(6,4),
                volume_oi_ratio DECIMAL(10,4),
                INDEX idx_ml_snapshot (symbol, timestamp),
                INDEX idx_ml_moneyness (symbol, moneyness, timestamp)
            )
        """
        )

        # Surface metrics table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_data.surface_metrics (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                atm_iv DECIMAL(8,6),
                iv_skew DECIMAL(8,6),
                term_structure_slope DECIMAL(8,6),
                put_call_volume_ratio DECIMAL(10,4),
                put_call_oi_ratio DECIMAL(10,4),
                total_gamma DECIMAL(12,2),
                avg_spread_otm DECIMAL(6,4),
                avg_spread_atm DECIMAL(6,4),
                avg_spread_itm DECIMAL(6,4),
                PRIMARY KEY (symbol, timestamp)
            )
        """
        )

    def _determine_collection_type(self) -> str:
        """Determine what type of collection this is"""

        now = datetime.now()
        current_time = now.time()

        # Check if during regular hours
        if (
            self.snapshot_rules["regular_hours"]["start"]
            <= current_time
            <= self.snapshot_rules["regular_hours"]["end"]
        ):
            # Check for special conditions
            # In production, would check:
            # - VIX level for high volatility
            # - Days to expiry for positions
            # - Proximity to events

            return "regular"

        return "after_hours"

    async def run_collection_schedule(self):
        """Run data collection on schedule"""

        while True:
            try:
                collection_type = self._determine_collection_type()

                if collection_type != "after_hours":
                    await self.collect_snapshot()

                # Determine next collection time
                if collection_type == "regular":
                    wait_minutes = self.snapshot_rules["regular_hours"][
                        "frequency_minutes"
                    ]
                else:
                    # Wait until market opens
                    wait_minutes = 60

                await asyncio.sleep(wait_minutes * 60)

            except Exception as e:
                logger.error(f"Collection error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

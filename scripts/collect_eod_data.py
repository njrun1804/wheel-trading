#!/usr/bin/env python3
"""
EOD Data Collection Script for ML-Enhanced Wheel Trading
Collects comprehensive options and market data at end of day
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct imports - avoid circular dependency issues
import aiohttp
import databento as db
import duckdb

from src.unity_wheel.secrets.manager import SecretManager

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EODDataCollector:
    """Collects end-of-day data for ML-enhanced wheel trading"""

    def __init__(self):
        self.db_path = "data/wheel_trading_optimized.duckdb"

        # Get API keys from SecretManager
        secret_mgr = SecretManager()
        self.databento_key = secret_mgr.get_secret("databento_api_key")
        self.fred_key = secret_mgr.get_secret("ofred_api_key") or secret_mgr.get_secret(
            "fred_api_key"
        )

        # Fallback to environment
        if not self.databento_key:
            self.databento_key = os.environ.get("DATABENTO_API_KEY")
        if not self.fred_key:
            self.fred_key = os.environ.get("FRED_API_KEY") or os.environ.get(
                "OFRED_API_KEY"
            )

        if not self.databento_key:
            raise ValueError(
                "Databento API key not found in SecretManager or environment"
            )
        if not self.fred_key:
            raise ValueError("FRED API key not found in SecretManager or environment")

        self.databento_client = db.Historical(self.databento_key)
        self.conn = duckdb.connect(self.db_path)

    async def collect_unity_options(self):
        """Collect Unity options data from Databento"""
        logger.info("Collecting Unity options data...")

        try:
            # Get the last trading day
            today = datetime.now()
            if today.weekday() == 0:  # Monday
                data_date = today - timedelta(days=3)  # Friday
            elif today.weekday() == 6:  # Sunday
                data_date = today - timedelta(days=2)  # Friday
            else:
                data_date = today - timedelta(days=1)

            logger.info(f"Fetching data for {data_date.date()}")

            # Get option definitions
            definitions = self.databento_client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                symbols=["U"],  # Unity
                stype_in="parent",
                schema="definition",
                start=data_date.strftime("%Y-%m-%d"),
                end=(data_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            )

            # Convert to dataframe
            def_df = definitions.to_df()

            if def_df.empty:
                logger.warning("No Unity option definitions found")
                return 0

            logger.info(f"Found {len(def_df)} Unity option definitions")

            # Get unique instruments
            instruments = def_df["instrument_id"].unique().tolist()

            # Get quotes for these instruments
            quotes = self.databento_client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                symbols=instruments[:100],  # Limit for testing
                stype_in="instrument_id",
                schema="mbp-1",  # Market by price
                start=data_date.replace(hour=15, minute=45),  # Near close
                end=data_date.replace(hour=16, minute=15),
            )

            quote_df = quotes.to_df()

            if quote_df.empty:
                logger.warning("No quotes found")
                return 0

            # Process and store data
            stored_count = self._store_options_data(def_df, quote_df, data_date)
            logger.info(f"Stored {stored_count} Unity options")

            return stored_count

        except Exception as e:
            logger.error(f"Failed to collect Unity options: {e}")
            return 0

    def _store_options_data(self, definitions, quotes, data_date):
        """Store options data in optimized database"""

        # Merge definitions with quotes
        merged = pd.merge(
            quotes,
            definitions[
                ["instrument_id", "strike_price", "expiration", "instrument_class"]
            ],
            on="instrument_id",
            how="inner",
        )

        if merged.empty:
            return 0

        # Calculate additional fields
        merged["option_type"] = merged["instrument_class"].map({"C": "C", "P": "P"})
        merged["year_month"] = (
            merged["expiration"].dt.year * 100 + merged["expiration"].dt.month
        )

        # Get bid/ask from levels
        merged["bid"] = merged["levels"].apply(
            lambda x: x[0]["bid_px"] / 1e9 if x else 0
        )
        merged["ask"] = merged["levels"].apply(
            lambda x: x[0]["ask_px"] / 1e9 if x else 0
        )
        merged["bid_size"] = merged["levels"].apply(
            lambda x: x[0]["bid_sz"] if x else 0
        )
        merged["ask_size"] = merged["levels"].apply(
            lambda x: x[0]["ask_sz"] if x else 0
        )

        # Prepare for insertion
        records = []
        for _, row in merged.iterrows():
            if row["bid"] > 0 and row["ask"] > 0:  # Valid quotes only
                records.append(
                    {
                        "symbol": "U",
                        "expiration": row["expiration"].date(),
                        "strike": float(row["strike_price"])
                        / 1000,  # Convert from scaled
                        "option_type": row["option_type"],
                        "bid": float(row["bid"]),
                        "ask": float(row["ask"]),
                        "volume": 0,  # Would need trades data
                        "open_interest": 0,  # Would need separate query
                        "timestamp": row["ts_recv"],
                        "year_month": int(row["year_month"]),
                    }
                )

        if records:
            # Insert into database
            pd.DataFrame(records)

            # Use INSERT OR REPLACE to handle duplicates
            self.conn.execute(
                """
                INSERT OR REPLACE INTO options.contracts 
                (symbol, expiration, strike, option_type, bid, ask, volume, 
                 open_interest, timestamp, year_month)
                SELECT * FROM df
            """
            )

            logger.info(f"Inserted {len(records)} option records")

        return len(records)

    async def collect_unity_stock_data(self):
        """Collect Unity stock price data"""
        logger.info("Collecting Unity stock data...")

        try:
            # Get stock trades
            today = datetime.now()
            start_date = today - timedelta(days=1)

            trades = self.databento_client.timeseries.get_range(
                dataset="EQUS.MINI",  # Composite NBBO
                symbols=["U"],
                stype_in="raw_symbol",
                schema="trades",
                start=start_date.strftime("%Y-%m-%d"),
                end=today.strftime("%Y-%m-%d"),
            )

            trade_df = trades.to_df()

            if not trade_df.empty:
                # Get OHLC
                ohlc = trade_df.groupby(trade_df["ts_event"].dt.date).agg(
                    {"price": ["first", "max", "min", "last"], "size": "sum"}
                )

                for date, row in ohlc.iterrows():
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO market.price_data
                        (symbol, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        [
                            "U",
                            date,
                            float(row["price"]["first"]) / 1e9,
                            float(row["price"]["max"]) / 1e9,
                            float(row["price"]["min"]) / 1e9,
                            float(row["price"]["last"]) / 1e9,
                            int(row["size"]["sum"]),
                        ],
                    )

                logger.info(f"Updated Unity stock data for {len(ohlc)} days")
                return len(ohlc)

        except Exception as e:
            logger.error(f"Failed to collect stock data: {e}")

        return 0

    async def collect_fred_data(self):
        """Collect economic indicators from FRED"""
        logger.info("Collecting FRED economic data...")

        series_list = [
            ("VIXCLS", "VIX"),
            ("DGS10", "10Y Treasury"),
            ("DFF", "Fed Funds Rate"),
            ("TEDRATE", "TED Spread"),
            ("BAMLH0A0HYM2", "High Yield Spread"),
        ]

        base_url = "https://api.stlouisfed.org/fred/series/observations"

        async with aiohttp.ClientSession() as session:
            for series_id, name in series_list:
                try:
                    params = {
                        "series_id": series_id,
                        "api_key": self.fred_key,
                        "file_type": "json",
                        "limit": 30,
                        "sort_order": "desc",
                    }

                    async with session.get(base_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            observations = data.get("observations", [])

                            for obs in observations:
                                if obs["value"] != ".":  # Skip missing values
                                    self.conn.execute(
                                        """
                                        INSERT OR REPLACE INTO fred_observations
                                        (series_id, observation_date, value)
                                        VALUES (?, ?, ?)
                                    """,
                                        [series_id, obs["date"], float(obs["value"])],
                                    )

                            logger.info(
                                f"Updated {name}: {len(observations)} observations"
                            )

                except Exception as e:
                    logger.error(f"Failed to fetch {name}: {e}")

    async def calculate_ml_features(self):
        """Calculate ML features from collected data"""
        logger.info("Calculating ML features...")

        try:
            # Get latest VIX
            vix = self.conn.execute(
                """
                SELECT value FROM fred_observations 
                WHERE series_id = 'VIXCLS' 
                ORDER BY observation_date DESC 
                LIMIT 1
            """
            ).fetchone()

            # Get latest Unity price
            unity_price = self.conn.execute(
                """
                SELECT close FROM market.price_data 
                WHERE symbol = 'U' 
                ORDER BY date DESC 
                LIMIT 1
            """
            ).fetchone()

            if vix and unity_price:
                # Determine market regime
                vix_level = float(vix[0])
                if vix_level < 15:
                    regime = "low_volatility"
                elif vix_level < 25:
                    regime = "normal"
                elif vix_level < 35:
                    regime = "volatile"
                else:
                    regime = "stressed"

                # Store ML features
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO analytics.ml_features
                    (symbol, feature_date, vix_level, market_regime, 
                     historical_volatility, iv_rank, iv_percentile)
                    VALUES (?, CURRENT_DATE, ?, ?, NULL, NULL, NULL)
                """,
                    ["U", vix_level, regime],
                )

                logger.info(
                    f"Stored ML features - VIX: {vix_level:.2f}, Regime: {regime}"
                )

                # Refresh materialized view
                self.conn.execute(
                    "REFRESH MATERIALIZED VIEW analytics.wheel_opportunities_mv"
                )
                logger.info("Refreshed wheel opportunities view")

        except Exception as e:
            logger.error(f"Failed to calculate ML features: {e}")

    async def main(self):
        """Run all EOD collection tasks"""
        logger.info("=" * 60)
        logger.info("Starting EOD Data Collection")
        logger.info(f"Time: {datetime.now()}")
        logger.info("=" * 60)

        # Collect all data
        options_count = await self.collect_unity_options()
        stock_count = await self.collect_unity_stock_data()
        await self.collect_fred_data()
        await self.calculate_ml_features()

        # Summary
        logger.info("=" * 60)
        logger.info("EOD Collection Complete")
        logger.info(f"Options collected: {options_count}")
        logger.info(f"Stock days updated: {stock_count}")
        logger.info(f"Database: {self.db_path}")
        logger.info("=" * 60)

        # Close connections
        self.conn.close()


if __name__ == "__main__":
    collector = EODDataCollector()
    asyncio.run(collector.main())
